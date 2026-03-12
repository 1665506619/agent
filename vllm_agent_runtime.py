from __future__ import annotations

import json
import os
import threading
from functools import partial
from typing import Any, Dict, Optional

import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.agent.agent_core import agent_inference
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.model.sam3_image_processor import Sam3Processor

from .vllm_agent_utils import log, safe_filename_component


def setup_torch_runtime() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_sam3_processor(
    confidence_threshold: float = 0.5,
    checkpoint_path: Optional[str] = None,
) -> Sam3Processor:
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    checkpoint_path = checkpoint_path or os.getenv("SAM3_CHECKPOINT_PATH")
    load_from_hf = checkpoint_path is None
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
    )
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


def create_vllm_runtime(
    *,
    api_key: Optional[str],
    model: str,
    base_url: str,
    confidence_threshold: float,
    sam3_checkpoint_path: Optional[str] = None,
    llm_request_interval_s: float = 0.0,
    llm_max_retries: int = 0,
    llm_retry_backoff_s: float = 8.0,
) -> Dict[str, Any]:
    log("Initializing vLLM-compatible runtime.")
    setup_torch_runtime()

    resolved_api_key = (
        api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("VLLM_API_KEY")
        or os.getenv("SILICONFLOW_API_KEY")
        or "EMPTY"
    )

    log(f"Building SAM3 processor with confidence_threshold={confidence_threshold}.")
    if sam3_checkpoint_path:
        log(f"Using local SAM3 checkpoint: {sam3_checkpoint_path}")
    processor = build_sam3_processor(
        confidence_threshold=confidence_threshold,
        checkpoint_path=sam3_checkpoint_path,
    )
    sam3_lock = threading.Lock()
    llm_config = {
        "name": "vllm",
        "provider": "openai_compatible",
        "model": model,
        "api_key": resolved_api_key,
        "base_url": base_url,
    }
    log(
        "vLLM-compatible runtime ready. "
        f"model={model}, base_url={base_url}, "
        f"request_interval_s={llm_request_interval_s}, "
        f"max_retries={llm_max_retries}, retry_backoff_s={llm_retry_backoff_s}"
    )

    def _locked_call_sam_service(**kwargs: Any) -> str:
        with sam3_lock:
            return call_sam_service_orig(sam3_processor=processor, **kwargs)

    return {
        "llm_config": llm_config,
        "sam3_processor": processor,
        "confidence_threshold": confidence_threshold,
        "send_generate_request": partial(
            send_generate_request_orig,
            server_url=base_url,
            model=model,
            api_key=resolved_api_key,
            request_interval_s=llm_request_interval_s,
            max_retries=llm_max_retries,
            retry_backoff_s=llm_retry_backoff_s,
            temperature=0,
        ),
        "call_sam_service": _locked_call_sam_service,
    }


def run_agent_once(
    image_path: str,
    prompt: str,
    *,
    runtime: Dict[str, Any],
    output_dir: str,
    debug: bool,
    max_generations: int,
    artifact_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    if artifact_prefix is None:
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        prompt_for_filename = safe_filename_component(prompt.replace("/", "_"))
        base_filename = (
            f"{image_basename}_{prompt_for_filename}_agent_"
            f"{runtime['llm_config']['name']}"
        )
    else:
        base_filename = artifact_prefix

    output_json_path = os.path.join(output_dir, f"{base_filename}_pred.json")
    output_image_path = os.path.join(output_dir, f"{base_filename}_pred.png")
    agent_history_path = os.path.join(output_dir, f"{base_filename}_history.json")

    log(
        "Starting agent session. "
        f"artifact_prefix={base_filename}, image_path={image_path}, prompt={prompt!r}"
    )

    processor = runtime["sam3_processor"]
    original_threshold = float(runtime["confidence_threshold"])
    log_prefix = f"[{base_filename} thread={threading.current_thread().name}]"
    try:
        processor.set_confidence_threshold(original_threshold)
        log(f"Calling agent_inference. confidence_threshold={original_threshold:.2f}")
        agent_history, final_output_dict, rendered_final_output = agent_inference(
            image_path,
            prompt,
            debug=debug,
            send_generate_request=runtime["send_generate_request"],
            call_sam_service=runtime["call_sam_service"],
            max_generations=max_generations,
            output_dir=output_dir,
            log_prefix=log_prefix,
        )
        log(
            "agent_inference finished. "
            f"confidence_threshold={original_threshold:.2f}, "
            f"num_masks={len(final_output_dict.get('pred_masks', []))}"
        )
    finally:
        processor.set_confidence_threshold(original_threshold)

    final_output_dict["text_prompt"] = prompt
    final_output_dict["image_path"] = image_path
    final_output_dict["confidence_threshold"] = original_threshold
    final_output_dict["llm_provider"] = runtime["llm_config"]["provider"]
    final_output_dict["llm_model"] = runtime["llm_config"]["model"]
    final_output_dict["output_json_path"] = output_json_path
    final_output_dict["output_image_path"] = output_image_path
    final_output_dict["agent_history_path"] = agent_history_path

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output_dict, f, indent=4, ensure_ascii=False)
    with open(agent_history_path, "w", encoding="utf-8") as f:
        json.dump(agent_history, f, indent=4, ensure_ascii=False)
    rendered_final_output.save(output_image_path)
    log(
        "Saved agent outputs. "
        f"pred_json={output_json_path}, history={agent_history_path}, image={output_image_path}"
    )

    return final_output_dict


def run_vllm_sam3_agent(
    image_path: str,
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    base_url: str = "http://127.0.0.1:8000/v1",
    output_dir: str = "agent_output_vllm",
    debug: bool = False,
    confidence_threshold: float = 0.5,
    max_generations: int = 20,
    sam3_checkpoint_path: Optional[str] = None,
    llm_request_interval_s: float = 0.0,
    llm_max_retries: int = 0,
    llm_retry_backoff_s: float = 8.0,
) -> Dict[str, Any]:
    runtime = create_vllm_runtime(
        api_key=api_key,
        model=model,
        base_url=base_url,
        confidence_threshold=confidence_threshold,
        sam3_checkpoint_path=sam3_checkpoint_path,
        llm_request_interval_s=llm_request_interval_s,
        llm_max_retries=llm_max_retries,
        llm_retry_backoff_s=llm_retry_backoff_s,
    )
    return run_agent_once(
        image_path=image_path,
        prompt=prompt,
        runtime=runtime,
        output_dir=output_dir,
        debug=debug,
        max_generations=max_generations,
    )
