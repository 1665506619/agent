"""vLLM-backed SAM3 agent runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sam3.agent.vllm_agent_dataset import run_vllm_sam3_dataset
from sam3.agent.vllm_agent_runtime import run_vllm_sam3_agent

AGENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = (
    AGENT_DIR.parent / "agent_datas" / "origin" / "hd_epic_qa_no_none_filtered.json"
)
DEFAULT_OUTPUT_DIR = AGENT_DIR.parent / "agent_datas" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SAM3 agent with a vLLM/OpenAI-compatible endpoint on either "
            "a single image prompt or the hd_epic refseg dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Refseg dataset JSON to process in batch mode.",
    )
    parser.add_argument(
        "--image-root",
        default="/lustre/fs11/portfolios/llmservice/users/zhidingy/wsh-ws/playground/region/data",
        help="Root directory used to resolve relative image paths in the dataset.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to the aggregated output json. Defaults to agent_<dataset>.json.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory for per-annotation pred/history/png artifacts.",
    )
    parser.add_argument(
        "--failed-debug-path",
        default=None,
        help="JSONL file that records failed annotations for retry.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output snapshots and restart dataset processing from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N dataset samples.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this 0-based dataset sample index.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of annotation worker threads in dataset mode.",
    )
    parser.add_argument("--image", default=None, help="Single-image mode input path.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single-image mode natural language query.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the OpenAI-compatible endpoint. Local vLLM can leave this empty.",
    )
    parser.add_argument(
        "--model",
        default="/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data/data_promote/weights/Qwen3-VL-30B-A3B-Instruct",
        help="Model name served by the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL. Point this to vLLM when running locally.",
    )
    parser.add_argument(
        "--output-dir",
        default="agent_output_vllm",
        help="Single-image mode output directory for pred json/png/history.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="SAM3 processor confidence threshold.",
    )
    parser.add_argument(
        "--sam3-checkpoint-path",
        default="../weights/sam3.pt",
        help=(
            "Local SAM3 checkpoint path. If set, local weights are used instead "
            "of downloading from Hugging Face."
        ),
    )
    parser.add_argument(
        "--llm-request-interval-s",
        type=float,
        default=0.0,
        help="Minimum sleep interval between MLLM requests. Local vLLM usually uses 0.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=0,
        help="Maximum retry count for MLLM requests.",
    )
    parser.add_argument(
        "--llm-retry-backoff-s",
        type=float,
        default=8.0,
        help="Initial backoff seconds for MLLM retries.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=20,
        help="Maximum MLLM generation rounds in the agent loop.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode in agent_inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image or args.prompt:
        if not args.image or not args.prompt:
            raise ValueError("Single-image mode requires both --image and --prompt.")
        result = run_vllm_sam3_agent(
            image_path=args.image,
            prompt=args.prompt,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            output_dir=args.output_dir,
            debug=args.debug,
            confidence_threshold=args.confidence_threshold,
            max_generations=args.max_generations,
            sam3_checkpoint_path=args.sam3_checkpoint_path,
            llm_request_interval_s=args.llm_request_interval_s,
            llm_max_retries=args.llm_max_retries,
            llm_retry_backoff_s=args.llm_retry_backoff_s,
        )
        print("\n=== Final Mask List ===")
        print(json.dumps(result.get("pred_masks", []), ensure_ascii=False, indent=2))
        return

    output_path = run_vllm_sam3_dataset(
        dataset_path=args.dataset,
        image_root=args.image_root,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        output_json_path=args.output_json,
        artifact_dir=args.artifact_dir,
        failed_debug_path=args.failed_debug_path,
        debug=args.debug,
        confidence_threshold=args.confidence_threshold,
        max_generations=args.max_generations,
        limit=args.limit,
        start_index=args.start_index,
        num_workers=args.num_workers,
        sam3_checkpoint_path=args.sam3_checkpoint_path,
        llm_request_interval_s=args.llm_request_interval_s,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff_s=args.llm_retry_backoff_s,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        resume=not args.no_resume,
    )
    print(f"\n=== Dataset Processing Complete ===\n{output_path}")


if __name__ == "__main__":
    main()
