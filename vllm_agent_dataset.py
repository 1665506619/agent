from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .vllm_agent_runtime import create_vllm_runtime, run_agent_once
from .vllm_agent_utils import (
    append_failed_debug,
    atomic_json_dump,
    build_failed_payload,
    clone_dataset,
    default_dataset_output_path,
    default_failed_debug_path,
    is_completed_annotation,
    load_refseg_dataset,
    log,
    prediction_to_ann,
    PROCESSED_ANSWER_MARKER,
    resolve_image_path,
    should_preserve_original_ann,
)


def process_annotation_task(
    *,
    item_idx: int,
    total_items: int,
    sample_idx: int,
    ann_idx: int,
    raw_image_path: str,
    resolved_image_path: str,
    prompt: str,
    original_ann: Any,
    runtime: Dict[str, Any],
    sample_artifact_dir: str,
    debug: bool,
    max_generations: int,
) -> Dict[str, Any]:
    log(
        f"Worker starting item={item_idx}/{total_items}, "
        f"sample={sample_idx}, annotation={ann_idx}, "
        f"thread={threading.current_thread().name}"
    )
    try:
        prediction = run_agent_once(
            image_path=resolved_image_path,
            prompt=prompt,
            runtime=runtime,
            output_dir=sample_artifact_dir,
            debug=debug,
            max_generations=max_generations,
            artifact_prefix=f"s{sample_idx:05d}_a{ann_idx:02d}",
        )
        preserve_original = should_preserve_original_ann(prediction)
        ann_value = (
            list(original_ann) if preserve_original else prediction_to_ann(prediction)
        )
        return {
            "item_idx": item_idx,
            "total_items": total_items,
            "sample_idx": sample_idx,
            "annotation_idx": ann_idx,
            "image": raw_image_path,
            "text": prompt,
            "ann": ann_value,
            "error": None,
            "preserve_original_ann": preserve_original,
            "num_masks": len(prediction.get("pred_masks", [])),
            "termination_reason": prediction.get("termination_reason"),
            "output_json_path": prediction.get("output_json_path"),
            "output_image_path": prediction.get("output_image_path"),
            "agent_history_path": prediction.get("agent_history_path"),
        }
    except Exception as exc:
        return {
            "item_idx": item_idx,
            "total_items": total_items,
            "sample_idx": sample_idx,
            "annotation_idx": ann_idx,
            "image": raw_image_path,
            "text": prompt,
            "ann": [],
            "error": str(exc),
            "preserve_original_ann": False,
            "num_masks": 0,
            "termination_reason": None,
            "output_json_path": None,
            "output_image_path": None,
            "agent_history_path": None,
        }


def run_vllm_sam3_dataset(
    dataset_path: str,
    *,
    image_root: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    base_url: str = "http://127.0.0.1:8000/v1",
    output_json_path: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    failed_debug_path: Optional[str] = None,
    debug: bool = False,
    confidence_threshold: float = 0.5,
    max_generations: int = 20,
    limit: Optional[int] = None,
    start_index: int = 0,
    num_workers: int = 1,
    sam3_checkpoint_path: Optional[str] = None,
    llm_request_interval_s: float = 0.0,
    llm_max_retries: int = 0,
    llm_retry_backoff_s: float = 8.0,
    default_output_dir: Optional[Path] = None,
    resume: bool = True,
) -> Path:
    dataset_path_obj = Path(dataset_path).resolve()
    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path_obj}")

    dataset = load_refseg_dataset(dataset_path_obj)
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if start_index > len(dataset):
        raise ValueError(
            f"start_index {start_index} is out of range for dataset with "
            f"{len(dataset)} samples"
        )

    process_end_index = len(dataset)
    if limit is not None:
        process_end_index = min(len(dataset), start_index + limit)
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

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

    resolved_default_output_dir = (
        default_output_dir.resolve()
        if default_output_dir is not None
        else Path(__file__).resolve().parent / "agent_datas" / "processed"
    )
    output_path = (
        Path(output_json_path).resolve()
        if output_json_path
        else default_dataset_output_path(resolved_default_output_dir, dataset_path_obj)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = (
        Path(failed_debug_path).resolve()
        if failed_debug_path
        else default_failed_debug_path(output_path)
    )

    artifact_root = (
        Path(artifact_dir).resolve()
        if artifact_dir
        else output_path.parent / f"{output_path.stem}_artifacts"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    output_dataset = clone_dataset(dataset)
    resumed_from_snapshot = False
    if resume and output_path.exists():
        try:
            existing_output = load_refseg_dataset(output_path)
        except Exception as exc:
            log(
                "Failed to load existing output snapshot for resume. "
                f"Starting fresh. error={exc}"
            )
        else:
            if len(existing_output) != len(dataset):
                log(
                    "Existing output snapshot size does not match input dataset. "
                    f"expected={len(dataset)}, actual={len(existing_output)}. Starting fresh."
                )
            else:
                output_dataset = clone_dataset(existing_output)
                resumed_from_snapshot = True
                log(f"Resuming from existing output snapshot: {output_path}")
    samples_to_process = dataset[start_index:process_end_index]
    total_annotations = sum(
        len(sample.get("annotation", [])) for sample in samples_to_process
    )
    completed_annotations = 0
    next_item_idx = 1

    print(
        f"Processing dataset: {dataset_path_obj} "
        f"(processing samples [{start_index}, {process_end_index}) "
        f"out of {len(dataset)}, {total_annotations} annotations)"
    )
    print(f"Output JSON: {output_path}")
    print(f"Failed debug JSONL: {failed_path}")
    print(f"Artifact directory: {artifact_root}")
    print(f"Worker threads: {num_workers}")
    if resumed_from_snapshot:
        print(f"Resume snapshot: {output_path}")

    if not samples_to_process:
        log("No samples selected for processing. Writing untouched dataset snapshot.")
        atomic_json_dump(dataset, output_path)
        return output_path

    future_to_meta: Dict[concurrent.futures.Future, Dict[str, Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix="sam3-agent",
    ) as executor:
        for sample_idx in range(start_index, process_end_index):
            sample = dataset[sample_idx]
            log(f"Starting sample {sample_idx} with image={sample.get('image', '')}.")
            raw_image_path = sample.get("image", "")
            resolved_image_path = None
            image_error = None

            try:
                log(
                    f"Resolving image path for sample {sample_idx}: raw_image_path={raw_image_path}"
                )
                resolved_image_path = resolve_image_path(
                    raw_image_path,
                    dataset_path=dataset_path_obj,
                    image_root=image_root,
                )
                log(f"Resolved image path for sample {sample_idx}: {resolved_image_path}")
            except Exception as exc:
                image_error = str(exc)
                log(f"Failed to resolve image path for sample {sample_idx}: {image_error}")

            annotations = sample.get("annotation", [])
            sample_artifact_dir = artifact_root / f"sample_{sample_idx:05d}"
            sample_artifact_dir.mkdir(parents=True, exist_ok=True)

            for ann_idx, annotation in enumerate(annotations):
                prompt = annotation.get("text", "")
                item_idx = next_item_idx
                next_item_idx += 1
                annotation_artifact_dir = (
                    sample_artifact_dir / f"annotation_{ann_idx:02d}"
                )
                annotation_artifact_dir.mkdir(parents=True, exist_ok=True)
                log(
                    "Scheduling item="
                    f"{item_idx}/{total_annotations}, sample={sample_idx}, "
                    f"annotation={ann_idx}, prompt={prompt!r}, "
                    f"artifact_dir={annotation_artifact_dir}"
                )

                if is_completed_annotation(
                    output_dataset[sample_idx]["annotation"][ann_idx],
                    image=raw_image_path,
                    prompt=prompt,
                ):
                    completed_annotations += 1
                    log(
                        "Skipping completed annotation from resume snapshot. "
                        f"item={item_idx}/{total_annotations}, "
                        f"sample={sample_idx}, annotation={ann_idx}"
                    )
                    print(
                        f"[{completed_annotations}/{total_annotations}] "
                        f"{raw_image_path} :: annotation {ann_idx + 1}/{len(annotations)} "
                        "(resumed)"
                    )
                    continue

                if image_error is not None:
                    output_dataset[sample_idx]["annotation"][ann_idx]["ann"] = []
                    output_dataset[sample_idx]["annotation"][ann_idx]["answer"] = ""
                    append_failed_debug(
                        failed_path,
                        build_failed_payload(
                            sample_idx=sample_idx,
                            ann_idx=ann_idx,
                            image=raw_image_path,
                            prompt=prompt,
                            error=image_error,
                        ),
                    )
                    completed_annotations += 1
                    print(
                        f"[{completed_annotations}/{total_annotations}] "
                        f"{raw_image_path} :: annotation {ann_idx + 1}/{len(annotations)} "
                        "(image resolve failed)"
                    )
                    atomic_json_dump(output_dataset, output_path)
                    continue

                if not isinstance(prompt, str) or not prompt.strip():
                    output_dataset[sample_idx]["annotation"][ann_idx]["ann"] = []
                    output_dataset[sample_idx]["annotation"][ann_idx]["answer"] = ""
                    append_failed_debug(
                        failed_path,
                        build_failed_payload(
                            sample_idx=sample_idx,
                            ann_idx=ann_idx,
                            image=raw_image_path,
                            prompt=prompt,
                            error="Missing annotation text prompt.",
                        ),
                    )
                    completed_annotations += 1
                    print(
                        f"[{completed_annotations}/{total_annotations}] "
                        f"{raw_image_path} :: annotation {ann_idx + 1}/{len(annotations)} "
                        "(missing prompt)"
                    )
                    atomic_json_dump(output_dataset, output_path)
                    continue

                future = executor.submit(
                    process_annotation_task,
                    item_idx=item_idx,
                    total_items=total_annotations,
                    sample_idx=sample_idx,
                    ann_idx=ann_idx,
                    raw_image_path=raw_image_path,
                    resolved_image_path=resolved_image_path,
                    prompt=prompt,
                    original_ann=annotation.get("ann", []),
                    runtime=runtime,
                    sample_artifact_dir=str(annotation_artifact_dir),
                    debug=debug,
                    max_generations=max_generations,
                )
                future_to_meta[future] = {
                    "item_idx": item_idx,
                    "sample_idx": sample_idx,
                    "ann_idx": ann_idx,
                    "raw_image_path": raw_image_path,
                    "annotation_count": len(annotations),
                }

        for future in concurrent.futures.as_completed(future_to_meta):
            meta = future_to_meta[future]
            result = future.result()
            item_idx = result["item_idx"]
            sample_idx = result["sample_idx"]
            ann_idx = result["annotation_idx"]
            raw_image_path = result["image"]
            output_dataset[sample_idx]["annotation"][ann_idx]["ann"] = result["ann"]
            output_dataset[sample_idx]["annotation"][ann_idx]["answer"] = (
                "" if result["error"] is not None else PROCESSED_ANSWER_MARKER
            )

            if result["error"] is not None:
                print(
                    f"Failed on item {item_idx}/{total_annotations}, "
                    f"sample {sample_idx}, annotation {ann_idx}: {result['error']}"
                )
                log(
                    f"Agent run failed for item={item_idx}/{total_annotations}, "
                    f"sample={sample_idx}, annotation={ann_idx}: "
                    f"{result['error']}"
                )
                append_failed_debug(
                    failed_path,
                    build_failed_payload(
                        sample_idx=sample_idx,
                        ann_idx=ann_idx,
                        image=raw_image_path,
                        prompt=result["text"],
                        error=result["error"],
                    ),
                )
            elif result["preserve_original_ann"]:
                log(
                    "Applied original ann because final pred_masks is empty. "
                    f"item={item_idx}/{total_annotations}, "
                    f"sample={sample_idx}, annotation={ann_idx}, "
                    f"termination_reason={result['termination_reason']}"
                )
            else:
                log(
                    "Prediction converted to ann. "
                    f"item={item_idx}/{total_annotations}, "
                    f"sample={sample_idx}, annotation={ann_idx}, "
                    f"num_masks={result['num_masks']}"
                )

            completed_annotations += 1
            print(
                f"[{completed_annotations}/{total_annotations}] "
                f"{meta['raw_image_path']} :: annotation "
                f"{ann_idx + 1}/{meta['annotation_count']} completed"
            )
            atomic_json_dump(output_dataset, output_path)
            log(
                f"Finished item={item_idx}/{total_annotations}, "
                f"sample={sample_idx}, annotation={ann_idx}. Progress snapshot saved."
            )

    log(f"Dataset processing finished successfully. Output JSON at {output_path}.")
    return output_path
