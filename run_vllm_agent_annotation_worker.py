#!/usr/bin/env python3
"""Process assigned annotation tasks and persist per-task WAL records."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sam3.agent.vllm_agent_runtime import create_vllm_runtime, run_agent_once
from sam3.agent.vllm_agent_utils import (
    prediction_to_ann,
    resolve_image_path,
    should_preserve_original_ann,
)

from vllm_agent_multinode_common import (
    PROCESSED_ANSWER_MARKER,
    USE_ORIGIN_ANSWER_MARKER,
    dataset_output_paths,
    iter_jsonl_records,
    load_json_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run annotation-level vllm_agent tasks for one shard rank."
    )
    parser.add_argument("--manifest", required=True, help="Annotation task manifest path.")
    parser.add_argument("--rank", type=int, required=True, help="Shard rank to execute.")
    parser.add_argument("--origin-dir", required=True, help="Origin dataset directory.")
    parser.add_argument("--processed-dir", required=True, help="Processed output directory.")
    parser.add_argument("--image-root", required=True, help="Image root for dataset image paths.")
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument("--base-url", required=True, help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--max-generations", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--delta-path", required=True, help="Node-local WAL output jsonl path.")
    parser.add_argument(
        "--dataset-cache-size",
        type=int,
        default=4,
        help="Max number of dataset JSON payloads cached in memory.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-checkpoint-path", default=None)
    parser.add_argument("--llm-request-interval-s", type=float, default=0.0)
    parser.add_argument("--llm-max-retries", type=int, default=0)
    parser.add_argument("--llm-retry-backoff-s", type=float, default=8.0)
    parser.add_argument("--job-id", default=None)
    return parser.parse_args()


class BoundedDatasetCache:
    """Thread-safe bounded cache to avoid loading all datasets into memory at once."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def get(self, dataset_path: str) -> List[Dict[str, Any]]:
        with self._lock:
            cached = self._cache.get(dataset_path)
            if cached is not None:
                if self._order and self._order[-1] != dataset_path:
                    self._order = [key for key in self._order if key != dataset_path]
                    self._order.append(dataset_path)
                return cached

        loaded = load_json_list(Path(dataset_path))

        with self._lock:
            cached = self._cache.get(dataset_path)
            if cached is not None:
                return cached

            self._cache[dataset_path] = loaded
            self._order = [key for key in self._order if key != dataset_path]
            self._order.append(dataset_path)

            while len(self._order) > self.max_entries:
                evict_key = self._order.pop(0)
                self._cache.pop(evict_key, None)

            return loaded


def load_tasks(manifest_path: Path, rank: int) -> List[Dict[str, Any]]:
    # 只取当前 rank 的任务，避免节点间任务重叠
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    for shard in manifest.get("shards", []):
        if int(shard.get("rank", -1)) == rank:
            tasks = shard.get("tasks", [])
            if not isinstance(tasks, list):
                raise ValueError(f"Invalid tasks payload for rank {rank} in {manifest_path}")
            return tasks

    raise ValueError(f"Rank {rank} not found in manifest: {manifest_path}")


def task_key(dataset_path: str, sample_idx: int, annotation_idx: int) -> tuple[str, int, int]:
    return (str(Path(dataset_path).resolve()), int(sample_idx), int(annotation_idx))


def load_completed_task_keys(result_path: Path) -> set[tuple[str, int, int]]:
    # 支持多次提交复用同一份静态 manifest：已写入本地累计结果的任务直接跳过
    completed: set[tuple[str, int, int]] = set()
    if not result_path.exists():
        return completed

    for record in iter_jsonl_records(result_path):
        dataset_path = record.get("dataset_path")
        sample_idx = record.get("sample_idx")
        annotation_idx = record.get("annotation_idx")
        if not isinstance(dataset_path, str):
            continue
        if not isinstance(sample_idx, int) or not isinstance(annotation_idx, int):
            continue
        completed.add(task_key(dataset_path, sample_idx, annotation_idx))

    return completed


def process_task(
    task: Dict[str, Any],
    *,
    dataset_cache: BoundedDatasetCache,
    artifact_dir_by_dataset: Dict[str, Path],
    runtime: Dict[str, Any],
    image_root: str,
    max_generations: int,
) -> Dict[str, Any]:
    # 单条 annotation 执行逻辑：成功写 processed，失败回退 use_origin
    dataset_path = str(Path(task["dataset_path"]).resolve())
    sample_idx = int(task["sample_idx"])
    annotation_idx = int(task["annotation_idx"])

    dataset = dataset_cache.get(dataset_path)
    sample = dataset[sample_idx]
    annotations = sample.get("annotation", [])
    if not isinstance(annotations, list) or annotation_idx < 0 or annotation_idx >= len(annotations):
        result = {
            "dataset_path": dataset_path,
            "sample_idx": sample_idx,
            "annotation_idx": annotation_idx,
            "image": sample.get("image", ""),
            "text": None,
            "ann": [],
            "answer": USE_ORIGIN_ANSWER_MARKER,
            "error": "Annotation index out of range.",
        }
        return result
    annotation = annotations[annotation_idx]

    raw_image_path = sample.get("image", "")
    prompt = annotation.get("text", "")
    raw_original_ann = annotation.get("ann", [])
    original_ann = list(raw_original_ann) if isinstance(raw_original_ann, list) else []

    result: Dict[str, Any] = {
        "dataset_path": dataset_path,
        "sample_idx": sample_idx,
        "annotation_idx": annotation_idx,
        "image": raw_image_path,
        "text": prompt,
        "ann": list(original_ann),
        "answer": USE_ORIGIN_ANSWER_MARKER,
        "error": None,
    }

    if not isinstance(prompt, str) or not prompt.strip():
        result["error"] = "Missing annotation text prompt."
        return result

    try:
        resolved_image_path = resolve_image_path(
            raw_image_path,
            dataset_path=Path(dataset_path),
            image_root=image_root,
        )
    except Exception as exc:
        result["error"] = str(exc)
        return result

    try:
        prediction = run_agent_once(
            image_path=resolved_image_path,
            prompt=prompt,
            runtime=runtime,
            output_dir=str(artifact_dir_by_dataset[dataset_path]),
            debug=False,
            max_generations=max_generations,
            artifact_prefix=f"s{sample_idx:05d}_a{annotation_idx:02d}",
        )
        preserve_original = should_preserve_original_ann(prediction)
        if not preserve_original:
            result["ann"] = prediction_to_ann(prediction)
        result["answer"] = PROCESSED_ANSWER_MARKER
        result["termination_reason"] = prediction.get("termination_reason")
        result["num_masks"] = len(prediction.get("pred_masks", []))
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def write_delta_record(delta_fp, payload: Dict[str, Any]) -> None:
    # WAL 关键点：每条任务落盘并 fsync，保证被抢占时进度可恢复
    delta_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    delta_fp.flush()
    os.fsync(delta_fp.fileno())


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    delta_path = Path(args.delta_path).resolve()
    completed_task_keys = load_completed_task_keys(delta_path)

    assigned_tasks = load_tasks(manifest_path, rank=args.rank)
    tasks = [
        task
        for task in assigned_tasks
        if task_key(task["dataset_path"], task["sample_idx"], task["annotation_idx"])
        not in completed_task_keys
    ]
    skipped_completed = len(assigned_tasks) - len(tasks)
    if not tasks:
        print(
            f"[worker rank={args.rank}] no remaining tasks; assigned={len(assigned_tasks)}, "
            f"already_completed={skipped_completed}, result={delta_path}"
        )
        return

    dataset_paths = sorted({str(Path(task["dataset_path"]).resolve()) for task in tasks})
    dataset_cache = BoundedDatasetCache(max_entries=args.dataset_cache_size)
    artifact_dir_by_dataset: Dict[str, Path] = {}
    origin_dir = Path(args.origin_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    for dataset_path in dataset_paths:
        _out_json, _failed, artifact_dir, _safe_rel = dataset_output_paths(
            Path(dataset_path),
            origin_dir=origin_dir,
            processed_dir=processed_dir,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir_by_dataset[dataset_path] = artifact_dir

    # runtime 初始化一次并在任务间复用，减少重复模型/处理器初始化开销
    runtime = create_vllm_runtime(
        api_key=None,
        model=args.model,
        base_url=args.base_url,
        confidence_threshold=args.confidence_threshold,
        sam3_checkpoint_path=args.sam3_checkpoint_path,
        llm_request_interval_s=args.llm_request_interval_s,
        llm_max_retries=args.llm_max_retries,
        llm_retry_backoff_s=args.llm_retry_backoff_s,
    )

    delta_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(tasks)
    ok = 0
    fail = 0
    print(
        f"[worker rank={args.rank}] start assigned={len(assigned_tasks)}, remaining={total}, "
        f"already_completed={skipped_completed}, datasets={len(dataset_paths)}, host={socket.gethostname()}"
    )

    with delta_path.open("a", encoding="utf-8") as delta_fp:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            max_inflight = max(args.num_workers * 4, args.num_workers)
            task_iter = iter(tasks)
            future_to_task: Dict[concurrent.futures.Future, Dict[str, Any]] = {}

            def submit_next_task() -> bool:
                try:
                    task = next(task_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    process_task,
                    task,
                    dataset_cache=dataset_cache,
                    artifact_dir_by_dataset=artifact_dir_by_dataset,
                    runtime=runtime,
                    image_root=args.image_root,
                    max_generations=args.max_generations,
                )
                future_to_task[future] = task
                return True

            for _ in range(min(max_inflight, total)):
                submit_next_task()

            completed = 0
            # 有界 inflight，避免一次性提交过多 future 占用内存
            while future_to_task:
                done_set, _ = concurrent.futures.wait(
                    future_to_task,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done_set:
                    task = future_to_task.pop(future)
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        dataset_path = str(Path(task["dataset_path"]).resolve())
                        sample_idx = int(task["sample_idx"])
                        annotation_idx = int(task["annotation_idx"])
                        fallback_ann: List[Any] = []
                        try:
                            dataset = dataset_cache.get(dataset_path)
                            sample = dataset[sample_idx]
                            annotations = sample.get("annotation", [])
                            if (
                                isinstance(annotations, list)
                                and 0 <= annotation_idx < len(annotations)
                                and isinstance(annotations[annotation_idx], dict)
                            ):
                                raw_ann = annotations[annotation_idx].get("ann", [])
                                fallback_ann = (
                                    list(raw_ann) if isinstance(raw_ann, list) else []
                                )
                        except Exception:
                            pass

                        result = {
                            "dataset_path": dataset_path,
                            "sample_idx": sample_idx,
                            "annotation_idx": annotation_idx,
                            "image": None,
                            "text": None,
                            "ann": fallback_ann,
                            "answer": USE_ORIGIN_ANSWER_MARKER,
                            "error": f"Unhandled worker exception: {exc}",
                        }

                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "job_id": args.job_id,
                        "rank": args.rank,
                        **result,
                    }

                    write_delta_record(delta_fp, record)

                    if result.get("error") is None:
                        ok += 1
                    else:
                        fail += 1

                    if completed % 10 == 0 or completed == total:
                        print(
                            f"[worker rank={args.rank}] progress {completed}/{total}, ok={ok}, fail={fail}"
                        )

                    submit_next_task()

    print(
        f"[worker rank={args.rank}] done total={total}, ok={ok}, fail={fail}, delta={delta_path}"
    )


if __name__ == "__main__":
    main()
