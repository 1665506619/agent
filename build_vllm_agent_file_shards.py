#!/usr/bin/env python3
"""Build balanced annotation-level shards for vllm_agent multi-node runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vllm_agent_multinode_common import (
    dataset_output_paths,
    discover_dataset_paths,
    is_completed_answer,
    load_json_list,
)


@dataclass
class AnnotationTask:
    # 一个最小调度单元：精确到某个 JSON 的某条 annotation
    dataset_path: str
    sample_idx: int
    annotation_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split pending annotations from all origin JSON datasets into balanced "
            "multi-node shards."
        )
    )
    parser.add_argument("--origin-dir", required=True, help="Directory of input JSON datasets.")
    parser.add_argument("--processed-dir", required=True, help="Directory of processed outputs.")
    parser.add_argument("--num-shards", type=int, required=True, help="Shard count.")
    parser.add_argument("--output", required=True, help="Output manifest path.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover input JSON files under origin-dir.",
    )
    return parser.parse_args()


def pending_tasks_for_dataset(
    dataset_path: Path,
    *,
    origin_dir: Path,
    processed_dir: Path,
) -> Tuple[List[AnnotationTask], Dict[str, int]]:
    # 对单个数据集：找出“尚未在 processed 里完成”的 annotation
    dataset = load_json_list(dataset_path)

    output_json_path, _, _, _ = dataset_output_paths(
        dataset_path,
        origin_dir=origin_dir,
        processed_dir=processed_dir,
    )

    processed_dataset: List[Dict[str, Any]] | None = None
    if output_json_path.exists():
        try:
            candidate = load_json_list(output_json_path)
        except Exception:
            candidate = None
        if isinstance(candidate, list) and len(candidate) == len(dataset):
            processed_dataset = candidate

    tasks: List[AnnotationTask] = []
    total_annotations = 0
    completed_from_processed = 0

    for sample_idx, sample in enumerate(dataset):
        annotations = sample.get("annotation", [])
        if not isinstance(annotations, list):
            continue

        for ann_idx, _ in enumerate(annotations):
            total_annotations += 1

            processed_done = False
            if processed_dataset is not None:
                try:
                    processed_ann = processed_dataset[sample_idx].get("annotation", [])[ann_idx]
                except Exception:
                    processed_ann = None
                if isinstance(processed_ann, dict) and is_completed_answer(
                    processed_ann.get("answer")
                ):
                    processed_done = True
                    completed_from_processed += 1

            if processed_done:
                continue

            tasks.append(
                AnnotationTask(
                    dataset_path=str(dataset_path.resolve()),
                    sample_idx=sample_idx,
                    annotation_idx=ann_idx,
                )
            )

    stats = {
        "total_annotations": total_annotations,
        "pending_annotations": len(tasks),
        "completed_from_processed": completed_from_processed,
    }
    return tasks, stats


def assign_tasks(tasks: List[AnnotationTask], num_shards: int) -> List[List[AnnotationTask]]:
    # 按“当前负载最小优先”把 annotation 均衡分到各个 shard
    shards: List[List[AnnotationTask]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards

    for task in tasks:
        target = min(range(num_shards), key=lambda idx: (loads[idx], idx))
        shards[target].append(task)
        loads[target] += 1

    return shards


def main() -> None:
    args = parse_args()

    origin_dir = Path(args.origin_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    output_path = Path(args.output).resolve()

    if args.num_shards < 1:
        raise ValueError(f"num-shards must be >= 1, got {args.num_shards}")
    if not origin_dir.exists() or not origin_dir.is_dir():
        raise FileNotFoundError(f"origin-dir not found or not a directory: {origin_dir}")

    dataset_paths = discover_dataset_paths(origin_dir, recursive=args.recursive)
    if not dataset_paths:
        raise FileNotFoundError(f"No JSON files found under {origin_dir}")

    # 汇总全部待处理 annotation，并记录每个数据集统计信息
    all_tasks: List[AnnotationTask] = []
    dataset_stats: List[Dict[str, Any]] = []
    for dataset_path in dataset_paths:
        tasks, stats = pending_tasks_for_dataset(
            dataset_path,
            origin_dir=origin_dir,
            processed_dir=processed_dir,
        )
        all_tasks.extend(tasks)
        dataset_stats.append(
            {
                "dataset_path": str(dataset_path),
                **stats,
            }
        )

    shards = assign_tasks(all_tasks, num_shards=args.num_shards)

    # manifest 是后续多节点执行的唯一任务来源
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_dir": str(origin_dir),
        "processed_dir": str(processed_dir),
        "num_shards": args.num_shards,
        "total_datasets": len(dataset_paths),
        "total_pending_annotations": len(all_tasks),
        "dataset_stats": dataset_stats,
        "shards": [
            {
                "rank": rank,
                "num_tasks": len(tasks),
                "tasks": [
                    {
                        "dataset_path": task.dataset_path,
                        "sample_idx": task.sample_idx,
                        "annotation_idx": task.annotation_idx,
                    }
                    for task in tasks
                ],
            }
            for rank, tasks in enumerate(shards)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[manifest] wrote: {output_path}")
    print(f"[manifest] datasets={len(dataset_paths)}")
    print(f"[manifest] pending_annotations={len(all_tasks)}")
    for shard in manifest["shards"]:
        print(f"[manifest] shard {shard['rank']}: tasks={shard['num_tasks']}")


if __name__ == "__main__":
    main()
