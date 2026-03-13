#!/usr/bin/env python3
"""Incrementally merge annotation WAL deltas into processed dataset snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vllm_agent_multinode_common import (
    USE_ORIGIN_ANSWER_MARKER,
    atomic_json_dump,
    clone_dataset,
    dataset_output_paths,
    load_json_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply new delta WAL records to processed dataset JSON files."
    )
    parser.add_argument("--origin-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--delta-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--truncate-consumed",
        action="store_true",
        help="Truncate fully-consumed delta files after merge to reclaim disk space.",
    )
    parser.add_argument(
        "--active-job-id",
        default=None,
        help=(
            "Optional active Slurm job id. When set with --truncate-consumed, files "
            "matching job_<id>_*.jsonl are not truncated to avoid races with writers."
        ),
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path) -> Dict[str, int]:
    # checkpoint 记录每个 delta 文件已消费到的 byte offset
    if not checkpoint_path.exists():
        return {}
    try:
        with checkpoint_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    offsets = payload.get("offsets", {}) if isinstance(payload, dict) else {}
    if not isinstance(offsets, dict):
        return {}

    normalized: Dict[str, int] = {}
    for key, value in offsets.items():
        if isinstance(key, str) and isinstance(value, int) and value >= 0:
            normalized[key] = value
    return normalized


def read_incremental_records(
    delta_path: Path,
    start_offset: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    # 只读取 start_offset 之后的新行，实现增量 merge
    records: List[Dict[str, Any]] = []
    with delta_path.open("rb") as f:
        f.seek(start_offset)
        while True:
            line_start = f.tell()
            line = f.readline()
            if not line:
                break
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Likely a partially written tail line; retry next merge cycle.
                f.seek(line_start)
                break
            if isinstance(payload, dict):
                records.append(payload)
        end_offset = f.tell()
    return end_offset, records


def is_active_job_delta(delta_path: Path, active_job_id: str | None) -> bool:
    if not active_job_id:
        return False
    return delta_path.name.startswith(f"job_{active_job_id}_")


def load_or_init_processed_dataset(
    dataset_path: Path,
    *,
    output_json_path: Path,
) -> List[Dict[str, Any]]:
    # 目标 processed 文件不存在或不匹配时，用 origin 快照初始化
    source_dataset = load_json_list(dataset_path)

    if output_json_path.exists():
        try:
            existing = load_json_list(output_json_path)
        except Exception:
            existing = None
        if isinstance(existing, list) and len(existing) == len(source_dataset):
            return existing

    return clone_dataset(source_dataset)


def apply_records(
    dataset: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> Tuple[int, int]:
    # 把 delta 中的 ann/answer 回填到 dataset 指定位置
    applied = 0
    skipped = 0

    for record in records:
        sample_idx = record.get("sample_idx")
        annotation_idx = record.get("annotation_idx")
        answer = record.get("answer", USE_ORIGIN_ANSWER_MARKER)
        ann = record.get("ann", [])

        if not isinstance(sample_idx, int) or not isinstance(annotation_idx, int):
            skipped += 1
            continue
        if sample_idx < 0 or sample_idx >= len(dataset):
            skipped += 1
            continue

        annotations = dataset[sample_idx].get("annotation", [])
        if not isinstance(annotations, list):
            skipped += 1
            continue
        if annotation_idx < 0 or annotation_idx >= len(annotations):
            skipped += 1
            continue

        ann_obj = annotations[annotation_idx]
        if not isinstance(ann_obj, dict):
            skipped += 1
            continue

        ann_obj["ann"] = ann
        ann_obj["answer"] = answer
        applied += 1

    return applied, skipped


def append_failed_debug(
    failed_path: Path,
    records: List[Dict[str, Any]],
) -> int:
    # 将失败记录附加到 failed_debug，便于后续排查
    failed_payloads: List[Dict[str, Any]] = []
    for record in records:
        error = record.get("error")
        if error in (None, ""):
            continue
        failed_payloads.append(
            {
                "sample_idx": record.get("sample_idx"),
                "annotation_idx": record.get("annotation_idx"),
                "image": record.get("image"),
                "text": record.get("text"),
                "error": error,
            }
        )

    if not failed_payloads:
        return 0

    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with failed_path.open("a", encoding="utf-8") as f:
        for payload in failed_payloads:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return len(failed_payloads)


def main() -> None:
    args = parse_args()

    origin_dir = Path(args.origin_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    delta_dir = Path(args.delta_dir).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()

    if not origin_dir.exists() or not origin_dir.is_dir():
        raise FileNotFoundError(f"origin-dir not found: {origin_dir}")
    if not delta_dir.exists() or not delta_dir.is_dir():
        print(f"[merge] delta-dir does not exist yet: {delta_dir}")
        return

    offsets = load_checkpoint(checkpoint_path)
    new_offsets = dict(offsets)

    # 先按 dataset 聚合增量，再批量写回，减少重复 I/O
    updates_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    total_new_records = 0
    file_size_by_key: Dict[str, int] = {}
    delta_path_by_key: Dict[str, Path] = {}

    delta_paths = sorted(path.resolve() for path in delta_dir.glob("**/*.jsonl") if path.is_file())

    for delta_path in delta_paths:
        key = str(delta_path)
        start_offset = offsets.get(key, 0)

        file_size = delta_path.stat().st_size
        file_size_by_key[key] = file_size
        delta_path_by_key[key] = delta_path
        if start_offset > file_size:
            start_offset = 0

        end_offset, records = read_incremental_records(delta_path, start_offset)
        new_offsets[key] = end_offset

        if not records:
            continue

        total_new_records += len(records)
        for record in records:
            dataset_path = record.get("dataset_path")
            if not isinstance(dataset_path, str):
                continue
            dataset_key = str(Path(dataset_path).resolve())
            updates_by_dataset.setdefault(dataset_key, []).append(record)

    applied_total = 0
    skipped_total = 0
    failed_debug_total = 0

    for dataset_key, records in updates_by_dataset.items():
        dataset_path = Path(dataset_key)
        if not dataset_path.exists():
            skipped_total += len(records)
            continue

        output_json_path, failed_debug_path, _artifact_dir, _safe_rel = dataset_output_paths(
            dataset_path,
            origin_dir=origin_dir,
            processed_dir=processed_dir,
        )
        processed_dataset = load_or_init_processed_dataset(
            dataset_path,
            output_json_path=output_json_path,
        )

        applied, skipped = apply_records(processed_dataset, records)
        applied_total += applied
        skipped_total += skipped

        atomic_json_dump(processed_dataset, output_json_path)
        failed_debug_total += append_failed_debug(failed_debug_path, records)

    checkpoint_payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "offsets": new_offsets,
    }

    truncated_files = 0
    if args.truncate_consumed:
        # 可选空间回收：仅截断“已完全消费且非活动作业”的 delta 文件
        for key, offset in list(new_offsets.items()):
            delta_path = delta_path_by_key.get(key)
            if delta_path is None:
                continue
            if is_active_job_delta(delta_path, args.active_job_id):
                continue

            consumed_size = file_size_by_key.get(key)
            if consumed_size is None:
                continue
            if offset < consumed_size:
                continue

            # Fully consumed and no active writer: safe to truncate.
            with delta_path.open("wb"):
                pass
            new_offsets[key] = 0
            truncated_files += 1

        checkpoint_payload["offsets"] = new_offsets
    atomic_json_dump(checkpoint_payload, checkpoint_path)

    print(
        "[merge] delta_files={files}, new_records={new_records}, datasets_touched={datasets}, applied={applied}, skipped={skipped}, failed_debug_appends={failed_debug}, truncated_files={truncated}".format(
            files=len(delta_paths),
            new_records=total_new_records,
            datasets=len(updates_by_dataset),
            applied=applied_total,
            skipped=skipped_total,
            failed_debug=failed_debug_total,
            truncated=truncated_files,
        )
    )


if __name__ == "__main__":
    main()
