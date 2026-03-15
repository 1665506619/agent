#!/usr/bin/env python3
"""Merge cumulative per-rank annotation results into processed dataset snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vllm_agent_multinode_common import (
    USE_ORIGIN_ANSWER_MARKER,
    atomic_json_dump,
    clone_dataset,
    dataset_output_paths,
    iter_jsonl_records,
    load_json_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge cumulative per-rank annotation result jsonl files into processed outputs. "
            "This is intended to run once after all static-manifest annotation jobs finish."
        )
    )
    parser.add_argument("--origin-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    return parser.parse_args()


def load_or_init_processed_dataset(
    dataset_path: Path,
    *,
    output_json_path: Path,
) -> List[Dict[str, Any]]:
    source_dataset = load_json_list(dataset_path)

    if output_json_path.exists():
        try:
            existing = load_json_list(output_json_path)
        except Exception:
            existing = None
        if isinstance(existing, list) and len(existing) == len(source_dataset):
            return existing

    return clone_dataset(source_dataset)


def append_failed_debug(failed_path: Path, records: List[Dict[str, Any]]) -> int:
    failed_payloads: List[Dict[str, Any]] = []
    for record in records:
        error = record.get("error")
        if error in (None, ""):
            continue
        failed_payloads.append(
            {
                "job_id": record.get("job_id"),
                "rank": record.get("rank"),
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
    results_dir = Path(args.results_dir).resolve()

    if not origin_dir.exists() or not origin_dir.is_dir():
        raise FileNotFoundError(f"origin-dir not found: {origin_dir}")
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"results-dir not found: {results_dir}")

    result_paths = sorted(path.resolve() for path in results_dir.glob("*.jsonl") if path.is_file())
    if not result_paths:
        raise FileNotFoundError(f"No jsonl result files found under {results_dir}")

    updates_by_dataset: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    total_records = 0

    for result_path in result_paths:
        file_records = 0
        for record in iter_jsonl_records(result_path):
            dataset_path = record.get("dataset_path")
            sample_idx = record.get("sample_idx")
            annotation_idx = record.get("annotation_idx")
            if not isinstance(dataset_path, str):
                continue
            if not isinstance(sample_idx, int) or not isinstance(annotation_idx, int):
                continue

            dataset_key = str(Path(dataset_path).resolve())
            dataset_updates = updates_by_dataset.setdefault(dataset_key, {})
            dataset_updates[(sample_idx, annotation_idx)] = record
            total_records += 1
            file_records += 1

        print(f"[final-merge] loaded {file_records} records from {result_path.name}")

    applied_total = 0
    skipped_total = 0
    failed_debug_total = 0

    for dataset_key, update_map in updates_by_dataset.items():
        dataset_path = Path(dataset_key)
        if not dataset_path.exists():
            skipped_total += len(update_map)
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

        failed_records_for_dataset: List[Dict[str, Any]] = []
        applied_for_dataset = 0
        skipped_for_dataset = 0

        for (sample_idx, annotation_idx), record in sorted(update_map.items()):
            answer = record.get("answer", USE_ORIGIN_ANSWER_MARKER)
            ann = record.get("ann", [])

            if sample_idx < 0 or sample_idx >= len(processed_dataset):
                skipped_for_dataset += 1
                continue

            annotations = processed_dataset[sample_idx].get("annotation", [])
            if not isinstance(annotations, list):
                skipped_for_dataset += 1
                continue
            if annotation_idx < 0 or annotation_idx >= len(annotations):
                skipped_for_dataset += 1
                continue

            ann_obj = annotations[annotation_idx]
            if not isinstance(ann_obj, dict):
                skipped_for_dataset += 1
                continue

            ann_obj["ann"] = ann
            ann_obj["answer"] = answer
            applied_for_dataset += 1
            if record.get("error") not in (None, ""):
                failed_records_for_dataset.append(record)

        atomic_json_dump(processed_dataset, output_json_path)
        failed_debug_total += append_failed_debug(failed_debug_path, failed_records_for_dataset)

        applied_total += applied_for_dataset
        skipped_total += skipped_for_dataset
        print(
            f"[final-merge] dataset={dataset_path.name} applied={applied_for_dataset} skipped={skipped_for_dataset}"
        )

    print(
        "[final-merge] result_files={files}, total_records={records}, datasets_touched={datasets}, applied={applied}, skipped={skipped}, failed_debug_appends={failed_debug}".format(
            files=len(result_paths),
            records=total_records,
            datasets=len(updates_by_dataset),
            applied=applied_total,
            skipped=skipped_total,
            failed_debug=failed_debug_total,
        )
    )


if __name__ == "__main__":
    main()
