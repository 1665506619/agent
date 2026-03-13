#!/usr/bin/env python3
"""Common helpers for vllm_agent multi-node annotation scheduling."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROCESSED_ANSWER_MARKER = "processed"
USE_ORIGIN_ANSWER_MARKER = "use_origin"


def is_completed_answer(answer: Any) -> bool:
    # 统一完成态判断：新格式 + 旧格式兼容
    if answer in {PROCESSED_ANSWER_MARKER, USE_ORIGIN_ANSWER_MARKER}:
        return True

    # Backward compatibility with older answer payloads.
    if isinstance(answer, dict):
        return str(answer.get("status", "")).lower() == "success"

    if isinstance(answer, str) and answer.strip():
        try:
            payload = json.loads(answer)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(payload, dict)
            and str(payload.get("status", "")).lower() == "success"
        )

    return False


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return data


def clone_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cloned: List[Dict[str, Any]] = []
    for sample in dataset:
        sample_copy = dict(sample)
        sample_copy["annotation"] = [
            dict(annotation) for annotation in sample.get("annotation", [])
        ]
        cloned.append(sample_copy)
    return cloned


def atomic_json_dump(data: Any, path: Path) -> None:
    # 原子落盘，避免中断时写出半文件
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def discover_dataset_paths(origin_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(path.resolve() for path in origin_dir.glob(pattern) if path.is_file())


def sanitize_rel_name(rel_name: str) -> str:
    # Preserve path boundaries in a readable form while keeping filenames safe.
    text = rel_name.replace("\\", "/").replace("/", "__")
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in text)


def dataset_rel_name(dataset_path: Path, origin_dir: Path) -> str:
    dataset_path = dataset_path.resolve()
    origin_dir = origin_dir.resolve()
    try:
        rel = dataset_path.relative_to(origin_dir)
    except ValueError:
        rel = Path(dataset_path.name)
    return str(rel)


def dataset_output_paths(
    dataset_path: Path,
    *,
    origin_dir: Path,
    processed_dir: Path,
) -> Tuple[Path, Path, Path, str]:
    # 统一 origin->processed 的文件命名规则，避免多脚本路径不一致
    rel_name = dataset_rel_name(dataset_path, origin_dir)
    rel_no_ext = rel_name[:-5] if rel_name.endswith(".json") else rel_name
    safe_rel = sanitize_rel_name(rel_no_ext)

    output_json = processed_dir / f"agent_{safe_rel}.json"
    failed_debug = processed_dir / f"agent_{safe_rel}_failed_debug.jsonl"
    artifact_dir = processed_dir / f"agent_{safe_rel}_artifacts"
    return output_json.resolve(), failed_debug.resolve(), artifact_dir.resolve(), safe_rel


def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed tails; periodic merges will pick up valid lines.
                continue
            if isinstance(record, dict):
                yield record
