from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[vllm_agent {timestamp}] {message}", flush=True)


def safe_filename_component(value: str, max_length: int = 120) -> str:
    sanitized = [
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip()
    ]
    filename = "".join(sanitized).strip("._") or "item"
    return filename[:max_length]


def load_refseg_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    log(f"Loading dataset from {dataset_path}.")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected a list dataset, got: {type(dataset).__name__}")
    log(f"Loaded dataset successfully with {len(dataset)} samples.")
    return dataset


def resolve_image_path(
    raw_image_path: str,
    *,
    dataset_path: Path,
    image_root: Optional[str],
) -> str:
    extra_image_root = Path(
        "/lustre/fs12/portfolios/nvr/projects/nvr_lpr_nvgptvision/users/shihaow/region/data"
    )
    raw_path = Path(raw_image_path)
    candidates: List[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    if image_root:
        candidates.append(Path(image_root) / raw_path)
    candidates.append(extra_image_root / raw_path)
    candidates.extend(
        [
            dataset_path.parent / raw_path,
            dataset_path.parent.parent / raw_path,
            Path.cwd() / raw_path,
        ]
    )

    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return str(candidate.resolve())

    candidate_str = [str(path) for path in unique_candidates]
    raise FileNotFoundError(
        f"Unable to resolve dataset image path '{raw_image_path}'. "
        f"Tried: {candidate_str}"
    )


def default_dataset_output_path(default_output_dir: Path, dataset_path: Path) -> Path:
    return default_output_dir / f"agent_{dataset_path.stem}.json"


def default_failed_debug_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_failed_debug.jsonl"


def atomic_json_dump(data: Any, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    log(f"Writing snapshot to temporary file {temp_path}.")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, output_path)
    log(f"Snapshot committed to {output_path}.")


def append_failed_debug(failed_debug_path: Path, payload: Dict[str, Any]) -> None:
    failed_debug_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_debug_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log(
        "Recorded failed annotation. "
        f"sample_idx={payload['sample_idx']}, annotation_idx={payload['annotation_idx']}, "
        f"debug_file={failed_debug_path}"
    )


def build_failed_payload(
    *,
    sample_idx: int,
    ann_idx: int,
    image: str,
    prompt: Any,
    error: str,
) -> Dict[str, Any]:
    return {
        "sample_idx": sample_idx,
        "annotation_idx": ann_idx,
        "image": image,
        "text": prompt,
        "error": error,
    }


PROCESSED_ANSWER_MARKER = "processed"
USE_ORIGIN_ANSWER_MARKER = "use_origin"


def is_completed_annotation(
    annotation: Dict[str, Any],
    *,
    image: str,
    prompt: Any,
) -> bool:
    answer = annotation.get("answer")
    if answer in {PROCESSED_ANSWER_MARKER, USE_ORIGIN_ANSWER_MARKER}:
        return True

    # Backward compatibility for older structured answer payloads.
    if isinstance(answer, dict):
        return str(answer.get("status", "")).lower() == "success"
    if isinstance(answer, str) and answer.strip():
        try:
            payload = json.loads(answer)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and str(payload.get("status", "")).lower() == "success"
    return False


def prediction_to_ann(prediction: Dict[str, Any]) -> List[str]:
    orig_h = int(prediction["orig_img_h"])
    orig_w = int(prediction["orig_img_w"])
    return [
        json.dumps(
            {"size": [orig_h, orig_w], "counts": counts},
            ensure_ascii=False,
        )
        for counts in prediction.get("pred_masks", [])
    ]


def should_preserve_original_ann(prediction: Dict[str, Any]) -> bool:
    pred_masks = prediction.get("pred_masks", [])
    return len(pred_masks) == 0


def clone_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cloned: List[Dict[str, Any]] = []
    for sample in dataset:
        sample_copy = dict(sample)
        sample_copy["annotation"] = [
            dict(annotation) for annotation in sample.get("annotation", [])
        ]
        cloned.append(sample_copy)
    return cloned
