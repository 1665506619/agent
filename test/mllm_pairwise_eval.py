"""MLLM scoring for before/after SAM3-agent outputs.

This script scores two result files with the same dataset structure by sending
both candidate segmentations together to an MLLM judge, then compares the
returned scores locally.

Primary expected format:
- ``annotation[].ann`` directly stores predicted masks using the same schema as
  the original dataset: a list of JSON strings with ``size`` and ``counts``.

Backward compatibility is kept for earlier experimental outputs.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pycocotools.mask as mask_utils
from openai import OpenAI
from PIL import Image, ImageDraw


SYSTEM_PROMPT = """You are an expert judge for referring image segmentation.
You will be given:
- the raw image
- the original natural-language segmentation query
- segmentation 1 visualization
- segmentation 2 visualization

Judge how well each segmentation satisfies the query.

Scoring rules:
- Use 1 to 10 integers.
- Higher is better.
- Focus on semantic correctness, spatial precision, completeness, and avoiding extra regions.
- Prefer a precise smaller correct mask over a large wrong region.
- Prefer "no mask" only when the segmentation is unsupported by the query/image.
- Treat segmentation 1 and segmentation 2 symmetrically. Do not assume either one is better.

Return exactly one JSON object and nothing else:
{
  "segmentation_1": {
    "score": 1-10,
    "confidence": 1-5,
    "reason": "short reason"
  },
  "segmentation_2": {
    "score": 1-10,
    "confidence": 1-5,
    "reason": "short reason"
  }
}
"""


_LAST_JUDGE_REQUEST_TS = 0.0


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[mllm_pairwise_eval {timestamp}] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use an MLLM to compare before/after segmentation outputs."
    )
    parser.add_argument("--before", required=True, help="Before-upgrade result json.")
    parser.add_argument("--after", required=True, help="After-upgrade result json.")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root directory used to resolve relative image paths.",
    )
    parser.add_argument(
        "--origin-dataset",
        default=None,
        help="Optional origin dataset for fallback image/query resolution.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the OpenAI-compatible MLLM endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.siliconflow.cn/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="MLLM name used as the judge.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to save the evaluation summary json.",
    )
    parser.add_argument(
        "--render-dir",
        default=None,
        help="Optional directory to save rendered candidate images for auditing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of annotation pairs to evaluate.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=600,
        help="Max completion tokens for the judge model.",
    )
    parser.add_argument(
        "--request-interval-s",
        type=float,
        default=3.0,
        help="Minimum sleep interval between judge MLLM requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry count for rate-limited judge requests.",
    )
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=8.0,
        help="Initial backoff seconds for 429/rate-limit judge retries.",
    )
    return parser.parse_args()


def _load_json(path: str) -> Any:
    _log(f"Loading json from {path}.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        _log(f"Loaded dataset-style json with {len(data)} samples from {path}.")
    else:
        _log(f"Loaded non-list json of type {type(data).__name__} from {path}.")
    return data


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_first_json(text: str) -> Dict[str, Any]:
    stripped = _strip_code_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Failed to parse JSON from model output: {text}")
    return json.loads(stripped[start : end + 1])


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_ann_list(ann: Any) -> Dict[str, Any]:
    if not isinstance(ann, list):
        return {"status": "missing", "error": "ann is not a list", "prediction": {}}

    pred_masks: List[str] = []
    errors: List[str] = []
    for item in ann:
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                errors.append(f"bad ann item: {item[:80]}")
                continue
        elif isinstance(item, dict):
            parsed = item
        else:
            errors.append(f"unsupported ann item type: {type(item).__name__}")
            continue

        counts = parsed.get("counts")
        if isinstance(counts, str):
            pred_masks.append(counts)
        else:
            errors.append(f"missing counts field in ann item: {parsed}")

    status = "success" if not errors else "error"
    return {
        "status": status,
        "error": "; ".join(errors) if errors else None,
        "prediction": {"pred_masks": pred_masks},
    }


def _normalize_status(status: Optional[str]) -> str:
    if not status:
        return "success"
    return str(status).lower()


def _parse_answer_payload(answer: Any) -> Optional[Dict[str, Any]]:
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str) or not answer.strip():
        return None
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": f"Unparseable answer string: {answer[:200]}",
        }
    if isinstance(payload, dict):
        return payload
    return {
        "status": "error",
        "error": f"Unsupported answer payload type: {type(payload).__name__}",
    }


def _canonicalize_annotation_prediction(
    annotation: Dict[str, Any],
) -> Dict[str, Any]:
    ann_payload = _parse_ann_list(annotation.get("ann"))
    if ann_payload["status"] == "success":
        return {
            "status": "success",
            "error": None,
            "resolved_image_path": None,
            "text_prompt": annotation.get("text"),
            "prediction": ann_payload["prediction"],
        }
    if ann_payload["prediction"].get("pred_masks"):
        return {
            "status": "success",
            "error": ann_payload.get("error"),
            "resolved_image_path": None,
            "text_prompt": annotation.get("text"),
            "prediction": ann_payload["prediction"],
        }

    payload = _parse_answer_payload(annotation.get("answer"))
    if payload is None and isinstance(annotation.get("agent_result"), dict):
        payload = dict(annotation["agent_result"])

    if payload is None:
        return {
            "status": "missing",
            "error": "No prediction payload found in answer or agent_result.",
        }

    status = _normalize_status(payload.get("status"))
    if "prediction" in payload and isinstance(payload["prediction"], dict):
        prediction = dict(payload["prediction"])
    else:
        prediction = {
            k: payload.get(k)
            for k in (
                "pred_masks",
                "pred_boxes",
                "pred_scores",
                "original_image_path",
                "image_path",
                "text_prompt",
                "output_json_path",
                "output_image_path",
                "agent_history_path",
            )
            if k in payload
        }

    return {
        "status": status,
        "error": payload.get("error"),
        "resolved_image_path": payload.get("resolved_image_path"),
        "text_prompt": payload.get("text_prompt") or annotation.get("text"),
        "prediction": prediction,
    }


def _resolve_image_path(
    raw_image_path: Optional[str],
    *,
    image_root: Optional[str],
    cwd: Path,
) -> str:
    if not raw_image_path:
        raise FileNotFoundError("Missing image path.")

    raw_path = Path(raw_image_path)
    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    if image_root:
        candidates.append(Path(image_root) / raw_path)
    candidates.append(cwd / raw_path)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Unable to resolve image path '{raw_image_path}' from {candidates}"
    )


def _get_effective_image_path(
    sample: Dict[str, Any],
    annotation_prediction: Dict[str, Any],
    *,
    image_root: Optional[str],
    cwd: Path,
) -> str:
    prediction = annotation_prediction.get("prediction", {})
    candidates = [
        prediction.get("original_image_path"),
        prediction.get("image_path"),
        annotation_prediction.get("resolved_image_path"),
        sample.get("image"),
    ]
    last_error = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return _resolve_image_path(candidate, image_root=image_root, cwd=cwd)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError("No usable image path found.")


def _overlay_masks(base_image: Image.Image, pred_masks: List[str]) -> Image.Image:
    rendered = np.array(base_image.convert("RGB"), dtype=np.uint8)
    base_rgb = rendered.copy()
    colors = [
        np.array([255, 99, 71], dtype=np.float32),
        np.array([30, 144, 255], dtype=np.float32),
        np.array([50, 205, 50], dtype=np.float32),
        np.array([255, 215, 0], dtype=np.float32),
    ]
    # Dim the non-mask area so the candidate region stands out without
    # destroying the texture inside the mask.
    rendered = (rendered.astype(np.float32) * 0.28).astype(np.uint8)
    height, width = rendered.shape[:2]
    outline_map = np.zeros((height, width), dtype=bool)
    label_positions: List[Tuple[int, int, int]] = []

    for idx, counts in enumerate(pred_masks):
        if not counts:
            continue
        rle = {"size": [height, width], "counts": counts}
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(bool)
        if not mask.any():
            continue

        mask_u8 = mask.astype(np.uint8)
        padded = np.pad(mask_u8, 1, mode="constant")
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        interior = mask & up.astype(bool) & down.astype(bool) & left.astype(bool) & right.astype(bool)
        boundary = mask & (~interior)

        padded_boundary = np.pad(boundary.astype(np.uint8), 1, mode="constant")
        boundary_thick = (
            padded_boundary[1:-1, 1:-1]
            | padded_boundary[:-2, 1:-1]
            | padded_boundary[2:, 1:-1]
            | padded_boundary[1:-1, :-2]
            | padded_boundary[1:-1, 2:]
            | padded_boundary[:-2, :-2]
            | padded_boundary[:-2, 2:]
            | padded_boundary[2:, :-2]
            | padded_boundary[2:, 2:]
        ).astype(bool)

        color = colors[idx % len(colors)]
        rendered[mask] = (
            base_rgb[mask].astype(np.float32) * 0.7 + color * 0.3
        ).astype(np.uint8)
        outline_map |= boundary_thick

        ys, xs = np.nonzero(mask)
        if len(xs) > 0 and len(ys) > 0:
            label_positions.append((int(xs.mean()), int(ys.mean()), idx + 1))

    rendered[outline_map] = np.array([255, 255, 255], dtype=np.uint8)
    image = Image.fromarray(rendered)
    draw = ImageDraw.Draw(image)
    for center_x, center_y, label_id in label_positions:
        radius = 13
        draw.ellipse(
            [
                (center_x - radius, center_y - radius),
                (center_x + radius, center_y + radius),
            ],
            fill=(15, 15, 15),
            outline=(255, 255, 255),
            width=2,
        )
        draw.text((center_x - 4, center_y - 8), str(label_id), fill=(255, 255, 255))
    return image


def _annotate_banner(image: Image.Image, text: str, color: Tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (image.width, 42)], fill=color)
    draw.text((12, 10), text, fill=(255, 255, 255))


def _annotate_judge_result(
    image: Image.Image,
    label: str,
    judge_result: Dict[str, Any],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    score = _coerce_int(judge_result.get("score", 0))
    confidence = _coerce_int(judge_result.get("confidence", 0))
    reason = str(judge_result.get("reason", "")).strip().replace("\n", " ")
    reason = reason[:160]
    text = (
        f"{label}\n"
        f"score={score}  confidence={confidence}\n"
        f"reason={reason}"
    )
    banner_height = 88
    draw.rectangle(
        [(0, annotated.height - banner_height), (annotated.width, annotated.height)],
        fill=(20, 20, 20),
    )
    draw.multiline_text(
        (12, annotated.height - banner_height + 8),
        text,
        fill=(255, 255, 255),
        spacing=4,
    )
    return annotated


def _image_output_dir(render_root: Path, image_name: str, ann_idx: int) -> Path:
    image_rel = Path(image_name)
    image_stem = image_rel.with_suffix("")
    return render_root / image_stem / f"ann_{ann_idx:02d}"


def _render_candidate_image(
    image_path: str,
    annotation_prediction: Dict[str, Any],
) -> Image.Image:
    base_image = Image.open(image_path).convert("RGB")
    status = annotation_prediction.get("status", "missing")
    prediction = annotation_prediction.get("prediction", {}) or {}
    pred_masks = prediction.get("pred_masks") or []

    if status != "success":
        rendered = base_image.copy()
        _annotate_banner(rendered, f"ERROR: {status}", (180, 50, 50))
        return rendered

    if not pred_masks:
        rendered = base_image.copy()
        _annotate_banner(rendered, "NO MASK", (120, 120, 120))
        return rendered

    rendered = _overlay_masks(base_image, pred_masks)
    _annotate_banner(rendered, f"{len(pred_masks)} mask(s)", (30, 30, 30))
    return rendered


def _image_to_data_url(image: Image.Image) -> str:
    max_side = 1024
    width, height = image.size
    scale = min(1.0, max_side / float(max(width, height)))
    if scale < 1.0:
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize(
            (int(width * scale), int(height * scale)),
            resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _candidate_summary(annotation_prediction: Dict[str, Any]) -> str:
    status = annotation_prediction.get("status", "missing")
    prediction = annotation_prediction.get("prediction", {}) or {}
    pred_masks = prediction.get("pred_masks") or []
    if status != "success":
        return f"status={status}, error={annotation_prediction.get('error')}"
    return f"status=success, num_masks={len(pred_masks)}"


def _prediction_bucket(annotation_prediction: Dict[str, Any]) -> str:
    status = annotation_prediction.get("status", "missing")
    pred_masks = (annotation_prediction.get("prediction", {}) or {}).get("pred_masks") or []
    if status != "success":
        return "error_or_missing"
    if len(pred_masks) == 0:
        return "empty_mask"
    return "non_empty_mask"


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _build_side_stats(details: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    scores = [detail[f"{side}_score"] for detail in details]
    non_empty = sum(1 for detail in details if detail[f"{side}_bucket"] == "non_empty_mask")
    empty = sum(1 for detail in details if detail[f"{side}_bucket"] == "empty_mask")
    error = sum(1 for detail in details if detail[f"{side}_bucket"] == "error_or_missing")
    return {
        "count": len(details),
        "avg_score": round(sum(scores) / max(len(scores), 1), 4),
        "median_score": round(_median(scores), 4),
        "non_empty_mask_count": non_empty,
        "empty_mask_count": empty,
        "error_or_missing_count": error,
        "non_empty_mask_rate": _safe_ratio(non_empty, len(details)),
        "empty_mask_rate": _safe_ratio(empty, len(details)),
        "error_or_missing_rate": _safe_ratio(error, len(details)),
    }


def _top_delta_examples(
    details: List[Dict[str, Any]],
    *,
    positive: bool,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    filtered = [detail for detail in details if (detail["score_delta"] > 0) == positive and detail["score_delta"] != 0]
    ordered = sorted(
        filtered,
        key=lambda detail: abs(detail["score_delta"]),
        reverse=True,
    )
    return [
        {
            "sample_idx": detail["sample_idx"],
            "annotation_idx": detail["annotation_idx"],
            "image": detail["image"],
            "query": detail["query"],
            "before_score": detail["before_score"],
            "after_score": detail["after_score"],
            "score_delta": detail["score_delta"],
            "winner": detail["winner"],
        }
        for detail in ordered[:top_k]
    ]


def _judge_pair(
    *,
    client: OpenAI,
    model: str,
    query: str,
    raw_image: Image.Image,
    segmentation_1_image: Image.Image,
    segmentation_2_image: Image.Image,
    segmentation_1_summary: str,
    segmentation_2_summary: str,
    max_completion_tokens: int,
    request_interval_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    global _LAST_JUDGE_REQUEST_TS

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Query: {query}\n"
                        f"Segmentation 1 summary: {segmentation_1_summary}\n"
                        f"Segmentation 2 summary: {segmentation_2_summary}\n"
                        "Please judge both segmentations using the raw image as reference."
                    ),
                },
                {
                    "type": "text",
                    "text": "Raw image:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(raw_image), "detail": "high"},
                },
                {
                    "type": "text",
                    "text": "Segmentation 1:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_url(segmentation_1_image),
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": "Segmentation 2:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_url(segmentation_2_image),
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    _log(
        "Sending judge request. "
        f"model={model}, query={query!r}, "
        f"segmentation_1_summary={segmentation_1_summary}, "
        f"segmentation_2_summary={segmentation_2_summary}"
    )
    for attempt in range(max_retries + 1):
        now = time.monotonic()
        elapsed = now - _LAST_JUDGE_REQUEST_TS
        if elapsed < request_interval_s:
            sleep_s = request_interval_s - elapsed
            _log(f"Sleeping {sleep_s:.2f}s before next judge request.")
            time.sleep(sleep_s)

        try:
            _log(
                f"Calling judge model... attempt={attempt + 1}/{max_retries + 1}"
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=0,
                n=1,
            )
            _LAST_JUDGE_REQUEST_TS = time.monotonic()
            content = response.choices[0].message.content or ""
            _log("Received judge response successfully.")
            raw_result = _extract_first_json(content)
            seg1 = raw_result.get("segmentation_1")
            seg2 = raw_result.get("segmentation_2")
            if not isinstance(seg1, dict) or not isinstance(seg2, dict):
                raise ValueError(
                    "Judge output must contain dict fields 'segmentation_1' and "
                    f"'segmentation_2'. Got: {raw_result}"
                )
            return seg1, seg2, raw_result
        except Exception as exc:
            _LAST_JUDGE_REQUEST_TS = time.monotonic()
            error_text = str(exc)
            _log(f"Judge request failed: {error_text}")
            if attempt >= max_retries:
                raise
            lowered = error_text.lower()
            if "429" not in lowered and "rate limit" not in lowered:
                raise
            backoff = retry_backoff_s * (2**attempt)
            _log(
                f"Judge request rate limited. Retrying after {backoff:.2f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(backoff)

    raise RuntimeError("Judge request retry loop exited unexpectedly.")


def _save_render(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    _log(f"Saved render to {output_path}.")


def _ensure_same_shape(
    before_data: List[Dict[str, Any]],
    after_data: List[Dict[str, Any]],
) -> None:
    _log("Checking before/after dataset shapes.")
    if len(before_data) != len(after_data):
        raise ValueError(
            f"Sample count mismatch: before={len(before_data)}, after={len(after_data)}"
        )
    for idx, (before_sample, after_sample) in enumerate(zip(before_data, after_data)):
        before_len = len(before_sample.get("annotation", []))
        after_len = len(after_sample.get("annotation", []))
        if before_len != after_len:
            raise ValueError(
                f"Annotation count mismatch at sample {idx}: "
                f"before={before_len}, after={after_len}"
            )
    _log("Before/after dataset shapes match.")


def main() -> None:
    args = _parse_args()
    api_key = args.api_key
    if not api_key:
        raise ValueError("Missing --api-key.")

    before_data = _load_json(args.before)
    after_data = _load_json(args.after)
    if not isinstance(before_data, list) or not isinstance(after_data, list):
        raise ValueError("Both before/after files must be dataset-style lists.")

    _ensure_same_shape(before_data, after_data)
    cwd = Path(args.origin_dataset).resolve().parent if args.origin_dataset else Path.cwd()
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else Path("sam3/agent/test/mllm_pairwise_eval_results.json").resolve()
    )
    render_dir = (
        Path(args.render_dir).resolve()
        if args.render_dir
        else Path(__file__).resolve().parent / "mllm_judge_renders"
    )
    render_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Using render directory: {render_dir}")
    _log(f"Output summary json path: {output_json}")

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    _log(f"Created OpenAI-compatible client. model={args.model}, base_url={args.base_url}")
    _log(
        "Judge request throttling enabled. "
        f"request_interval_s={args.request_interval_s}, "
        f"max_retries={args.max_retries}, retry_backoff_s={args.retry_backoff_s}"
    )

    details: List[Dict[str, Any]] = []
    wins = {"before": 0, "after": 0, "tie": 0, "invalid": 0}
    before_scores: List[int] = []
    after_scores: List[int] = []

    eval_count = 0
    for sample_idx, (before_sample, after_sample) in enumerate(
        zip(before_data, after_data)
    ):
        _log(
            f"Starting sample {sample_idx}: image={before_sample.get('image', '')}"
        )
        for ann_idx, (before_ann, after_ann) in enumerate(
            zip(before_sample.get("annotation", []), after_sample.get("annotation", []))
        ):
            if args.limit is not None and eval_count >= args.limit:
                break

            before_pred = _canonicalize_annotation_prediction(before_ann)
            after_pred = _canonicalize_annotation_prediction(after_ann)
            query = (
                after_pred.get("text_prompt")
                or before_pred.get("text_prompt")
                or after_ann.get("text")
                or before_ann.get("text")
            )
            _log(
                f"Scoring sample={sample_idx}, annotation={ann_idx}, query={query!r}"
            )

            image_path = _get_effective_image_path(
                after_sample,
                after_pred,
                image_root=args.image_root,
                cwd=cwd,
            )
            _log(
                f"Resolved image for sample={sample_idx}, annotation={ann_idx}: {image_path}"
            )
            raw_image = Image.open(image_path).convert("RGB")
            before_image = _render_candidate_image(image_path, before_pred)
            after_image = _render_candidate_image(image_path, after_pred)
            (
                before_judge_result,
                after_judge_result,
                judge_raw_result,
            ) = _judge_pair(
                client=client,
                model=args.model,
                query=query,
                raw_image=raw_image,
                segmentation_1_image=before_image,
                segmentation_2_image=after_image,
                segmentation_1_summary=_candidate_summary(before_pred),
                segmentation_2_summary=_candidate_summary(after_pred),
                max_completion_tokens=args.max_completion_tokens,
                request_interval_s=args.request_interval_s,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff_s,
            )

            before_score = _coerce_int(before_judge_result.get("score", 0))
            after_score = _coerce_int(after_judge_result.get("score", 0))
            score_delta = after_score - before_score
            _log(
                f"Scores for sample={sample_idx}, annotation={ann_idx}: "
                f"before={before_score}, after={after_score}, delta={score_delta}"
            )

            if before_score > after_score:
                winner = "before"
            elif after_score > before_score:
                winner = "after"
            else:
                winner = "tie"

            before_scores.append(before_score)
            after_scores.append(after_score)
            wins[winner] = wins.get(winner, 0) + 1

            detail = {
                "sample_idx": sample_idx,
                "annotation_idx": ann_idx,
                "image": before_sample.get("image"),
                "query": query,
                "before_summary": _candidate_summary(before_pred),
                "after_summary": _candidate_summary(after_pred),
                "winner": winner,
                "before_score": before_score,
                "after_score": after_score,
                "score_delta": score_delta,
                "before_bucket": _prediction_bucket(before_pred),
                "after_bucket": _prediction_bucket(after_pred),
                "before_judge_raw_result": before_judge_result,
                "after_judge_raw_result": after_judge_result,
                "judge_raw_result": judge_raw_result,
            }
            details.append(detail)

            pair_dir = _image_output_dir(
                render_dir,
                str(before_sample.get("image", f"sample_{sample_idx:05d}.jpg")),
                ann_idx,
            )
            _log(
                f"Saving render artifacts for sample={sample_idx}, annotation={ann_idx} "
                f"to {pair_dir}"
            )
            before_scored_image = _annotate_judge_result(
                before_image,
                "SEGMENTATION 1",
                before_judge_result,
            )
            after_scored_image = _annotate_judge_result(
                after_image,
                "SEGMENTATION 2",
                after_judge_result,
            )
            _save_render(raw_image, pair_dir / "raw.jpg")
            _save_render(before_image, pair_dir / "before_sent.jpg")
            _save_render(after_image, pair_dir / "after_sent.jpg")
            _save_render(before_scored_image, pair_dir / "before_scored.jpg")
            _save_render(after_scored_image, pair_dir / "after_scored.jpg")

            eval_count += 1
        _log(f"Completed sample {sample_idx}.")
        if args.limit is not None and eval_count >= args.limit:
            break

    score_deltas = [detail["score_delta"] for detail in details]
    improved_count = sum(1 for delta in score_deltas if delta > 0)
    worsened_count = sum(1 for delta in score_deltas if delta < 0)
    tied_count = sum(1 for delta in score_deltas if delta == 0)
    large_improvement_count = sum(1 for delta in score_deltas if delta >= 2)
    large_regression_count = sum(1 for delta in score_deltas if delta <= -2)

    before_stats = _build_side_stats(details, "before")
    after_stats = _build_side_stats(details, "after")

    summary = {
        "before_path": str(Path(args.before).resolve()),
        "after_path": str(Path(args.after).resolve()),
        "model": args.model,
        "base_url": args.base_url,
        "request_interval_s": args.request_interval_s,
        "max_retries": args.max_retries,
        "retry_backoff_s": args.retry_backoff_s,
        "evaluated_pairs": len(details),
        "wins": wins,
        "win_rates": {
            "before": _safe_ratio(wins["before"], len(details)),
            "after": _safe_ratio(wins["after"], len(details)),
            "tie": _safe_ratio(wins["tie"], len(details)),
        },
        "before_stats": before_stats,
        "after_stats": after_stats,
        "avg_before_score": before_stats["avg_score"],
        "avg_after_score": after_stats["avg_score"],
        "median_before_score": before_stats["median_score"],
        "median_after_score": after_stats["median_score"],
        "avg_score_gain": round(sum(score_deltas) / max(len(score_deltas), 1), 4),
        "median_score_gain": round(_median(score_deltas), 4),
        "improved_count": improved_count,
        "worsened_count": worsened_count,
        "tied_count": tied_count,
        "improved_rate": _safe_ratio(improved_count, len(details)),
        "worsened_rate": _safe_ratio(worsened_count, len(details)),
        "tied_rate": _safe_ratio(tied_count, len(details)),
        "large_improvement_count": large_improvement_count,
        "large_regression_count": large_regression_count,
        "top_improvements": _top_delta_examples(details, positive=True),
        "top_regressions": _top_delta_examples(details, positive=False),
        "details": details,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _log(f"Saved evaluation summary to {output_json}.")

    print("\n=== Evaluation Summary ===")
    print(
        f"pairs={len(details)} | "
        f"before_avg={summary['avg_before_score']} | "
        f"after_avg={summary['avg_after_score']} | "
        f"avg_gain={summary['avg_score_gain']}"
    )
    print(
        f"wins(before/after/tie)={wins['before']}/{wins['after']}/{wins['tie']} | "
        f"improved/worsened/tied={improved_count}/{worsened_count}/{tied_count}"
    )
    print(
        "before(non_empty/empty/error)="
        f"{before_stats['non_empty_mask_count']}/"
        f"{before_stats['empty_mask_count']}/"
        f"{before_stats['error_or_missing_count']} | "
        "after(non_empty/empty/error)="
        f"{after_stats['non_empty_mask_count']}/"
        f"{after_stats['empty_mask_count']}/"
        f"{after_stats['error_or_missing_count']}"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
