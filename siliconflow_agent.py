"""Backward-compatible wrapper for the renamed vLLM SAM3 agent entrypoint."""

from __future__ import annotations

from sam3.agent.vllm_agent import DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_DIR, main, parse_args
from sam3.agent.vllm_agent_dataset import (
    run_vllm_sam3_dataset as run_siliconflow_sam3_dataset,
)
from sam3.agent.vllm_agent_runtime import (
    run_vllm_sam3_agent as run_siliconflow_sam3_agent,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_OUTPUT_DIR",
    "main",
    "parse_args",
    "run_siliconflow_sam3_agent",
    "run_siliconflow_sam3_dataset",
]


if __name__ == "__main__":
    main()
