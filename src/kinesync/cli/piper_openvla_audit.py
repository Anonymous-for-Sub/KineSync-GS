"""Audit shared PiPER-D455 multitask data and OpenVLA-OFT artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from kinesync.data.piper_multitask import PiperMultitaskDataset
from kinesync.models.openvla_oft import audit_openvla_oft_checkpoint


CANONICAL_TASKS = {
    "corn_to_plate": "Put the corn onto the plate.",
    "red_cube_to_plate": "Put the red cube onto the plate.",
    "red_pen_to_bucket": "Put the red pen into the bucket.",
}


def build_integration_receipt(
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    tasks: Mapping[str, str] = CANONICAL_TASKS,
    expected_episodes_per_task: int = 40,
    unnorm_key: str = "piper_d455_oft",
) -> dict[str, object]:
    dataset = PiperMultitaskDataset(
        root=data_root,
        tasks=tasks,
        expected_episodes_per_task=expected_episodes_per_task,
    )
    dataset_receipt = dataset.audit_receipt()
    checkpoint_receipt = audit_openvla_oft_checkpoint(checkpoint, unnorm_key)
    matching_counts = (
        dataset_receipt["episode_count"] == checkpoint_receipt["num_trajectories"]
        and dataset_receipt["frame_count"] == checkpoint_receipt["num_transitions"]
    )
    return {
        "schema": "kinesync.piper_openvla_integration.v1",
        "valid": bool(
            dataset_receipt["valid"]
            and checkpoint_receipt["valid"]
            and matching_counts
        ),
        "dataset": dataset_receipt,
        "checkpoint": checkpoint_receipt,
        "cross_contract": {
            "trajectory_and_transition_counts_match": matching_counts,
            "policy_state_dimension": 7,
            "policy_action_dimension": 7,
            "gaussian_qpos_dimension": 8,
        },
        "safety": {
            "command_mode": "disabled",
            "robot_sdk_imported": False,
            "robot_command_api_called": False,
            "model_inference_performed": False,
            "scope": "offline_data_and_checkpoint_audit",
        },
    }


def write_integration_receipt(path: str | Path, receipt: Mapping[str, object]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f"{output.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only PiPER-D455 multitask and OpenVLA-OFT audit"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-episodes-per-task", type=int, default=40)
    parser.add_argument("--unnorm-key", default="piper_d455_oft")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = build_integration_receipt(
        data_root=arguments.data_root,
        checkpoint=arguments.checkpoint,
        tasks=CANONICAL_TASKS,
        expected_episodes_per_task=arguments.expected_episodes_per_task,
        unnorm_key=arguments.unnorm_key,
    )
    output = write_integration_receipt(arguments.output, receipt)
    print(json.dumps({"valid": receipt["valid"], "output": str(output)}, sort_keys=True))
    return 0 if receipt["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_TASKS",
    "build_integration_receipt",
    "main",
    "write_integration_receipt",
]
