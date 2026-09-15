#!/usr/bin/env python3
"""Run OpenVLA-OFT on recorded PiPER-D455 frames without hardware access."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from kinesync.models.openvla_oft import select_high_motion_indices, validate_action_chunk
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM, ROBOT_PLATFORM
from experiments.robot.openvla_utils import get_vla_action


def _component_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def _load_model(checkpoint: Path):
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model.vision_backbone.set_num_images_in_input(2)
    model.norm_stats = json.loads(
        (checkpoint / "dataset_statistics.json").read_text(encoding="utf-8")
    )
    model = model.eval().to("cuda:0")
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    action_head = L1RegressionActionHead(
        input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=ACTION_DIM
    ).to(dtype=torch.bfloat16, device="cuda:0")
    action_head.load_state_dict(
        _component_state(checkpoint / "action_head--30000_checkpoint.pt")
    )
    action_head.eval()
    proprio_projector = ProprioProjector(
        llm_dim=model.llm_dim, proprio_dim=PROPRIO_DIM
    ).to(dtype=torch.bfloat16, device="cuda:0")
    proprio_projector.load_state_dict(
        _component_state(checkpoint / "proprio_projector--30000_checkpoint.pt")
    )
    proprio_projector.eval()
    return model, processor, action_head, proprio_projector


def _episode_state_action(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        state = np.concatenate(
            (
                np.asarray(handle["left_arm/joint"], dtype=np.float64),
                np.asarray(handle["left_arm/gripper"], dtype=np.float64)[:, None],
            ),
            axis=1,
        )
        action = np.asarray(handle["left_arm/action"], dtype=np.float64)
    return state, action


def _observation(path: Path, frame_index: int) -> tuple[dict[str, object], np.ndarray]:
    with h5py.File(path, "r") as handle:
        frame_count = int(handle["left_arm/timestamp"].shape[0])
        if frame_index < 0 or frame_index >= frame_count:
            raise IndexError(f"Frame {frame_index} outside {path}")
        state = np.concatenate(
            (
                np.asarray(handle["left_arm/joint"][frame_index], dtype=np.float64),
                [float(handle["left_arm/gripper"][frame_index])],
            )
        )
        action = np.asarray(handle["left_arm/action"][frame_index], dtype=np.float64)
        observation = {
            "full_image": np.asarray(handle["cam_head/color"][frame_index], dtype=np.uint8),
            "wrist_image": np.asarray(handle["cam_wrist/color"][frame_index], dtype=np.uint8),
            "state": state,
        }
    return observation, action


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--visual-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unnorm-key", default="piper_d455_oft")
    parser.add_argument("--frames-per-task", type=int, default=10)
    parser.add_argument("--minimum-separation", type=int, default=5)
    arguments = parser.parse_args()

    if ROBOT_PLATFORM != "PIPER" or (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (10, 7, 7):
        raise RuntimeError("Patched PiPER OpenVLA-OFT constants are required")
    checkpoint = Path(arguments.checkpoint).expanduser().resolve()
    data_root = Path(arguments.data_root).expanduser().resolve()
    visual_manifest = json.loads(
        Path(arguments.visual_manifest).expanduser().resolve().read_text(encoding="utf-8")
    )
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    started = time.perf_counter()
    model, processor, action_head, proprio_projector = _load_model(checkpoint)
    load_seconds = time.perf_counter() - started
    statistics = model.norm_stats[arguments.unnorm_key]
    task_records = []
    for task in visual_manifest["tasks"]:
        source = data_root / task["task"] / f"{task['episode_id']}.hdf5"
        state, action = _episode_state_action(source)
        frame_indices = select_high_motion_indices(
            state,
            action,
            count=arguments.frames_per_task,
            minimum_separation=arguments.minimum_separation,
        )
        observations = []
        for frame_index in frame_indices:
            observation, recorded_action = _observation(source, int(frame_index))
            recorded_state = np.asarray(observation["state"], dtype=np.float64).copy()
            inference_started = time.perf_counter()
            predicted = np.asarray(
                get_vla_action(
                    type(
                        "OfflineConfig",
                        (),
                        {
                            "num_images_in_input": 2,
                            "center_crop": True,
                            "use_proprio": True,
                            "unnorm_key": arguments.unnorm_key,
                        },
                    )(),
                    model,
                    processor,
                    observation,
                    task["instruction"],
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=False,
                ),
                dtype=np.float64,
            )
            inference_seconds = time.perf_counter() - inference_started
            contract = validate_action_chunk(
                predicted, statistics["action"], envelope="min_max"
            )
            observations.append(
                {
                    "frame_index": int(frame_index),
                    "predicted_shape": list(predicted.shape),
                    "predicted_actions": predicted.tolist(),
                    "recorded_state": recorded_state.tolist(),
                    "recorded_action": recorded_action.tolist(),
                    "first_action_mae": float(np.mean(np.abs(predicted[0] - recorded_action))),
                    "hold_action_mae": float(np.mean(np.abs(recorded_state - recorded_action))),
                    "inference_seconds": inference_seconds,
                    "action_contract": contract,
                }
            )
        model_mae = np.asarray([item["first_action_mae"] for item in observations])
        hold_mae = np.asarray([item["hold_action_mae"] for item in observations])
        task_records.append(
            {
                "task": task["task"],
                "instruction": task["instruction"],
                "episode_id": task["episode_id"],
                "source_hdf5": str(source),
                "selection": "highest_recorded_action_vs_hold_mae_with_spacing",
                "frame_count": len(observations),
                "mean_first_action_mae": float(np.mean(model_mae)),
                "mean_hold_action_mae": float(np.mean(hold_mae)),
                "model_better_than_hold_count": int(np.count_nonzero(model_mae < hold_mae)),
                "observations": observations,
            }
        )
    payload = {
        "schema": "kinesync.openvla_oft_offline_acceptance.v1",
        "valid": all(
            observation["predicted_shape"] == [10, 7]
            for record in task_records
            for observation in record["observations"]
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_source": os.path.realpath(checkpoint),
        "unnorm_key": arguments.unnorm_key,
        "robot_platform": ROBOT_PLATFORM,
        "num_actions_chunk": NUM_ACTIONS_CHUNK,
        "action_dimension": ACTION_DIM,
        "proprio_dimension": PROPRIO_DIM,
        "model_load_seconds": load_seconds,
        "tasks": task_records,
        "safety": {
            "command_mode": "disabled",
            "hardware_imports": False,
            "network_server_started": False,
            "robot_commands_sent": 0,
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": payload["valid"], "output": str(output)}, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
