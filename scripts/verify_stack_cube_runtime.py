#!/usr/bin/env python3
# ruff: noqa: E402
# Isaac Sim must start before importing task modules.
"""Check the real IsaacLab observation and action managers without loading a policy."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.pop("DISPLAY", None)
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils import load_cfg_from_registry
from PIL import Image
from rlinf.envs.sim.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv
from scipy.spatial.transform import Rotation

from embodied_template.stack_cube import (
    CONTRACT_PATH,
    canonical_axis_angle,
    read_json,
    write_json,
)


def verify():
    contract = read_json(CONTRACT_PATH)
    lock = read_json(ROOT / "configs/dependencies.lock.json")
    cfg = load_cfg_from_registry(
        lock["runtime"]["target_task_id"], "env_cfg_entry_point"
    )
    cfg.seed, cfg.scene.num_envs = 0, 1
    # Default task is 200x200; enforce the dataset's measured resolution.
    for key in ["table_cam", "wrist_cam"]:
        camera = getattr(cfg.scene, key)
        camera.height = camera.width = 256
    assert cfg.sim.dt * cfg.decimation == 0.05
    cameras = {}
    for key in ["image", "wrist_image"]:
        expected = contract["cameras"][key]
        actual = getattr(cfg.scene, expected["runtime"])
        np.testing.assert_allclose(actual.offset.pos, expected["position"])
        np.testing.assert_allclose(actual.offset.rot, expected["quaternion_wxyz"])
        assert actual.offset.convention == "ros"
        assert (actual.spawn.focal_length, actual.spawn.horizontal_aperture) == (
            24.0,
            20.955,
        )
        cameras[key] = {
            "position": actual.offset.pos,
            "quaternion_wxyz": actual.offset.rot,
            "resolution": [actual.height, actual.width],
        }
    env = gym.make(lock["runtime"]["target_task_id"], cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        policy = obs["policy"]
        wrapped = IsaaclabStackCubeEnv._wrap_obs(
            SimpleNamespace(num_envs=1, task_description="stack cubes"), obs
        )
        state = wrapped["states"].cpu().numpy()
        assert state.shape == (1, 8) and state.dtype == np.float32
        np.testing.assert_array_equal(state[:, :3], policy["eef_pos"].cpu().numpy())
        quat = policy["eef_quat"][:, [1, 2, 3, 0]].cpu().numpy()
        principal = Rotation.from_quat(quat).as_rotvec()
        np.testing.assert_allclose(
            canonical_axis_angle(state[:, 3:6]), principal, atol=1e-6
        )
        robot = env.scene["robot"]
        ids, _ = robot.find_joints("panda_finger_.*")
        fingers = robot.data.joint_pos[:, ids].cpu().numpy()
        np.testing.assert_allclose(state[:, 6:], fingers * np.array([1, -1]), atol=1e-7)
        report_dir = ROOT / "runs/stack-cube/contract"
        report_dir.mkdir(parents=True, exist_ok=True)
        for image, key in [("main_images", "image"), ("wrist_images", "wrist_image")]:
            tensor = wrapped[image]
            assert tensor.shape == (1, 256, 256, 3) and tensor.dtype == torch.uint8
            assert torch.equal(tensor, policy[contract["cameras"][key]["runtime"]])
            Image.fromarray(tensor[0].cpu().numpy()).save(
                report_dir / f"runtime-{key}.png"
            )
        manager = env.action_manager
        assert manager.total_action_dim == 7
        assert manager.active_terms == ["arm_action", "gripper_action"]
        arm = manager.get_term("arm_action")
        gripper = manager.get_term("gripper_action")
        assert arm.cfg.scale == 0.5 and arm.cfg.clip is None
        assert arm.cfg.controller.use_relative_mode is True
        assert arm.cfg.controller.command_type == "pose"
        commands = []
        for dimension in range(6):
            for sign in [-1, 1]:
                command = torch.zeros((1, 7), device=env.device)
                command[0, dimension] = sign * 0.02
                command[0, 6] = 1
                manager.process_action(command)
                np.testing.assert_allclose(
                    arm.processed_actions.cpu().numpy(),
                    command[:, :6].cpu().numpy() * 0.5,
                    atol=1e-7,
                )
                position, quaternion = arm._compute_frame_pose()
                controller = arm._ik_controller
                delta = command[0, :6].cpu().numpy() * 0.5
                np.testing.assert_allclose(
                    controller.ee_pos_des.cpu().numpy(),
                    position.cpu().numpy() + delta[:3],
                    atol=1e-7,
                )
                current = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]].cpu().numpy())
                expected = Rotation.from_rotvec(delta[3:]) * current
                target = Rotation.from_quat(
                    controller.ee_quat_des[:, [1, 2, 3, 0]].cpu().numpy()
                )
                np.testing.assert_allclose(
                    target.as_matrix(), expected.as_matrix(), atol=1e-6
                )
                commands.append(command[0].tolist())
        gripper_targets = {}
        for sign, target in [(-1, 0.0), (0, 0.04), (1, 0.04)]:
            command = torch.zeros((1, 7), device=env.device)
            command[0, 6] = sign
            manager.process_action(command)
            np.testing.assert_allclose(
                gripper.processed_actions.cpu().numpy(), [[target, target]], atol=1e-7
            )
            gripper_targets[str(sign)] = gripper.processed_actions.tolist()
        # Exercise actual application of both signs, while keeping the arm at its pose.
        achieved = {}
        for sign in [-1, 1]:
            command = torch.zeros((1, 7), device=env.device)
            command[0, 6] = sign
            for _ in range(20):
                next_obs, _, _, _, _ = env.step(command)
            achieved[str(sign)] = next_obs["policy"]["gripper_pos"].tolist()
        assert abs(achieved["-1"][0][0]) < 0.005
        assert achieved["1"][0][0] > 0.035 and achieved["1"][0][1] < -0.035
        report = {
            "status": "passed",
            "task_id": lock["runtime"]["target_task_id"],
            "source_commits": {k: v["commit"] for k, v in lock["sources"].items()},
            "seed": 0,
            "state_dim": 8,
            "action_dim": 7,
            "raw_wrapper_state": state.tolist(),
            "principal_axis_angle": principal.tolist(),
            "quaternion_xyzw": quat.tolist(),
            "gripper_joint_positions": fingers.tolist(),
            "cameras": cameras,
            "camera_capture_calibration": (
                "not supplied by public dataset; runtime matches pinned task config"
            ),
            "tested_arm_commands": commands,
            "arm_scale": 0.5,
            "gripper_targets": gripper_targets,
            "gripper_after_20_steps": achieved,
        }
        write_json(report_dir / "runtime.json", report)
        print("STACK_CUBE_RUNTIME", json.dumps(report))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        verify()
    finally:
        app.close()
