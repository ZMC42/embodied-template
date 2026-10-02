#!/usr/bin/env python3
"""Run a micro SFT policy through RLinf in one IsaacLab environment."""

import argparse
import json
import os
import select
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_policy(checkpoint, denoising_steps):
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model
    from safetensors import safe_open

    cfg = OmegaConf.load(
        ROOT / "third_party/RLinf/examples/embodiment/config/model/gr00t_n1d7.yaml"
    )
    cfg.model_path = str(checkpoint)
    cfg.backbone_model_path = str(ROOT / "models/nvidia/Cosmos-Reason2-2B")
    cfg.obs_converter_type = "isaaclab_stack_cube"
    cfg.embodiment_tag = "libero_sim"
    cfg.add_value_head = False
    cfg.num_action_chunks = 16
    cfg.denoising_steps = denoising_steps
    policy = get_model(cfg)
    policy.eval()
    policy.to("cuda")
    processor = policy._modality_transform
    persisted = json.loads((checkpoint / "statistics.json").read_text())
    assert (
        processor.state_action_processor.statistics["libero_sim"]
        == persisted["libero_sim"]
    )
    mapping = json.loads((checkpoint / "embodiment_id.json").read_text())
    assert processor.embodiment_id_mapping["libero_sim"] == mapping["libero_sim"]
    state_dim = sum(
        len(persisted["libero_sim"]["state"][k]["mean"])
        for k in persisted["libero_sim"]["state"]
    )
    assert state_dim == 8
    # Check an actually updated SFT tensor survived the RL action-head replacement.
    name = "action_head.action_decoder.layer2.W"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    with safe_open(checkpoint / index["weight_map"][name], framework="pt") as weights:
        expected = weights.get_tensor(name).to(device="cuda", dtype=torch.bfloat16)
    assert torch.equal(policy.state_dict()[name], expected)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models/stack-cube-n1.7-sft/checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=2)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--offline-load", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/stack-cube/integration/closed-loop"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.pop("DISPLAY", None)
    started = time.monotonic()
    print("STARTING", args.output, flush=True)
    if args.offline_load:
        assert socket.if_nameindex() == [(1, "lo")]
        assert (
            os.readlink("/proc/self/ns/net")
            != os.environ["INTEGRATION_HOST_NETWORK_NAMESPACE"]
        )
        import numpy as np
        import torch
        from PIL import Image

        policy = load_policy(args.checkpoint, args.denoising_steps)
        runtime = json.loads(
            (ROOT / "runs/stack-cube/contract/runtime.json").read_text()
        )
        observation = {
            "states": torch.tensor(runtime["raw_wrapper_state"]),
            "main_images": torch.from_numpy(
                np.array(
                    Image.open(ROOT / "runs/stack-cube/contract/runtime-image.png")
                )
            )[None],
            "wrist_images": torch.from_numpy(
                np.array(
                    Image.open(
                        ROOT / "runs/stack-cube/contract/runtime-wrist_image.png"
                    )
                )
            )[None],
            "task_descriptions": [
                "Stack the red block on the blue block, then stack the green block on the red block."
            ],
        }
        action, _ = policy.predict_action_batch(observation, mode="eval")
        assert action.shape == (1, 16, 7) and np.isfinite(action).all()
        report = {
            "status": "passed",
            "interfaces": socket.if_nameindex(),
            "checkpoint": str(args.checkpoint),
            "action_shape": list(action.shape),
            "action": action.tolist(),
            "wall_time_s": time.monotonic() - started,
        }
        (args.output / "offline.json").write_text(json.dumps(report, indent=2) + "\n")
        print("OFFLINE_LOAD_PASSED", flush=True)
        return

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=True)
    app = launcher.app
    print("APP_READY", flush=True)
    import gymnasium as gym
    import imageio.v2 as imageio
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from isaaclab_tasks.utils import load_cfg_from_registry
    from PIL import Image, ImageDraw
    from rlinf.envs.action_utils import prepare_actions_for_isaaclab
    from rlinf.envs.sim.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv
    from scipy.spatial.transform import Rotation

    task = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.seed, cfg.scene.num_envs = args.seed, 1
    for camera in (cfg.scene.table_cam, cfg.scene.wrist_cam):
        camera.height = camera.width = 256
    env = gym.make(task, cfg=cfg, render_mode="rgb_array").unwrapped
    writer = imageio.get_writer(str(args.output / f"seed-{args.seed}.mp4"), fps=20)
    records = []
    wrapper = SimpleNamespace(
        num_envs=1,
        task_description="Stack the red block on the blue block, then stack the green block on the red block.",
    )
    paused, single_step, camera_name = args.interactive, False, "table"
    success, step, chunks = False, 0, 0
    pending = []

    def set_camera(name):
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        viewport.updates_enabled = True
        viewport.freeze_frame = False
        viewport.set_active_camera(
            "/World/envs/env_0/table_cam"
            if name == "table"
            else "/World/envs/env_0/Robot/panda_hand/wrist_cam"
        )

    def record(obs, reward):
        front = obs["policy"]["table_cam"][0].cpu().numpy()
        wrist = obs["policy"]["wrist_cam"][0].cpu().numpy()
        frame = Image.fromarray(np.concatenate([front, wrist], axis=1))
        ImageDraw.Draw(frame).text(
            (6, 6),
            f"SFT closed-loop | seed={args.seed} step={step} reward={reward:.1f} success={success}",
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
        writer.append_data(np.asarray(frame))

    try:
        obs, _ = env.reset(seed=args.seed)
        print("ENV_RESET", flush=True)
        arm = env.action_manager.get_term("arm_action")
        assert env.action_manager.total_action_dim == 7
        assert arm.cfg.scale == 0.5 and arm.cfg.controller.use_relative_mode
        assert arm.cfg.controller.command_type == "pose"
        torch.manual_seed(args.seed)
        policy = load_policy(args.checkpoint, args.denoising_steps)
        print("POLICY_READY", flush=True)
        record(obs, 0)
        if args.interactive:
            set_camera(camera_name)
            print(
                "STREAM_READY: pause/play, step, reset, table, wrist, quit (enter a command in the server terminal)",
                flush=True,
            )
        while app.is_running():
            frame_started = time.monotonic()
            if args.interactive:
                if paused:
                    env.sim.render()
                if select.select([sys.stdin], [], [], 0)[0]:
                    command = sys.stdin.readline().strip()
                    if command == "quit":
                        break
                    if command in ("pause", "play"):
                        paused = command == "pause"
                    elif command == "step":
                        paused, single_step = True, True
                    elif command == "reset":
                        obs, _ = env.reset(seed=args.seed)
                        pending = []
                        chunks = 0
                        torch.manual_seed(args.seed)
                        success = False
                    elif command in ("table", "wrist"):
                        camera_name = command
                        set_camera(command)
                if paused and not single_step:
                    time.sleep(0.01)
                    continue
            elif chunks >= args.chunks and not pending:
                break
            if not pending:
                wrapped = IsaaclabStackCubeEnv._wrap_obs(wrapper, obs)
                # Verify runtime principal branch against the simulator quaternion.
                converted = policy.obs_convert_fn(wrapped)
                state_rotation = np.concatenate(
                    [converted[f"state.{k}"][:, 0] for k in ("roll", "pitch", "yaw")],
                    axis=-1,
                )
                quat = obs["policy"]["eef_quat"][:, [1, 2, 3, 0]].cpu().numpy()
                np.testing.assert_allclose(
                    state_rotation, Rotation.from_quat(quat).as_rotvec(), atol=1e-6
                )
                tick = time.monotonic()
                action, _ = policy.predict_action_batch(wrapped, mode="eval")
                torch.cuda.synchronize()
                latency = time.monotonic() - tick
                assert action.shape == (1, 16, 7) and np.isfinite(action).all()
                pending = list(prepare_actions_for_isaaclab(action, "gr00t_n1d7")[0])
                chunks += 1
                print(f"CHUNK {chunks} inference_s={latency:.3f}", flush=True)
            action = pending.pop(0).unsqueeze(0).to(env.device)
            tick = time.monotonic()
            simulation_time_before = env.sim.current_time
            obs, reward, terminated, truncated, _ = env.step(action)
            sim_time = time.monotonic() - tick
            assert env.sim.current_time > simulation_time_before, (
                "Physics did not advance"
            )
            step += 1
            value = reward.item()
            success |= value > 0
            record(obs, value)
            records.append(
                {
                    "step": step,
                    "action": action[0].tolist(),
                    "reward": value,
                    "success_once": success,
                    "terminated": terminated.item(),
                    "truncated": truncated.item(),
                    "inference_s": latency,
                    "simulation_step_s": sim_time,
                    "simulation_time_s": env.sim.current_time,
                    "eef_state": IsaaclabStackCubeEnv._wrap_obs(wrapper, obs)["states"][
                        0
                    ].tolist(),
                    "actual_fps": 1 / (time.monotonic() - frame_started),
                }
            )
            print(json.dumps(records[-1]), flush=True)
            single_step = False
        report = {
            "status": "passed",
            "stage": "SFT closed-loop via RLinf; micro checkpoint",
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "livestream": os.environ.get("LIVESTREAM", "0"),
            "public_ip": os.environ.get("PUBLIC_IP"),
            "task_id": task,
            "steps": records,
            "wall_time_s": time.monotonic() - started,
            "success_once": success,
        }
        (args.output / "rollout.json").write_text(json.dumps(report, indent=2) + "\n")
        print("CLOSED_LOOP_PASSED", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        writer.close()
        env.close()
        app.close()


if __name__ == "__main__":
    main()
