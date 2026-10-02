#!/usr/bin/env python3
"""Check a completed baseline run and record metrics, resources and versions."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--baseline",
        choices=("isaaclab_n1_5", "n1_7_libero", "stack_cube_n1_7"),
        default="isaaclab_n1_5",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    cfg = OmegaConf.load(run_dir / "tensorboard/config.yaml")
    step = cfg.runner.max_epochs
    if cfg.runner.max_steps >= 0:
        step = min(step, cfg.runner.max_steps)

    events = EventAccumulator(
        str(run_dir / "tensorboard"), size_guidance={"scalars": 0}
    )
    events.Reload()
    metrics = {}
    for tag in events.Tags()["scalars"]:
        values = events.Scalars(tag)
        assert all(math.isfinite(value.value) for value in values), tag
        assert values[-1].step == step - 1, (tag, values[-1].step)
        if cfg.runner.resume_dir:
            resumed_step = int(
                Path(cfg.runner.resume_dir).name.removeprefix("global_step_")
            )
            assert values[0].step == resumed_step, (tag, values[0].step)
        metrics[tag] = values[-1].value
    required = (
        "env/reward",
        "env/episode_len",
        "rollout/rewards"
        if args.baseline == "isaaclab_n1_5"
        else "rollout/returns_mean",
        "train/actor/total_loss",
        "train/actor/approx_kl",
        "train/actor/grad_norm",
        "train/critic/value_loss",
    )
    assert all(tag in metrics for tag in required)
    assert metrics["train/actor/grad_norm"] > 0
    assert metrics["time/sync_weights"] > 0
    if cfg.runner.resume_dir:
        console = (run_dir / "console.log").read_text()
        assert "Resuming training from checkpoint directory" in console

    checkpoint = (
        run_dir
        / cfg.runner.logger.experiment_name
        / "checkpoints"
        / f"global_step_{step}"
        / "actor"
    )
    if cfg.actor.fsdp_config.get("save_full_model_weights", True):
        assert (checkpoint / "model_state_dict/full_weights.pt").stat().st_size > 0
    metadata = pickle.loads((checkpoint / "dcp_checkpoint/.metadata").read_bytes())
    assert any("optimizers.state." in key for key in metadata.state_dict_metadata)
    assert any("lr_schedulers." in key for key in metadata.state_dict_metadata)
    assert any("model." in key for key in metadata.state_dict_metadata)
    checkpoint_files = list(checkpoint.glob("dcp_checkpoint/*.distcp"))
    assert checkpoint_files
    assert all(path.stat().st_size > 0 for path in checkpoint_files)

    videos = []
    for path in sorted((run_dir / "video").rglob("*.mp4")):
        with av.open(str(path)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        assert frames > 0, path
        videos.append({"path": str(path.relative_to(run_dir)), "frames": frames})
    assert videos

    with (run_dir / "gpu.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    peak = max(int(row[" memory.used [MiB]"].split()[0]) for row in rows)
    fmt = "%Y/%m/%d %H:%M:%S.%f"
    wall_time = (
        datetime.strptime(rows[-1]["timestamp"], fmt).astimezone()
        - datetime.strptime(rows[0]["timestamp"], fmt).astimezone()
    ).total_seconds()
    lock = json.loads((ROOT / "configs/dependencies.lock.json").read_text())
    sources = {name: source["path"] for name, source in lock["sources"].items()}
    packages = ["torch", "torchvision", "flash-attn", "transformers", "ray", "gr00t"]
    if args.baseline == "isaaclab_n1_5":
        sources["isaac_groot_n1_5"] = lock["baselines"]["isaaclab_n1_5"][
            "gr00t_source_path"
        ]
        packages += ["isaacsim", "isaaclab"]
    elif args.baseline == "n1_7_libero":
        del sources["isaaclab"]
        sources["rlinf"] = lock["baselines"][args.baseline]["rlinf_source_path"]
        packages += ["rlinf-libero", "robosuite", "mujoco", "av"]
        assert cfg.actor.model.embodiment_tag == "libero_sim"
        assert cfg.actor.model.action_dim == 7
    else:
        packages += ["isaacsim", "isaaclab", "av"]
        assert cfg.actor.model.embodiment_tag == "libero_sim"
        assert cfg.actor.model.obs_converter_type == "isaaclab_stack_cube"
        assert cfg.actor.model.action_dim == 7
    report = {
        "status": "passed",
        "baseline": args.baseline,
        "checkpoint_step": step,
        "resumed_from": cfg.runner.resume_dir,
        "metrics": metrics,
        "peak_gpu_memory_mib": peak,
        "gpu_sampling_interval_ms": 200,
        "wall_time_seconds": wall_time,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": len({row[" index"] for row in rows}),
        "gpu_capacity_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "videos": videos,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "sources": {
            name: subprocess.check_output(
                ["git", "-C", str(ROOT / path), "rev-parse", "HEAD"], text=True
            ).strip()
            for name, path in sources.items()
        },
        "model_revision": (
            lock["assets"]["groot_n1_7_3b"]["revision"]
            if args.baseline == "stack_cube_n1_7"
            else lock["baselines"][args.baseline]["model_revision"]
        ),
    }
    if args.baseline in ("n1_7_libero", "stack_cube_n1_7"):
        from torch.distributed import checkpoint as dcp

        assert "CUDA out of memory" not in (run_dir / "console.log").read_text()
        parameter = "action_head.value_head.mlp.6.bias"
        state = {
            "fsdp_checkpoint": {
                "optimizers": {"state": {parameter: {"step": torch.tensor(0.0)}}}
            }
        }
        dcp.load(state, checkpoint_id=str(checkpoint / "dcp_checkpoint"))
        optimizer_step = int(
            state["fsdp_checkpoint"]["optimizers"]["state"][parameter]["step"].item()
        )
        assert optimizer_step == step, optimizer_step
        report["optimizer_step"] = optimizer_step
        report["backbone_revision"] = lock["assets"]["cosmos_reason2_2b"]["revision"]
        if args.baseline == "n1_7_libero":
            report["libero_assets_revision"] = lock["baselines"][args.baseline][
                "libero_assets"
            ]["revision"]
        report["checkpoint_size_bytes"] = sum(
            path.stat().st_size for path in checkpoint_files
        )
    if args.baseline == "stack_cube_n1_7":
        trajectories = []
        for path in sorted((run_dir / "trajectories").rglob("*.pkl")):
            episode = pickle.loads(path.read_bytes())
            actions = np.asarray(episode["actions"])
            assert actions.shape == (cfg.env.train.max_episode_steps, 7)
            assert np.isfinite(actions).all(), path
            assert np.isfinite(episode["rewards"]).all(), path
            states = np.asarray([obs["states"] for obs in episode["observations"]])
            assert states.shape[-1] == 8 and np.isfinite(states).all(), path
            trajectories.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "action_shape": list(actions.shape),
                    "action_min": actions.min(axis=0).tolist(),
                    "action_max": actions.max(axis=0).tolist(),
                    "state_shape": list(states.shape),
                }
            )
        assert trajectories, "Missing recorded actions and observations"
        report["trajectories"] = trajectories
        bundle = ROOT / lock["integrations"]["stack_cube_n1_7"]["bundle"]
        report["bundle_files_sha256"] = {
            name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            for name in (
                "MANIFEST.json",
                "checkpoint/config.json",
                "checkpoint/model.safetensors.index.json",
                "checkpoint/processor_config.json",
                "checkpoint/statistics.json",
                "checkpoint/embodiment_id.json",
            )
        }
        if cfg.runner.resume_dir:
            previous_run = Path(cfg.runner.resume_dir).parents[2]
            previous = json.loads((previous_run / "summary.json").read_text())
            assert previous["checkpoint_step"] == step - 1
            assert previous["optimizer_step"] == optimizer_step - 1
            assert previous["bundle_files_sha256"] == report["bundle_files_sha256"]
            assert (
                previous["trajectories"][0]["action_shape"]
                == trajectories[0]["action_shape"]
            )
            previous_episode = pickle.loads(
                (previous_run / previous["trajectories"][0]["path"]).read_bytes()
            )
            np.testing.assert_allclose(
                episode["observations"][0]["states"],
                previous_episode["observations"][0]["states"],
                atol=1e-6,
                rtol=0,
            )
            report["resume_contract"] = {
                "processor_and_bundle_match": True,
                "action_shape_match": True,
                "fixed_seed_initial_state_match": True,
                "optimizer_step_before": previous["optimizer_step"],
                "optimizer_step_after": optimizer_step,
            }
        model_config = json.loads((bundle / "checkpoint/config.json").read_text())
        report["training_scope"] = {
            key: value for key, value in model_config.items() if key.startswith("tune_")
        }
        from safetensors import safe_open

        parameter = "action_head.action_decoder.layer2.b"
        index = json.loads(
            (bundle / "checkpoint/model.safetensors.index.json").read_text()
        )
        with safe_open(
            bundle / "checkpoint" / index["weight_map"][parameter], framework="pt"
        ) as weights:
            original = weights.get_tensor(parameter).to(torch.bfloat16).float()
        state = {"fsdp_checkpoint": {"model": {parameter: torch.zeros_like(original)}}}
        dcp.load(state, checkpoint_id=str(checkpoint / "dcp_checkpoint"))
        updated = state["fsdp_checkpoint"]["model"][parameter]
        assert torch.isfinite(updated).all()
        delta = (updated - original).abs()
        assert delta.max() > 0, "PPO action decoder did not change"
        report["policy_weight_check"] = {
            "parameter": parameter,
            "reference": "SFT weights cast to BF16 as in the model loader",
            "changed_elements": int(torch.count_nonzero(delta)),
            "max_absolute_change_from_sft": float(delta.max()),
        }
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(str(run_dir / "tensorboard")) as writer:
            for tag, value in {
                "resource/gpu_peak_mib": peak,
                "resource/gpu_capacity_mib": report["gpu_capacity_mib"],
                "resource/wall_time_seconds": wall_time,
            }.items():
                writer.add_scalar(tag, value, step - 1)
                report["metrics"][tag] = value
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: checkpoint step {step}, peak {peak} MiB, {wall_time:.1f} s")


if __name__ == "__main__":
    main()
