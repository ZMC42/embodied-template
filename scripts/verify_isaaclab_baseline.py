#!/usr/bin/env python3
"""Check a completed baseline run and record metrics, resources and versions."""

import argparse
import csv
import importlib.metadata
import json
import math
import pickle
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import av
import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--baseline", choices=("isaaclab_n1_5", "n1_7_libero"), default="isaaclab_n1_5"
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
    else:
        del sources["isaaclab"]
        sources["rlinf"] = lock["baselines"][args.baseline]["rlinf_source_path"]
        packages += ["rlinf-libero", "robosuite", "mujoco", "av"]
        assert cfg.actor.model.embodiment_tag == "libero_sim"
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
        "model_revision": lock["baselines"][args.baseline]["model_revision"],
    }
    if args.baseline == "n1_7_libero":
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
        report["libero_assets_revision"] = lock["baselines"][args.baseline][
            "libero_assets"
        ]["revision"]
        report["checkpoint_size_bytes"] = sum(
            path.stat().st_size for path in checkpoint_files
        )
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: checkpoint step {step}, peak {peak} MiB, {wall_time:.1f} s")


if __name__ == "__main__":
    main()
