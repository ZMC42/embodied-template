#!/usr/bin/env python3
"""Verify recorded RLinf stack-cube rollout, video, and resource measurements."""

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def verify_ray(run):
    """Validate native RLinf evaluation metrics and its front-camera recording."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(run / "tensorboard"))
    events.Reload()
    metrics = {
        tag: [event.value for event in events.Scalars(tag)]
        for tag in events.Tags()["scalars"]
    }
    assert metrics and all(np.isfinite(values).all() for values in metrics.values())
    assert metrics["eval/episode_len"] == [32.0]
    assert metrics["eval/num_trajectories"] == [1.0]
    video_path = run / "video/eval/seed_0/0.mp4"
    with av.open(str(video_path)) as video:
        frames = list(video.decode(video=0))
        assert len(frames) >= 33 and (frames[0].width, frames[0].height) == (256, 256)
        assert video.streams.video[0].average_rate == 20
    with (run / "gpu.csv").open() as file:
        rows = list(csv.DictReader(file))
    peak = max(float(row[" memory.used [MiB]"].split()[0]) for row in rows)
    started, ended = [
        datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").astimezone()
        for row in (rows[0], rows[-1])
    ]
    lock = json.loads((ROOT / "configs/dependencies.lock.json").read_text())
    return {
        "status": "passed",
        "stage": "RLinf Ray evaluation; no PPO update",
        "source_commits": {
            key: value["commit"] for key, value in lock["sources"].items()
        },
        "metrics": metrics,
        "peak_gpu_mib": peak,
        "sampled_wall_time_s": (ended - started).total_seconds(),
        "video": str(video_path),
        "video_frames": len(frames),
        "video_resolution": [256, 256],
        "video_fps": 20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--ray", action="store_true")
    args = parser.parse_args()
    if args.ray:
        summary = verify_ray(args.run)
        (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    report = json.loads((args.run / "rollout.json").read_text())
    assert report["status"] == "passed"
    steps = report["steps"]
    assert len(steps) >= 16
    actions = np.asarray([step["action"] for step in steps])
    assert actions.shape == (len(steps), 7) and np.isfinite(actions).all()
    assert np.isin(actions[:, -1], [-1, 0, 1]).all()
    assert np.isfinite([step["reward"] for step in steps]).all()
    times = np.asarray([step["simulation_time_s"] for step in steps])
    states = np.asarray([step["eef_state"] for step in steps])
    assert np.all(np.diff(times) > 0), "Physics time must advance"
    assert states.shape == (len(steps), 8) and np.isfinite(states).all()
    assert np.ptp(states[:, :3], axis=0).max() > 1e-4, "EEF did not move"
    with av.open(str(args.run / f"seed-{report['seed']}.mp4")) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == len(steps) + 1
        assert (frames[0].width, frames[0].height) == (512, 256)
        assert video.streams.video[0].average_rate == 20
    with (args.run / "gpu.csv").open() as file:
        rows = list(csv.DictReader(file))
    peak = max(float(row[" memory.used [MiB]"].split()[0]) for row in rows)
    sources = {}
    lock = json.loads((ROOT / "configs/dependencies.lock.json").read_text())
    for key, source in lock["sources"].items():
        sources[key] = subprocess.check_output(
            ["git", "-C", str(ROOT / source["path"]), "rev-parse", "HEAD"], text=True
        ).strip()
    summary = {
        "status": "passed",
        "checkpoint": report["checkpoint"],
        "seed": report["seed"],
        "steps": len(steps),
        "success_once": report["success_once"],
        "reward_sum": sum(step["reward"] for step in steps),
        "eef_xyz_range_m": np.ptp(states[:, :3], axis=0).tolist(),
        "left_finger_range_m": float(np.ptp(states[:, 6])),
        "simulation_time_s": times[-1],
        "video_frames": len(frames),
        "video_resolution": [512, 256],
        "peak_gpu_mib": peak,
        "wall_time_s": report["wall_time_s"],
        "source_commits": sources,
        "bundle_manifest_sha256": hashlib.sha256(
            (ROOT / "models/stack-cube-n1.7-sft/MANIFEST.json").read_bytes()
        ).hexdigest(),
    }
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
