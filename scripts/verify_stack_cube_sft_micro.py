"""Load the SFT bundle in a new network namespace and evaluate one held-out episode."""

import os
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from safetensors import safe_open
from train_stack_cube_sft_micro import file_digest
from transformers import set_seed

from embodied_template.stack_cube import (
    KEYS,
    PREPARED,
    ROOT,
    SPLIT_PATH,
    read_json,
    write_json,
)


def main():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.readlink("/proc/self/ns/net") != os.environ["SFT_HOST_NETWORK_NAMESPACE"]
    assert socket.if_nameindex() == [(1, "lo")]
    bundle = ROOT / "models/stack-cube-n1.7-sft"
    checkpoint = bundle / "checkpoint"
    manifest = read_json(bundle / "MANIFEST.json")
    for name, digest in manifest["files"].items():
        assert file_digest(bundle / name) == digest, name
    state = read_json(checkpoint / "trainer_state.json")
    assert state["global_step"] == 1
    logs = [x for x in state["log_history"] if "loss" in x]
    assert len(logs) == 1 and np.isfinite(logs[0]["loss"])
    assert np.isfinite(logs[0]["grad_norm"]) and logs[0]["grad_norm"] > 0

    base = ROOT / "models/gr00t-n1.7-3b"
    index = read_json(base / "model.safetensors.index.json")["weight_map"]
    name = "action_head.action_decoder.layer2.W"
    with safe_open(base / index[name], framework="pt") as source:
        before = source.get_tensor(name)
    trained_index = checkpoint / "model.safetensors.index.json"
    trained_file = (
        read_json(trained_index)["weight_map"][name]
        if trained_index.exists()
        else "model.safetensors"
    )
    with safe_open(checkpoint / trained_file, framework="pt") as trained:
        after = trained.get_tensor(name)
    delta = (after.float() - before.float()).abs()
    changed = int(torch.count_nonzero(delta))
    assert changed > 0
    weight_change = {
        "tensor": name,
        "changed_elements": changed,
        "max_abs_delta": delta.max().item(),
    }
    del before, after, delta

    started = time.monotonic()
    set_seed(42)
    policy = Gr00tPolicy("LIBERO_PANDA", str(checkpoint), device="cuda")
    processor = policy.processor
    assert not processor.use_percentiles and not processor.use_relative_action
    assert processor.state_dropout_prob == 0.0
    assert processor.model_name == manifest["backbone_model_path"]
    assert "libero_sim" in processor.embodiment_id_mapping
    expected_stats = read_json(
        ROOT / "runs/stack-cube/contract/processor/statistics.json"
    )["libero_sim"]
    actual_stats = read_json(checkpoint / "statistics.json")["libero_sim"]
    for modality in ["state", "action"]:
        for key in KEYS:
            for statistic, expected in expected_stats[modality][key].items():
                np.testing.assert_allclose(
                    actual_stats[modality][key][statistic], expected, atol=1e-7
                )
    split = read_json(SPLIT_PATH)
    episode = split["validation"][0]
    assert episode not in split["train"] and episode not in split["test"]
    loader = LeRobotEpisodeLoader(PREPARED / "validation", policy.modality_configs)
    trajectory = loader[0]
    horizon = len(policy.modality_configs["action"].delta_indices)
    assert horizon == 16
    obs_configs = {k: v for k, v in policy.modality_configs.items() if k != "action"}
    predictions, latency = [], []
    for frame in range(0, len(trajectory), horizon):
        step = extract_step_data(
            trajectory, frame, obs_configs, EmbodimentTag.LIBERO_PANDA
        )
        assert np.concatenate(list(step.states.values()), axis=-1).shape == (1, 8)
        observation = {
            "video": {k: np.asarray(v)[None] for k, v in step.images.items()},
            "state": {k: v[None] for k, v in step.states.items()},
            "language": {policy.language_key: [[step.text]]},
        }
        begin = time.monotonic()
        action, _ = policy.get_action(observation)
        torch.cuda.synchronize()
        latency.append(time.monotonic() - begin)
        values = np.concatenate([action[k][0] for k in KEYS], axis=-1)
        assert values.shape == (16, 7) and np.isfinite(values).all()
        predictions.extend(values)
        print(
            f"Episode {episode}, frame {frame}: finite {values.shape} action chunk",
            flush=True,
        )
    prediction = np.asarray(predictions)[: len(trajectory)]
    expert = np.concatenate(
        [np.stack(trajectory[f"action.{k}"]) for k in KEYS], axis=-1
    )
    assert prediction.shape == expert.shape
    mse = np.mean((prediction - expert) ** 2, axis=0)
    mae = np.mean(np.abs(prediction - expert), axis=0)
    assert np.isfinite(mse).all() and np.isfinite(mae).all()
    report = ROOT / "runs/stack-cube/sft-micro"
    output = ROOT / "runs/stack-cube/visualization/open-loop"
    output.mkdir(parents=True, exist_ok=True)
    plot = output / f"micro-episode-{episode:06d}.png"
    fig, axes = plt.subplots(7, 1, figsize=(12, 15), sharex=True)
    labels = [
        "delta x / 0.5",
        "delta y / 0.5",
        "delta z / 0.5",
        "rotvec x / 0.5",
        "rotvec y / 0.5",
        "rotvec z / 0.5",
        "gripper sign",
    ]
    for i, (axis, key) in enumerate(zip(axes, labels)):
        axis.plot(expert[:, i], label="Expert", linewidth=1)
        axis.plot(prediction[:, i], label="Micro SFT", linewidth=1, alpha=0.8)
        axis.set_ylabel(key)
        axis.set_title(f"MSE={mse[i]:.6f}, MAE={mae[i]:.6f}", fontsize=9)
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("Expert trajectory frame (20 Hz)")
    fig.suptitle(
        f"Held-out episode {episode}: one projector update, open-loop predictions"
    )
    fig.tight_layout()
    fig.savefig(plot, dpi=120)
    plt.close(fig)
    np.savez(
        output / f"micro-episode-{episode:06d}.npz",
        expert=expert,
        prediction=prediction,
    )
    write_json(
        report / "offline-verification.json",
        {
            "network_isolation": "unshare --user --map-root-user --net; loopback only",
            "hf_hub_offline": True,
            "global_step": state["global_step"],
            "training_log": logs[0],
            "weight_change": weight_change,
            "episode_id": episode,
            "split": "validation",
            "frames": len(trajectory),
            "action_shape": list(prediction.shape),
            "action_horizon": horizon,
            "denoising_steps": policy.model.action_head.num_inference_timesteps,
            "action_keys": KEYS,
            "mse_per_dimension": mse.tolist(),
            "mae_per_dimension": mae.tolist(),
            "mse": float(mse.mean()),
            "mae": float(mae.mean()),
            "inference_latency_seconds": latency,
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "plot": str(plot.relative_to(ROOT)),
            "scope": "Engineering gate only; open-loop does not measure closed-loop task success",
        },
    )
    print(f"Offline verification passed; plot: {plot}")


if __name__ == "__main__":
    main()
