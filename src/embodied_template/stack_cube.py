"""The pinned stack-cube dataset and its explicit SFT representation."""

import hashlib
import json
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/isaaclab-stack-cube"
PREPARED = ROOT / "runs/stack-cube/data"
SPLIT_PATH = ROOT / "experiments/stack_cube/sft/episode_split.json"
CONTRACT_PATH = ROOT / "experiments/stack_cube/sft/data_contract.json"
KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_path(root, episode):
    info = read_json(Path(root) / "meta/info.json")
    return Path(root) / info["data_path"].format(
        episode_chunk=episode // info["chunks_size"], episode_index=episode
    )


def video_path(root, episode, key):
    info = read_json(Path(root) / "meta/info.json")
    modality = read_json(Path(root) / "meta/modality.json")
    return Path(root) / info["video_path"].format(
        episode_chunk=episode // info["chunks_size"],
        episode_index=episode,
        video_key=modality["video"][key]["original_key"],
    )


def canonical_axis_angle(values):
    """Select the principal rotation vector (angle <= pi), in radians."""
    return Rotation.from_rotvec(values).as_rotvec().astype(np.float32)


def convert_dataset_state(state):
    """Convert raw Euler xyz to principal axis-angle; keep xyz and signed fingers."""
    result = np.asarray(state, dtype=np.float32).copy()
    result[:, 3:6] = Rotation.from_euler("xyz", result[:, 3:6]).as_rotvec()
    return result


def make_split(episode_ids, seed=42):
    order = (
        np.random.Generator(np.random.PCG64(seed))
        .permutation(sorted(episode_ids))
        .tolist()
    )
    return {
        "seed": seed,
        "algorithm": "numpy.random.Generator(PCG64).permutation(sorted episode IDs)",
        "train": sorted(order[:117]),
        "validation": sorted(order[117:132]),
        "test": sorted(order[132:]),
    }


def statistics(arrays):
    values = np.concatenate(arrays).astype(np.float32)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def create_processor():
    """Use the official base processor settings with measured stack-cube statistics."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    contract = read_json(CONTRACT_PATH)
    model = ROOT / "models/gr00t-n1.7-3b"
    lock = read_json(ROOT / "configs/dependencies.lock.json")
    for asset_name in ["groot_n1_7_3b", "cosmos_reason2_2b"]:
        asset = lock["assets"][asset_name]
        directory = ROOT / asset["path"]
        manifest = read_json(directory / ".asset-manifest.json")
        assert manifest["revision"] == asset["revision"]
        for filename, digest in manifest["files"].items():
            if Path(filename).suffix in [".json", ".txt"]:
                assert sha256(directory / filename) == digest, filename
    kwargs = read_json(model / "processor_config.json")["processor_kwargs"]
    kwargs["modality_configs"]["libero_sim"] = MODALITY_CONFIGS["libero_sim"]
    kwargs.update(contract["processor"])
    kwargs["model_name"] = str(ROOT / "models/cosmos-reason2-2b")
    kwargs["embodiment_id_mapping"] = read_json(model / "embodiment_id.json")
    kwargs["statistics"] = read_json(model / "statistics.json")
    loader = LeRobotEpisodeLoader(PREPARED / "train", MODALITY_CONFIGS["libero_sim"])
    kwargs["statistics"]["libero_sim"] = loader.get_dataset_statistics()
    return Gr00tN1d7Processor(
        **kwargs,
        transformers_loading_kwargs={"local_files_only": True},
    )


def prepare_dataset(destination=PREPARED):
    """Write small transformed parquets; link immutable source videos by episode ID."""
    lock = read_json(ROOT / "configs/dependencies.lock.json")
    asset = lock["assets"]["stack_cube_dataset"]
    source_manifest = read_json(DATASET / ".asset-manifest.json")
    assert source_manifest["revision"] == asset["revision"]
    for name, expected in source_manifest["files"].items():
        assert sha256(DATASET / name) == expected, name

    episodes = [
        json.loads(line)
        for line in (DATASET / "meta/episodes.jsonl").read_text().splitlines()
    ]
    split = make_split([ep["episode_index"] for ep in episodes])
    split.update(repo_id=asset["repo_id"], revision=asset["revision"])
    write_json(SPLIT_PATH, split)
    train_states, train_actions = [], []
    converted = {}
    codecs = set()
    for ep in episodes:
        episode = ep["episode_index"]
        df = pd.read_parquet(episode_path(DATASET, episode))
        state = convert_dataset_state(np.stack(df["observation.state"]))
        df["observation.state"] = list(state)
        df["action"] = list(np.stack(df["action"]).astype(np.float32))
        converted[episode] = df
        for key in ["image", "wrist_image"]:
            with av.open(str(video_path(DATASET, episode, key))) as video:
                stream = video.streams.video[0]
                assert stream.frames == len(df)
                assert (stream.width, stream.height) == (256, 256)
                assert float(stream.average_rate) == 20
                codecs.add(stream.codec_context.name)
        if episode in split["train"]:
            train_states.append(state)
            train_actions.append(np.stack(df["action"]))
    assert codecs == {"mpeg4"}
    stats = {
        "observation.state": statistics(train_states),
        "action": statistics(train_actions),
    }
    for name in ["train", "validation", "test"]:
        root = Path(destination) / name
        (root / "meta").mkdir(parents=True, exist_ok=True)
        selected = [ep for ep in episodes if ep["episode_index"] in split[name]]
        for filename in ["modality.json", "tasks.jsonl"]:
            (root / "meta" / filename).write_bytes(
                (DATASET / "meta" / filename).read_bytes()
            )
        (root / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(ep) + "\n" for ep in selected)
        )
        info = read_json(DATASET / "meta/info.json")
        for key in ["observation.images.front", "observation.images.wrist"]:
            info["features"][key]["info"]["video.codec"] = "mpeg4"
        info.update(
            total_episodes=len(selected),
            total_frames=sum(ep["length"] for ep in selected),
            total_videos=2 * len(selected),
        )
        write_json(root / "meta/info.json", info)
        write_json(root / "meta/stats.json", stats)
        for ep in selected:
            episode = ep["episode_index"]
            path = episode_path(root, episode)
            path.parent.mkdir(parents=True, exist_ok=True)
            converted[episode].to_parquet(path, index=False)
            for key in ["image", "wrist_image"]:
                link = video_path(root, episode, key)
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(
                    os.path.relpath(
                        video_path(DATASET, episode, key).resolve(),
                        link.parent.resolve(),
                    )
                )
        files = {
            str(p.relative_to(root)): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and not p.is_symlink() and p.name != "MANIFEST.json"
        }
        write_json(
            root / "MANIFEST.json",
            {
                "source": asset,
                "split": name,
                "episode_ids": split[name],
                "statistics_episode_ids": split["train"],
                "statistics_scope": "train only, shared across all splits",
                "contract_sha256": sha256(CONTRACT_PATH),
                "converter_source_sha256": sha256(Path(__file__)),
                "dependency_lock_sha256": sha256(
                    ROOT / "configs/dependencies.lock.json"
                ),
                "source_manifest_sha256": sha256(DATASET / ".asset-manifest.json"),
                "split_sha256": sha256(SPLIT_PATH),
                "files": files,
            },
        )
    print(f"Prepared {len(episodes)} episodes at {destination}")
