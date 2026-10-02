"""Integration contracts against real, pinned stack-cube assets (no model weights)."""

import av
import numpy as np
import pandas as pd
import pytest
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import (
    ShardedSingleStepDataset,
    extract_step_data,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from scipy.spatial.transform import Rotation
from torchcodec.decoders import VideoDecoder

from embodied_template.stack_cube import (
    DATASET,
    KEYS,
    PREPARED,
    ROOT,
    SPLIT_PATH,
    canonical_axis_angle,
    convert_dataset_state,
    create_processor,
    episode_path,
    make_split,
    read_json,
    statistics,
    video_path,
    write_json,
)

CONFIG = MODALITY_CONFIGS["libero_sim"]
TAG = EmbodimentTag.LIBERO_PANDA
REPORT_DIR = ROOT / "runs/stack-cube/contract"


@pytest.fixture(scope="session")
def processor():
    value = create_processor()
    value.eval()
    return value


def test_episode_split_schema_and_train_only_statistics():
    split = read_json(SPLIT_PATH)
    generated = make_split(range(147))
    for key in ["train", "validation", "test"]:
        assert split[key] == generated[key]
    assert [len(split[k]) for k in ["train", "validation", "test"]] == [117, 15, 15]
    assert len(set(split["train"] + split["validation"] + split["test"])) == 147
    info = read_json(DATASET / "meta/info.json")
    assert (info["total_frames"], info["total_episodes"], info["fps"]) == (
        53265,
        147,
        20,
    )
    modality = read_json(DATASET / "meta/modality.json")
    for kind, end in [("state", 8), ("action", 7)]:
        assert list(modality[kind]) == KEYS
        assert [(v["start"], v["end"]) for v in modality[kind].values()] == [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, end),
        ]
    assert [
        modality["video"][k]["original_key"] for k in CONFIG["video"].modality_keys
    ] == ["observation.images.front", "observation.images.wrist"]
    train_states, train_actions = [], []
    total = 0
    max_errors = []
    for name in ["train", "validation", "test"]:
        path = PREPARED / name
        manifest = read_json(path / "MANIFEST.json")
        assert manifest["statistics_episode_ids"] == split["train"]
        assert read_json(path / "meta/stats.json") == read_json(
            PREPARED / "train/meta/stats.json"
        )
        for episode in split[name]:
            source = pd.read_parquet(episode_path(DATASET, episode))
            df = pd.read_parquet(episode_path(path, episode))
            n = len(df)
            total += n
            np.testing.assert_array_equal(df.frame_index, np.arange(n))
            np.testing.assert_allclose(df.timestamp, np.arange(n) / 20, atol=1e-6)
            assert (df.episode_index == episode).all()
            assert (df.task_index == 0).all()
            state, action = np.stack(df["observation.state"]), np.stack(df.action)
            assert state.shape == (n, 8) and action.shape == (n, 7)
            assert state.dtype == action.dtype == np.float32
            assert np.isfinite(state).all() and np.isfinite(action).all()
            np.testing.assert_allclose(action, np.stack(source.action), atol=1e-7)
            np.testing.assert_allclose(
                state,
                convert_dataset_state(np.stack(source["observation.state"])),
                atol=1e-7,
            )
            assert set(np.unique(action[:, -1])) == {-1.0, 1.0}
            assert np.max(np.abs(action[:, :6])) <= 1
            assert state[:, 6].min() >= -1e-4 and state[:, 6].max() <= 0.041
            assert state[:, 7].max() <= 1e-4 and state[:, 7].min() >= -0.041
            assert np.linalg.norm(state[:, 3:6], axis=1).max() <= np.pi + 1e-6
            raw_rotation = np.stack(source["observation.state"])[:, 3:6]
            euler = Rotation.from_euler("xyz", raw_rotation)
            rotation = Rotation.from_rotvec(state[:, 3:6])
            error = (rotation * euler.inv()).magnitude()
            max_errors.append(float(error.max()))
            assert error.max() < 1e-6
            # Euler interpretation is smooth across all captured trajectories.
            assert (euler[1:] * euler[:-1].inv()).magnitude().max() < 0.1
            if name == "train":
                train_states.append(state)
                train_actions.append(action)
    assert total == 53265
    stats = read_json(PREPARED / "train/meta/stats.json")
    assert stats == {
        "observation.state": statistics(train_states),
        "action": statistics(train_actions),
    }
    write_json(
        REPORT_DIR / "schema.json",
        {
            "episodes": 147,
            "frames": total,
            "split_counts": [117, 15, 15],
            "statistics_frames": sum(map(len, train_actions)),
            "state_rotation_matrix_max_error_rad": max(max_errors),
        },
    )


def test_raw_rotation_is_not_axis_angle():
    df = pd.read_parquet(episode_path(DATASET, 0))
    values = np.stack(df["observation.state"])[:, 3:6]
    wrong = Rotation.from_rotvec(values)
    right = Rotation.from_euler("xyz", values)
    assert (wrong[1:] * wrong[:-1].inv()).magnitude().max() > 2
    assert (right[1:] * right[:-1].inv()).magnitude().max() < 0.01
    assert wrong.apply([0, 0, 1])[0, 2] > 0
    assert right.apply([0, 0, 1])[0, 2] < -0.99


@pytest.mark.parametrize("episode", [0, 7, 146])
def test_random_mpeg4_rgb_decode_and_timestamps(episode):
    n = len(pd.read_parquet(episode_path(DATASET, episode)))
    indices = [n - 1, 0, n // 2, 1, n - 16, n // 2, 0]
    for key in ["image", "wrist_image"]:
        path = video_path(DATASET, episode, key)
        decoder = VideoDecoder(str(path), dimension_order="NHWC", num_ffmpeg_threads=1)
        assert len(decoder) == n
        assert decoder.metadata.average_fps == 20
        batch = decoder.get_frames_at(indices=indices)
        data = batch.data.numpy()
        assert data.shape == (len(indices), 256, 256, 3) and data.dtype == np.uint8
        np.testing.assert_allclose(
            batch.pts_seconds.numpy(), np.array(indices) / 20, atol=1e-6
        )
        np.testing.assert_array_equal(data[0], decoder[n - 1].numpy())
        np.testing.assert_array_equal(data[2], data[5])
        np.testing.assert_array_equal(data[1], data[6])
        assert not np.array_equal(data[0], data[1])
        # Independent bundled PyAV decoder checks RGB ordering and exact frame indices.
        with av.open(str(path)) as container:
            assert container.streams.video[0].codec_context.name == "mpeg4"
            reference = {
                i: f.to_ndarray(format="rgb24")
                for i, f in enumerate(container.decode(video=0))
                if i in indices
            }
        for i, frame in zip(indices, data):
            np.testing.assert_allclose(frame, reference[i], atol=2)


@pytest.mark.parametrize("episode", [0, 7])
def test_official_loader_boundaries_gripper_and_processor(processor, episode):
    split = read_json(SPLIT_PATH)
    name = next(k for k in ["train", "validation", "test"] if episode in split[k])
    loader = LeRobotEpisodeLoader(PREPARED / name, CONFIG)
    index = split[name].index(episode)
    df = loader[index]
    assert loader._video_key_mapping == {}
    raw = pd.read_parquet(episode_path(PREPARED / name, episode))
    action = np.stack(raw.action)
    changes = np.flatnonzero(np.diff(action[:, -1]) != 0) + 1
    assert len(changes) >= 2
    samples = sorted(
        set(
            [0, len(df) // 2, len(df) - 16, len(df) - 15, len(df) - 1]
            + [int(i) for j in changes for i in [j - 1, j]]
        )
    )
    errors = []
    for frame in samples:
        step = extract_step_data(df, frame, CONFIG, TAG, allow_padding=True)
        expected = action[np.minimum(frame + np.arange(16), len(df) - 1)]
        np.testing.assert_array_equal(
            np.concatenate([step.actions[k] for k in KEYS], axis=-1), expected
        )
        np.testing.assert_array_equal(
            np.concatenate([step.states[k] for k in KEYS], axis=-1),
            np.stack(raw["observation.state"])[frame : frame + 1],
        )
        for key in ["image", "wrist_image"]:
            np.testing.assert_array_equal(
                step.images[key][0],
                loader._load_video_data(episode, np.array([frame]))[key][0],
            )
        encoded = processor([{"type": "episode_step", "content": step}])
        assert encoded["state"].shape == (1, 132)
        assert encoded["action"].shape == (40, 132)
        offset = 0
        normalized_state = {}
        for key in KEYS:
            width = step.states[key].shape[-1]
            normalized_state[key] = encoded["state"].numpy()[:, offset : offset + width]
            offset += width
        restored_state = processor.state_action_processor.unapply_state(
            normalized_state, "libero_sim"
        )
        np.testing.assert_allclose(
            np.concatenate([restored_state[key] for key in KEYS], axis=-1),
            np.concatenate([step.states[key] for key in KEYS], axis=-1),
            atol=1e-6,
        )
        mask = encoded["action_mask"].numpy()
        assert mask[:16, :7].sum() == 112 and mask.sum() == 112
        assert np.count_nonzero(encoded["state"].numpy()[:, 8:]) == 0
        assert np.count_nonzero(encoded["action"].numpy()[16:]) == 0
        decoded = processor.decode_action(encoded["action"].numpy(), TAG)
        reconstructed = np.concatenate([decoded[k] for k in KEYS], axis=-1)
        # Episodes 0 and 7 are frozen train episodes, so min/max clipping is lossless.
        assert name == "train"
        error = np.max(np.abs(reconstructed - expected))
        errors.append(float(error))
        np.testing.assert_allclose(reconstructed, expected, atol=1e-6)
        batch = processor.collator([encoded])["inputs"]
        assert batch["state"].shape == (1, 1, 132)
        assert batch["action"].shape == (1, 40, 132)
        assert batch["embodiment_id"].item() == 2
        assert batch["image_grid_thw"].shape[0] == 2
    with pytest.raises(IndexError):
        extract_step_data(df, len(df) - 1, CONFIG, TAG, allow_padding=False)
    training = ShardedSingleStepDataset(
        PREPARED / name, TAG, CONFIG, allow_padding=True
    )
    assert training.get_effective_episode_length(index) == len(df) - 15
    write_json(
        REPORT_DIR / f"processor-episode-{episode}.json",
        {
            "episode": episode,
            "frames": samples,
            "gripper_switch_frames": changes.tolist(),
            "round_trip_max_abs_error": max(errors),
            "action_horizon": 16,
            "model_horizon": 40,
            "state_dim": 8,
            "decoded_action_dim": 7,
        },
    )


def test_processor_serialization_offline(processor):
    destination = REPORT_DIR / "processor"
    processor.save_pretrained(destination)
    for name in ["statistics.json", "embodiment_id.json"]:
        assert "libero_sim" in read_json(destination / name)
    assert (
        "libero_sim"
        in read_json(destination / "processor_config.json")["processor_kwargs"][
            "modality_configs"
        ]
    )
    restored = Gr00tN1d7Processor.from_pretrained(
        destination, transformers_loading_kwargs={"local_files_only": True}
    )
    assert restored.modality_configs["libero_sim"]["action"].delta_indices == list(
        range(16)
    )
    assert restored.embodiment_id_mapping["libero_sim"] == 2
    assert restored.use_percentiles is False
    assert restored.use_relative_action is False
    assert str(ROOT / "models/cosmos-reason2-2b") == restored.model_name


def test_principal_axis_angle_branch():
    values = np.array([[3.16, 0.02, 0], [-3.16, -0.02, 0], [0, 0, 0]])
    converted = canonical_axis_angle(values)
    assert np.linalg.norm(converted, axis=-1).max() <= np.pi
    np.testing.assert_allclose(
        Rotation.from_rotvec(converted).as_matrix(),
        Rotation.from_rotvec(values).as_matrix(),
        atol=1e-6,
    )


def test_all_train_actions_round_trip_without_percentile_clipping(processor):
    actions = np.concatenate(
        [
            np.stack(pd.read_parquet(episode_path(PREPARED / "train", episode)).action)
            for episode in read_json(SPLIT_PATH)["train"]
        ]
    )
    action_dict = {key: actions[:, i : i + 1] for i, key in enumerate(KEYS)}
    normalizer = processor.state_action_processor
    encoded = normalizer.apply_action(action_dict, "libero_sim")
    decoded = normalizer.unapply_action(encoded, "libero_sim")
    restored = np.concatenate([decoded[key] for key in KEYS], axis=-1)
    np.testing.assert_allclose(restored, actions, atol=1e-6)
    # Percentile clipping loses real commands; record the configuration decision.
    stats = read_json(PREPARED / "train/meta/stats.json")["action"]
    clipped = np.clip(actions, stats["q01"], stats["q99"])
    affected = int(np.any(np.abs(clipped - actions) > 1e-6, axis=1).sum())
    assert affected > 0
    write_json(
        REPORT_DIR / "normalization.json",
        {
            "train_frames": len(actions),
            "round_trip_max_abs_error": float(np.abs(restored - actions).max()),
            "percentile_clipping_affected_frames": affected,
            "percentile_clipping_max_abs_error": float(np.abs(clipped - actions).max()),
            "use_percentiles": False,
            "clip_outliers": True,
        },
    )
