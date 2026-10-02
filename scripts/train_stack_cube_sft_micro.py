"""One official N1.7 SFT update, preserving the stack-cube processor contract."""

import hashlib
import shutil
import subprocess
import time
from pathlib import Path

import torch
from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

from embodied_template.stack_cube import (
    CONTRACT_PATH,
    PREPARED,
    ROOT,
    SPLIT_PATH,
    create_processor,
    read_json,
    sha256,
    write_json,
)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    report = ROOT / "runs/stack-cube/sft-micro"
    report.mkdir(parents=True, exist_ok=True)
    # Stage config/processor locally; immutable base weights stay on the NAS.
    initial = ROOT / "tmp/stack-cube-sft-micro/initial"
    if initial.parent.exists():
        shutil.rmtree(initial.parent)
    initial.mkdir(parents=True, exist_ok=True)
    base = ROOT / "models/gr00t-n1.7-3b"
    lock = read_json(ROOT / "configs/dependencies.lock.json")
    for source in lock["sources"].values():
        actual = subprocess.check_output(
            ["git", "-C", str(ROOT / source["path"]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        assert actual == source["commit"]
    for name in ["groot_n1_7_3b", "cosmos_reason2_2b"]:
        asset = lock["assets"][name]
        directory = ROOT / asset["path"]
        manifest = read_json(directory / ".asset-manifest.json")
        assert manifest["revision"] == asset["revision"]
        for filename, digest in manifest["files"].items():
            assert file_digest(directory / filename) == digest, filename
    for path in base.glob("model*"):
        (initial / path.name).symlink_to(path.resolve())
    model_config = read_json(base / "config.json")
    # Upstream chooses the backbone class using this canonical name substring.
    backbone = ROOT / "models/nvidia/Cosmos-Reason2-2B"
    backbone.parent.mkdir(parents=True, exist_ok=True)
    if not backbone.is_symlink():
        backbone.symlink_to("../cosmos-reason2-2b")
    model_config["model_name"] = str(backbone)
    model_config.update(read_json(CONTRACT_PATH)["processor"])
    write_json(initial / "config.json", model_config)
    create_processor().save_pretrained(initial)

    # Official DatasetFactory writes stats; isolate these metadata writes.
    dataset = initial.parent / "train"
    dataset.mkdir(exist_ok=True)
    shutil.copytree(PREPARED / "train/meta", dataset / "meta")
    for name in ["data", "videos"]:
        (dataset / name).symlink_to((PREPARED / "train" / name).resolve())

    config_path = ROOT / "configs/stack_cube_sft_micro.json"
    config = get_default_config().load_dict(read_json(config_path))
    config.model.model_name = model_config["model_name"]
    config.data.datasets = []
    from gr00t.configs.data.data_config import SingleDatasetConfig

    config.data.datasets.append(
        SingleDatasetConfig(dataset_paths=[str(dataset)], embodiment_tag="libero_sim")
    )
    config.training.start_from_checkpoint = str(initial)
    config.training.output_dir = str(report / "training")
    # Verify exactly the overrides consumed by the official pipeline.
    processor = Gr00tN1d7Processor.from_pretrained(
        initial,
        use_relative_action=config.model.use_relative_action,
        model_name=config.model.model_name,
        transformers_loading_kwargs={"local_files_only": True},
    )
    assert not processor.use_percentiles and not processor.use_relative_action
    assert processor.state_dropout_prob == 0.0
    del processor
    started = time.monotonic()
    run(config)
    torch.cuda.synchronize()
    write_json(
        report / "training-resources.json",
        {
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu": torch.cuda.get_device_name(),
        },
    )

    bundle = ROOT / "models/stack-cube-n1.7-sft"
    bundle.mkdir(parents=True, exist_ok=True)
    checkpoint = report / "training/checkpoint-1"
    shutil.copytree(checkpoint, bundle / "checkpoint")
    shutil.copytree(report / "training/processor", bundle / "processor")
    shutil.copytree(report / "training/experiment_cfg", bundle / "experiment_cfg")
    shutil.copy2(base / "LICENSE", bundle / "LICENSE")
    for path in [config_path, CONTRACT_PATH, SPLIT_PATH]:
        shutil.copy2(path, bundle / "experiment_cfg" / path.name)
    for name in ["environment.json", "requirements.lock.txt"]:
        shutil.copy2(
            ROOT / "runs/stack-cube/contract" / name,
            bundle / "experiment_cfg" / name,
        )
    write_json(
        bundle / "MANIFEST.json",
        {
            "stage": "micro SFT: one projector-only update; not a learned stack-cube policy",
            "official_training_function": "gr00t.experiment.experiment.run (launch_finetune.py backend)",
            "source_commits": {k: v["commit"] for k, v in lock["sources"].items()},
            "assets": lock["assets"],
            "asset_manifest_sha256": {
                k: sha256(ROOT / v["path"] / ".asset-manifest.json")
                for k, v in lock["assets"].items()
            },
            "data_manifest_sha256": sha256(PREPARED / "train/MANIFEST.json"),
            "split_sha256": sha256(SPLIT_PATH),
            "contract_sha256": sha256(CONTRACT_PATH),
            "backbone_model_path": config.model.model_name,
            "environment_sha256": sha256(
                ROOT / "runs/stack-cube/contract/environment.json"
            ),
            "entrypoint_sha256": sha256(Path(__file__)),
            "files": {
                str(p.relative_to(bundle)): file_digest(p)
                for p in sorted(bundle.rglob("*"))
                if p.is_file()
            },
        },
    )
    print(f"Bundle saved at {bundle}")


if __name__ == "__main__":
    main()
