#!/usr/bin/env python3
"""Install the pinned N1.5 baseline without changing the N1.7 checkout."""

import json
import subprocess
from pathlib import Path

from download_assets import sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    lock = json.loads((ROOT / "configs/dependencies.lock.json").read_text())
    baseline = lock["baselines"]["isaaclab_n1_5"]
    assert (
        sha256(ROOT / baseline["requirements_lock"])
        == baseline["requirements_lock_sha256"]
    )
    for source in lock["sources"].values():
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT / source["path"]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        assert commit == source["commit"], source["path"]

    source_path = ROOT / baseline["gr00t_source_path"]
    if not source_path.exists():
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT / lock["sources"]["isaac_groot"]["path"]),
                "-c",
                "filter.lfs.required=false",
                "-c",
                "filter.lfs.smudge=",
                "-c",
                "filter.lfs.process=",
                "worktree",
                "add",
                "--detach",
                str(source_path),
                baseline["gr00t_source_commit"],
            ],
            check=True,
        )
    assert (
        subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True
        ).strip()
        == baseline["gr00t_source_commit"]
    )

    venv = ROOT / ".venvs/isaaclab-n1.5"
    subprocess.run(
        ["uv", "venv", "--allow-existing", str(venv), "--python", baseline["python"]],
        check=True,
    )
    install = [
        "uv",
        "pip",
        "install",
        "--python",
        str(venv / "bin/python"),
        "--index-strategy",
        "unsafe-best-match",
    ]
    subprocess.run(
        install
        + [
            "--no-deps",
            "-r",
            str(ROOT / baseline["requirements_lock"]),
            "--extra-index-url",
            "https://pypi.nvidia.com",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        check=True,
    )
    projects = [ROOT / lock["sources"]["rlinf"]["path"], source_path]
    projects += [
        ROOT / lock["sources"]["isaaclab"]["path"] / "source" / package
        for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")
    ]
    subprocess.run(
        install
        + ["--no-deps"]
        + [arg for path in projects for arg in ("-e", str(path))],
        check=True,
    )
    subprocess.run(
        [
            str(venv / "bin/hf"),
            "download",
            baseline["model_repo_id"],
            "--revision",
            baseline["model_revision"],
            "--local-dir",
            str(ROOT / baseline["model_path"]),
            "--include",
            "config.json",
            "experiment_cfg/metadata.json",
            "*.safetensors",
            "model.safetensors.index.json",
        ],
        check=True,
    )
    for name, expected in baseline["model_files_sha256"].items():
        assert sha256(ROOT / baseline["model_path"] / name) == expected, name


if __name__ == "__main__":
    main()
