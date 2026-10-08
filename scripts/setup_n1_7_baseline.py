#!/usr/bin/env python3
"""Install the pinned N1.7 + LIBERO baseline in its own environment."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from download_assets import sha256

ROOT = Path(__file__).resolve().parents[1]


def main(*, skip_assets: bool = False) -> None:
    lock = json.loads((ROOT / "configs/dependencies.lock.json").read_text())
    baseline = lock["baselines"]["n1_7_libero"]
    assert (
        sha256(ROOT / baseline["requirements_lock"])
        == baseline["requirements_lock_sha256"]
    )
    projects = []
    for name in ("rlinf", "isaac_groot"):
        source = lock["sources"][name]
        path = ROOT / source["path"]
        assert (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            == source["commit"]
        )
        projects.append(path)

    rlinf_path = ROOT / baseline["rlinf_source_path"]
    patch = ROOT / baseline["rlinf_patch"]
    assert sha256(patch) == baseline["rlinf_patch_sha256"]
    if not rlinf_path.exists():
        subprocess.run(
            [
                "git",
                "-C",
                str(projects[0]),
                "worktree",
                "add",
                "--detach",
                str(rlinf_path),
                baseline["rlinf_base_commit"],
            ],
            check=True,
        )
        # Replay the archived commit with its original identity and date.
        subprocess.run(
            [
                "git",
                "-C",
                str(rlinf_path),
                "-c",
                "user.name=zmc123",
                "-c",
                "user.email=zjz122342@gmail.com",
                "-c",
                "commit.gpgSign=false",
                "am",
                "--committer-date-is-author-date",
                str(patch),
            ],
            check=True,
        )
    assert (
        subprocess.check_output(
            ["git", "-C", str(rlinf_path), "rev-parse", "HEAD"], text=True
        ).strip()
        == baseline["rlinf_source_commit"]
    )
    projects[0] = rlinf_path

    venv = ROOT / ".venvs/n1.7-libero"
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
        "--no-deps",
        "--index-strategy",
        "unsafe-best-match",
    ]
    subprocess.run(
        install
        + [
            "-r",
            str(ROOT / baseline["requirements_lock"]),
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(venv / "bin/python"),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--ignore-requires-python",
        ]
        + [arg for path in projects for arg in ("-e", str(path))],
        check=True,
    )
    assets = baseline["libero_assets"]
    asset_root = Path(
        os.environ.get(
            "EMBODIED_ASSET_ROOT", "/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets"
        )
    )
    asset_path = asset_root / assets["path"]
    if not skip_assets:
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
                "README.md",
                "libero_spatial/**",
            ],
            check=True,
        )
        for name, expected in baseline["model_files_sha256"].items():
            assert sha256(ROOT / baseline["model_path"] / name) == expected, name
        (ROOT / baseline["model_path"] / ".asset-manifest.json").write_text(
            json.dumps(
                {
                    "repo_id": baseline["model_repo_id"],
                    "revision": baseline["model_revision"],
                    "license_file": baseline["license_file"],
                    "files": baseline["model_files_sha256"],
                },
                indent=2,
            )
            + "\n"
        )

        subprocess.run(
            [
                str(venv / "bin/hf"),
                "download",
                assets["repo_id"],
                "--repo-type",
                "dataset",
                "--revision",
                assets["revision"],
                "--local-dir",
                str(asset_path),
            ],
            check=True,
        )
        manifest = {
            "repo_id": assets["repo_id"],
            "revision": assets["revision"],
            "license": assets["license"],
            "files": {
                str(path.relative_to(asset_path)): sha256(path)
                for path in sorted(asset_path.rglob("*"))
                if path.is_file()
                and ".cache" not in path.relative_to(asset_path).parts
                and path.name != ".asset-manifest.json"
            },
        }
        (asset_path / ".asset-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    package = venv / "lib/python3.11/site-packages/libero/libero"
    package_assets = package / "assets"
    if package_assets.is_symlink():
        package_assets.unlink()
    package_assets.symlink_to(asset_path, target_is_directory=True)
    config = {
        "benchmark_root": str(package),
        "datasets": str(package.parent / "datasets"),
    }
    config.update({name: str(package / name) for name in ("assets", "bddl_files")})
    config["init_states"] = str(package / "init_files")
    config_dir = venv / "libero-config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yaml").write_text(
        "".join(f"{key}: {value}\n" for key, value in config.items())
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-assets", action="store_true")
    main(skip_assets=parser.parse_args().skip_assets)
