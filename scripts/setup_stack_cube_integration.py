#!/usr/bin/env python3
"""Install the pinned N1.7 + IsaacLab runtime in an independent environment."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "configs/dependencies.lock.json").read_text())
    for source in lock["sources"].values():
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT / source["path"]), "rev-parse", "HEAD"], text=True
        ).strip()
        assert commit == source["commit"], source["path"]
    venv = ROOT / ".venvs/ppo-isaaclab"
    subprocess.run(
        ["uv", "venv", "--allow-existing", str(venv), "--python", "3.11.14"], check=True
    )
    requirements = (
        (ROOT / "configs/ppo_isaaclab_n1_7_requirements.lock.txt")
        .read_text()
        .splitlines()
    )
    staging = ROOT / "tmp/stack-cube-integration"
    staging.mkdir(parents=True, exist_ok=True)
    for group, index in [
        ("gpu", "https://download.pytorch.org/whl/cu128"),
        ("sim", "https://pypi.nvidia.com"),
        ("rest", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ]:
        selected = []
        for line in requirements:
            name = line.split("==")[0]
            kind = (
                "gpu"
                if name in ("torch", "torchvision", "torchaudio")
                else "sim"
                if name.startswith("isaacsim")
                else "rest"
            )
            if kind == group:
                selected.append(line)
        path = staging / f"{group}.txt"
        path.write_text("\n".join(selected) + "\n")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin/python"),
                "--no-deps",
                "--default-index",
                index,
                "-r",
                str(path),
            ],
            check=True,
        )
    projects = [ROOT / lock["sources"][key]["path"] for key in ("rlinf", "isaac_groot")]
    projects += [
        ROOT / lock["sources"]["isaaclab"]["path"] / "source" / name
        for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl")
    ]
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
    report = ROOT / "runs/stack-cube/integration"
    report.mkdir(parents=True, exist_ok=True)
    with (report / "environment.lock.txt").open("w") as output:
        subprocess.run(
            ["uv", "pip", "freeze", "--python", str(venv / "bin/python")],
            stdout=output,
            check=True,
        )
    checked = subprocess.run(
        ["uv", "pip", "check", "--python", str(venv / "bin/python")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (report / "dependency-check.txt").write_text(checked.stdout)


if __name__ == "__main__":
    main()
