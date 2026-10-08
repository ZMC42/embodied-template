"""Exercise server setup in a fresh checkout, with real local Git submodules."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
    )


@pytest.fixture
def server(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-q")
    git(
        upstream,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "source",
    )
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    git(source, "init", "-q")
    for name in ("setup_project.sh", "setup_assets.sh", "project_env.sh"):
        shutil.copy(ROOT / "scripts" / name, source / "scripts" / name)
    shutil.copy(ROOT / ".gitignore", source / ".gitignore")
    for name in ("RLinf", "Isaac-GR00T", "IsaacLab"):
        git(
            source,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(upstream),
            f"third_party/{name}",
        )
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-q",
        "-m",
        "template",
    )
    checkout = tmp_path / "fresh checkout"
    subprocess.run(["git", "clone", "-q", str(source), str(checkout)], check=True)

    # Package installation and GPU commands are external system boundaries.
    tools = tmp_path / "tools"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'venv':\n"
        "    target = next(a for a in args if a.startswith('.venvs/'))\n"
        "    binary = pathlib.Path(target) / 'bin/python'\n"
        "    binary.parent.mkdir(parents=True, exist_ok=True)\n"
        "    if not binary.exists(): binary.symlink_to(sys.executable)\n"
        "if os.environ.get('FAIL_INSTALL') and args[:2] == ['pip', 'install']:\n"
        "    sys.exit(19)\n"
    )
    uv.chmod(0o755)
    for name in ("nvidia-smi", "nvcc"):
        binary = tools / name
        binary.write_text("#!/bin/sh\necho 'test GPU / CUDA'\n")
        binary.chmod(0o755)
    env = os.environ.copy()
    env.pop("EMBODIED_ASSET_ROOT", None)
    env.pop("CUDA_HOME", None)
    env.update(
        HOME=str(tmp_path / "home"),
        PATH=f"{tools}:{env['PATH']}",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="protocol.file.allow",
        GIT_CONFIG_VALUE_0="always",
    )
    return checkout, env


def setup(server, *args):
    checkout, env = server
    return subprocess.run(
        [
            "bash",
            str(checkout / "scripts/setup_project.sh"),
            "--envs",
            "dev",
            "--skip-system-deps",
            *args,
        ],
        cwd=checkout.parent,
        env=env,
        capture_output=True,
        text=True,
    )


def test_fresh_checkout_and_repeat_preserve_asset_root(server):
    checkout, env = server
    assets = checkout.parent / "assets with spaces"
    first = setup(server, "--asset-root", str(assets))
    assert first.returncode == 0, first.stdout + first.stderr
    assert (checkout / "models").resolve() == assets / "models"
    for name in ("RLinf", "Isaac-GR00T", "IsaacLab"):
        assert git(checkout / "third_party" / name, "rev-parse", "HEAD").stdout.strip()
    second = setup(server)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (checkout / "models").resolve() == assets / "models"
    configured = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s" "$EMBODIED_ASSET_ROOT"',
            "bash",
            str(checkout / "scripts/project_env.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert configured.stdout == str(assets)


def test_install_failure_stops_and_reports_log(server):
    checkout, env = server
    env["FAIL_INSTALL"] = "1"
    result = setup(server)
    assert result.returncode == 19
    assert "Setup failed" in result.stdout
    assert "Setup complete" not in result.stdout
    assert list((checkout / "tmp/setup").glob("setup-*.log"))


def test_invalid_environment_fails_before_install(server):
    checkout, _ = server
    result = setup(server, "--envs", "unknown")
    assert result.returncode == 2
    assert "Unknown environment" in result.stderr
    assert not (checkout / ".venvs").exists()
