"""Runtime checks use the installed simulator on servers without a binary archive."""

import json
import platform
import subprocess
import sys
from types import SimpleNamespace

from scripts import preflight


def test_pip_isaacsim_does_not_require_standalone_archive(tmp_path, monkeypatch):
    lock = json.loads(preflight.LOCK_PATH.read_text())
    lock["environments"]["ppo"]["python"] = platform.python_version()
    lock_path = tmp_path / "dependencies.lock.json"
    lock_path.write_text(json.dumps(lock))
    isaaclab = tmp_path / "third_party/IsaacLab"
    isaaclab.mkdir(parents=True)
    (isaaclab / "VERSION").write_text(lock["runtime"]["isaaclab"])
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(preflight, "LOCK_PATH", lock_path)
    monkeypatch.setattr(
        preflight.importlib.metadata,
        "version",
        lambda name: "5.1.0.0" if name == "isaacsim" else "1.0",
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="12.8"))
    )

    def run(command, **kwargs):
        stdout = "580.178.04" if command[0] == "nvidia-smi" else "torch==2.8.0\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(preflight.subprocess, "run", run)
    output = tmp_path / "ppo-environment.json"
    assert preflight.runtime("ppo", output) == 0
    assert json.loads(output.read_text())["isaac_sim"] == "5.1.0.0"
