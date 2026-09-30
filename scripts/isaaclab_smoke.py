#!/usr/bin/env python3
# ruff: noqa: I001
"""List the pinned stack-cube task and perform one headless reset."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "configs" / "dependencies.lock.json").read_text())
TASK_ID = LOCK["runtime"]["target_task_id"]

os.environ.pop("DISPLAY", None)

from isaaclab.app import AppLauncher


app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry


print(TASK_ID)
assert TASK_ID in gym.registry
cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
cfg.scene.num_envs = 1
env = gym.make(TASK_ID, cfg=cfg).unwrapped
env.reset()
print("headless reset: OK")
env.close()
app.close()
