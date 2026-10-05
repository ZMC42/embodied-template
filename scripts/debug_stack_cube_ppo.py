"""Run the validated PPO smoke pipeline in a fresh directory for VS Code debugging."""

import runpy
import sys
from datetime import datetime
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "tmp/debug/ppo" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    entry = root / "third_party/RLinf/examples/embodiment/train_embodied_agent.py"
    sys.argv = [
        str(entry),
        "--config-path",
        str(root / "experiments/stack_cube/ppo"),
        "--config-name",
        "isaaclab_n1_7_ppo_smoke",
        f"runner.logger.log_path={output}",
        *sys.argv[1:],
    ]
    print(f"PPO output: {output}", flush=True)
    runpy.run_path(str(entry), run_name="__main__")
