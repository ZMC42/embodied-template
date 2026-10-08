#!/usr/bin/env python3
"""Check GPU kernels and video libraries without loading training assets."""

import argparse
import importlib.metadata
import json
import platform
import sysconfig
from pathlib import Path

import av
import torch
import torchcodec
from flash_attn import flash_attn_func


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Torch cannot access the NVIDIA GPU"
    tensors = [
        torch.randn(
            1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]
    output = flash_attn_func(*tensors)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all(), "FlashAttention output is not finite"
    assert all(torch.isfinite(value.grad).all() for value in tensors)

    report = {
        "python": platform.python_version(),
        "python_headers": str(Path(sysconfig.get_path("include")) / "Python.h"),
        "gpu": torch.cuda.get_device_name(),
        "gpu_capability": torch.cuda.get_device_capability(),
        "cuda_runtime": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "flash-attn", "transformers", "av", "torchcodec")
        },
        "flash_attention_forward_backward": "passed",
        "video_library_imports": "passed",
        "av_libraries": av.library_versions,
        "torchcodec_path": torchcodec.__file__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GPU kernels and video libraries: OK ({args.output})")


if __name__ == "__main__":
    main()
