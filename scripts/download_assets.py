#!/usr/bin/env python3
"""Download the Hugging Face assets pinned in dependencies.lock.json."""

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs" / "dependencies.lock.json"
DEFAULT_ASSET_ROOT = Path(
    os.environ.get(
        "EMBODIED_ASSET_ROOT",
        "/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", nargs="*", default=["all"])
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text())
    names = list(lock["assets"]) if args.assets == ["all"] else args.assets
    for name in names:
        asset = lock["assets"][name]
        destination = args.asset_root / asset["path"]
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "hf",
                "download",
                asset["repo_id"],
                "--repo-type",
                asset["repo_type"],
                "--revision",
                asset["revision"],
                "--local-dir",
                str(destination),
            ],
            check=True,
        )
        files = {
            str(path.relative_to(destination)): sha256(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
            and ".cache" not in path.relative_to(destination).parts
            and path.name != ".asset-manifest.json"
        }
        if asset["license_file"]:
            license_sha256 = files[asset["license_file"]]
        else:
            api_url = (
                f"https://huggingface.co/api/models/{asset['repo_id']}"
                f"/revision/{asset['revision']}"
            )
            with urllib.request.urlopen(api_url) as response:
                model_info = json.load(response)
            terms = model_info["cardData"]["extra_gated_prompt"]
            license_sha256 = hashlib.sha256(terms.encode()).hexdigest()
            if license_sha256 != asset["license_terms_sha256"]:
                raise RuntimeError("The gated license terms changed")
        manifest = {
            "repo_id": asset["repo_id"],
            "repo_type": asset["repo_type"],
            "revision": asset["revision"],
            "license": asset["license"],
            "license_version": asset["license_version"],
            "license_sha256": license_sha256,
            "verified_by": "huggingface_hub",
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }
        (destination / ".asset-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"{name}: {destination}")


if __name__ == "__main__":
    main()
