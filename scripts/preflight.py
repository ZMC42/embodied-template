#!/usr/bin/env python3
"""Run dependency, runtime, and WebRTC-path preflight checks."""

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "dependencies.lock.json"
ASSET_ROOT = Path(
    os.environ.get(
        "EMBODIED_ASSET_ROOT",
        "/mnt/nas/Vol2/EmbodiedAI/embodied-template-assets",
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def offline() -> int:
    lock = json.loads(LOCK_PATH.read_text())
    failures = []

    for name, source in lock["sources"].items():
        path = ROOT / source["path"]
        actual = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        ok = actual == source["commit"]
        print(f"source {name}: {'OK' if ok else 'FAIL'} ({actual or 'missing'})")
        if not ok:
            failures.append(name)

    for environment, config in lock["environments"].items():
        for relative, expected in config["lock_inputs"].items():
            path = ROOT / relative
            ok = path.is_file() and sha256(path) == expected
            print(f"lock {environment}/{relative}: {'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(relative)

    for name, asset in lock["assets"].items():
        asset_path = ASSET_ROOT / asset["path"]
        manifest_path = asset_path / ".asset-manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        files = manifest.get("files", {})
        ok = manifest.get("revision") == asset["revision"] and all(
            (asset_path / relative).is_file()
            and sha256(asset_path / relative) == expected
            for relative, expected in files.items()
        )
        ok = ok and bool(files)
        expected_license_sha256 = (
            files.get(asset["license_file"])
            if asset["license_file"]
            else asset["license_terms_sha256"]
        )
        ok = ok and manifest.get("license_sha256") == expected_license_sha256
        print(f"asset {name}: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    return int(bool(failures))


def runtime(environment: str, output: Path) -> int:
    lock = json.loads(LOCK_PATH.read_text())
    required = lock["environments"][environment]["required_packages"]
    packages = {}
    for name in required:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    cuda = None
    if packages["torch"]:
        import torch

        cuda = torch.version.cuda

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    frozen = subprocess.run(
        [
            "uv",
            "pip",
            "freeze",
            "--strict",
            "--exclude-editable",
            "--python",
            sys.executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not frozen:
        raise RuntimeError("The environment lock is empty")
    requirements_lock = output.with_name(f"{environment}-requirements.lock.txt")
    requirements_lock.parent.mkdir(parents=True, exist_ok=True)
    requirements_lock.write_text("\n".join(frozen) + "\n")
    report = {
        "environment": environment,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "expected_python": lock["environments"][environment]["python"],
        "packages": packages,
        "cuda_runtime": cuda,
        "driver": driver,
        "isaac_sim": (packages["isaacsim"] if environment == "ppo" else None),
        "isaaclab_source": (ROOT / "third_party" / "IsaacLab" / "VERSION")
        .read_text()
        .strip(),
        "requirements_lock": str(requirements_lock),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    expected_python = lock["environments"][environment]["python"]
    python_ok = (
        platform.python_version().startswith(expected_python.removesuffix("*"))
        if expected_python.endswith("*")
        else platform.python_version() == expected_python
    )
    packages_ok = all(packages[name] for name in required)
    runtime_ok = (
        driver
        and cuda == lock["runtime"]["cuda"]
        and report["isaaclab_source"] == lock["runtime"]["isaaclab"]
    )
    if environment == "ppo":
        runtime_ok = runtime_ok and (report["isaac_sim"] or "").startswith(
            lock["runtime"]["isaac_sim"]
        )
    return int(not (python_ok and packages_ok and runtime_ok))


def network(mode: str, client_ip: str, port: int) -> int:
    import socket

    address = ipaddress.ip_address(client_ip)
    if (
        mode == "private"
        and not address.is_private
        and address not in ipaddress.ip_network("100.64.0.0/10")
    ):
        print("private mode requires a private/VPN client IP", file=sys.stderr)
        return 1
    if mode == "public":
        public_ip = ipaddress.ip_address(os.environ["PUBLIC_IP"])
        if public_ip.is_private:
            print("PUBLIC_IP must be publicly routable", file=sys.stderr)
            return 1
    print(f"waiting for UDP probe from {client_ip} on port {port}")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
        listener.settimeout(60)
        listener.bind(("0.0.0.0", port))
        payload, peer = listener.recvfrom(64)
        listener.sendto(b"embodied-preflight-ok", peer)
        acknowledgement, acknowledgement_peer = listener.recvfrom(64)
    ok = (
        peer[0] == client_ip
        and payload == b"embodied-preflight"
        and acknowledgement_peer == peer
        and acknowledgement == b"embodied-preflight-ack"
    )
    print(f"WebRTC UDP path: {'OK' if ok else 'FAIL'}")
    return int(not ok)


def network_client(server_ip: str, port: int) -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(60)
        client.sendto(b"embodied-preflight", (server_ip, port))
        response, peer = client.recvfrom(64)
        client.sendto(b"embodied-preflight-ack", peer)
    ok = response == b"embodied-preflight-ok"
    print(f"WebRTC UDP path: {'OK' if ok else 'FAIL'}")
    return int(not ok)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("offline")
    runtime_parser = subparsers.add_parser("runtime")
    runtime_parser.add_argument("environment", choices=("sft", "ppo"))
    runtime_parser.add_argument("--output", type=Path, required=True)
    network_parser = subparsers.add_parser("network")
    network_parser.add_argument("mode", choices=("private", "public"))
    network_parser.add_argument("client_ip")
    network_parser.add_argument("--port", type=int, default=49100)
    client_parser = subparsers.add_parser("network-client")
    client_parser.add_argument("server_ip")
    client_parser.add_argument("--port", type=int, default=49100)
    args = parser.parse_args()

    if args.command == "offline":
        return offline()
    if args.command == "runtime":
        return runtime(args.environment, args.output)
    if args.command == "network-client":
        return network_client(args.server_ip, args.port)
    return network(args.mode, args.client_ip, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
