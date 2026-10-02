#!/usr/bin/env python3
"""Render synchronized expert RGB views with original state and action labels."""

import argparse
import json

import av
import cv2
import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder

from embodied_template.stack_cube import (
    DATASET,
    ROOT,
    episode_path,
    read_json,
    sha256,
    video_path,
    write_json,
)


def replay(episode, output):
    df = pd.read_parquet(episode_path(DATASET, episode))
    front = VideoDecoder(
        str(video_path(DATASET, episode, "image")),
        dimension_order="NHWC",
        num_ffmpeg_threads=1,
    )
    wrist = VideoDecoder(
        str(video_path(DATASET, episode, "wrist_image")),
        dimension_order="NHWC",
        num_ffmpeg_threads=1,
    )
    assert len(front) == len(wrist) == len(df)
    task = json.loads((DATASET / "meta/tasks.jsonl").read_text())["task"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width, stream.height, stream.pix_fmt = 1024, 640, "yuv420p"
        stream.options = {"crf": "20"}
        for i, row in df.iterrows():
            frame = np.zeros((640, 1024, 3), np.uint8)
            for camera, x in [(front, 0), (wrist, 512)]:
                frame[40:552, x : x + 512] = cv2.resize(camera[i].numpy(), (512, 512))
            lines = [
                ("FRONT / image", 10, 28),
                ("WRIST / wrist_image", 522, 28),
                (
                    f"Episode {episode} | frame {i}/{len(df) - 1} "
                    f"| t={row.timestamp:.2f}s",
                    10,
                    575,
                ),
                (
                    "raw state (xyz, Euler xyz, signed fingers): "
                    + np.array2string(
                        row["observation.state"], precision=3, max_line_width=200
                    ),
                    10,
                    596,
                ),
                (
                    "action (relative IK, gripper sign): "
                    + np.array2string(row.action, precision=3, max_line_width=200),
                    10,
                    617,
                ),
                (task, 10, 638),
            ]
            for text, x, y in lines:
                cv2.putText(
                    frame,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(frame, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    write_json(
        output.with_suffix(".json"),
        {
            "stage": "expert_dataset_replay",
            "episode": episode,
            "frames": len(df),
            "fps": 20,
            "task": task,
            "state_rotation": "raw Euler xyz, radians",
            "source_revision": read_json(DATASET / ".asset-manifest.json")["revision"],
            "source_success": "not labeled; public expert demonstration",
            "video_sha256": sha256(output),
        },
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=lambda x: ROOT / x)
    args = parser.parse_args()
    output = (
        args.output
        or ROOT
        / f"runs/stack-cube/visualization/dataset/episode_{args.episode:06d}.mp4"
    )
    replay(args.episode, output)
