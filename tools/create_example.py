"""Create a compact aligned F3D5 example from the original local recording."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from scipy.io import savemat

from imu_fusion.config import FusionConfig
from imu_fusion.io import write_aligned_imu
from imu_fusion.pipeline import fuse_aligned_imu
from imu_fusion.prepare import prepare_aligned_segment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analog", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-timestamps", required=True, type=Path)
    parser.add_argument("--ephys-start", required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/f3d5"))
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    args = parser.parse_args()

    config = FusionConfig.from_yaml(args.config)
    timestamps = pd.read_csv(args.video_timestamps)
    absolute = pd.to_datetime(timestamps["Timestamp"])
    ephys_start = datetime.fromisoformat(args.ephys_start)
    offset_s = (absolute.iloc[0].to_pydatetime() - ephys_start).total_seconds()
    relative = (absolute - absolute.iloc[0]).dt.total_seconds()
    selected = relative[relative <= args.duration].to_numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned = prepare_aligned_segment(args.analog, config, offset_s, args.duration)
    write_aligned_imu(
        args.output_dir / "aligned_imu_100hz.h5", aligned, config.sample_rate_hz
    )
    fuse_aligned_imu(aligned, config).write(args.output_dir / "fusion_result.h5")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise OSError(f"Cannot open {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    output_video = args.output_dir / "video_sample.mp4"
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    written = 0
    while written < len(selected):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    capture.release()
    writer.release()
    if written == 0:
        raise RuntimeError("No video frames were decoded.")
    savemat(
        args.output_dir / "video_timestamps.mat",
        {"video_timestamps_s": selected[:written]},
        do_compression=True,
    )
    print(
        f"Created {written} frames and {len(aligned)} IMU samples; "
        f"start offset={offset_s:.6f} s"
    )


if __name__ == "__main__":
    main()
