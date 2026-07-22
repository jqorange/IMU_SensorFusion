"""Create target-rate start-aligned inputs from raw analog recordings."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal

from imu_fusion.config import FusionConfig
from imu_fusion.io import read_analog_channels
from imu_fusion.preprocessing import percentile_clip_imu


def parse_session_start(folder_name: str) -> datetime:
    """Parse ``<number>_yyyyMMdd_HHmmss.SSS[_...]`` like MATLAB."""
    parts = folder_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"No session datetime in {folder_name!r}.")
    return datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S.%f")


def prepare_aligned_segment(
    analog_path: str | Path,
    config: FusionConfig,
    start_offset_s: float,
    duration_s: float,
) -> np.ndarray:
    """Read, MATLAB-style resample, crop, then percentile-clamp one segment.

    A small filter margin is read on both sides so resampling at a nonzero
    starting point does not create an avoidable edge transient.
    """
    if start_offset_s < 0.0 or duration_s <= 0.0:
        raise ValueError("start_offset_s must be >= 0 and duration_s must be > 0.")
    margin_s = 1.0
    source_i = int(round(config.raw_sample_rate_hz))
    target_i = int(round(config.sample_rate_hz))
    common = int(np.gcd(source_i, target_i))
    up, down = target_i // common, source_i // common
    desired_output_start = int(np.floor(start_offset_s * config.sample_rate_hz))
    nominal_read_start = max(
        0, int(np.floor((start_offset_s - margin_s) * config.raw_sample_rate_hz))
    )
    # Anchor the subset on a global polyphase boundary. This preserves the
    # phase of resample(full_recording) while still avoiding a full-file read.
    read_start_sample = nominal_read_start - nominal_read_start % down
    subset_output_start = read_start_sample * up // down
    first = desired_output_start - subset_output_start
    count = int(np.floor(duration_s * config.sample_rate_hz))
    requested_output_count = first + count + int(np.ceil(margin_s * target_i))
    raw_count = int(np.ceil(requested_output_count * down / up))
    raw = read_analog_channels(
        analog_path,
        config.channel_count,
        config.imu_channels_one_based,
        read_start_sample,
        raw_count,
    )
    resampled = signal.resample_poly(raw, up, down, axis=0)
    segment = resampled[first : first + count]
    if len(segment) != count:
        raise ValueError("Analog file ends before the requested aligned segment.")
    return percentile_clip_imu(
        segment,
        config.clip_percentiles,
        clip_gyroscope=config.clip_gyroscope_percentiles,
    )
