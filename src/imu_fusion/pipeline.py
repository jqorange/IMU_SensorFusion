"""End-to-end fusion pipeline corresponding to CE32_scaleIMU_gravity.m."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal

from imu_fusion.ahrs_backend import run_ahrs
from imu_fusion.config import FusionConfig
from imu_fusion.orientation import align_orientation, corrected_euler
from imu_fusion.preprocessing import (
    calibrate_sensors,
    estimate_magnetic_dip_degrees,
    lowpass_magnetometer,
    raw_to_physical,
)


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Numerical products from one start-aligned IMU segment."""

    time_s: np.ndarray
    quaternion: np.ndarray
    quaternion_aligned: np.ndarray
    euler_roll_yaw_pitch: np.ndarray
    world_acceleration: np.ndarray
    world_speed: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Create the stable, viewer-ready output schema."""
        return pd.DataFrame(
            np.column_stack(
                (
                    self.time_s,
                    self.quaternion,
                    self.quaternion_aligned,
                    self.euler_roll_yaw_pitch,
                    self.world_acceleration,
                    self.world_speed,
                )
            ),
            columns=(
                "time_s",
                "quaternion_w",
                "quaternion_x",
                "quaternion_y",
                "quaternion_z",
                "quaternion_aligned_w",
                "quaternion_aligned_x",
                "quaternion_aligned_y",
                "quaternion_aligned_z",
                "roll",
                "yaw",
                "pitch",
                "acc_x",
                "acc_y",
                "acc_z",
                "speed_x",
                "speed_y",
                "speed_z",
            ),
        )

    def write_csv(self, path: str | Path) -> None:
        """Write a fusion result with deterministic numeric formatting."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(target, index=False, float_format="%.10g")

    def write_h5(self, path: str | Path) -> None:
        """Write chunked, fast-compressed native fusion datasets."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        chunk_rows = min(8192, len(self.time_s))
        with h5py.File(target, "w") as handle:
            group = handle.create_group("fusion")
            datasets = {
                "time_s": self.time_s,
                "quaternion": self.quaternion,
                "quaternion_aligned": self.quaternion_aligned,
                "euler_roll_yaw_pitch": self.euler_roll_yaw_pitch,
                "world_acceleration": self.world_acceleration,
                "world_speed": self.world_speed,
            }
            for name, values in datasets.items():
                shape = (
                    (chunk_rows,) if values.ndim == 1 else (chunk_rows, values.shape[1])
                )
                group.create_dataset(
                    name,
                    data=values,
                    chunks=shape,
                    compression="lzf",
                    shuffle=True,
                )
            handle.attrs["format"] = "imu_fusion_result_v1"

    def write(self, path: str | Path) -> None:
        """Write HDF5 by default, retaining CSV only for explicit CSV paths."""
        if Path(path).suffix.lower() in {".h5", ".hdf5"}:
            self.write_h5(path)
        else:
            self.write_csv(path)


def fuse_aligned_imu(raw_counts: np.ndarray, config: FusionConfig) -> FusionResult:
    """Fuse an already start-aligned, target-rate, clipped 9-axis segment."""
    acceleration, gyroscope, magnetometer = raw_to_physical(raw_counts)
    if config.calibrate:
        acceleration, gyroscope, magnetometer = calibrate_sensors(
            acceleration, gyroscope, magnetometer
        )
    magnetometer = lowpass_magnetometer(
        magnetometer,
        config.sample_rate_hz,
        config.magnetometer_lowpass_hz,
        config.magnetometer_lowpass_order,
    )
    magnetic_reference = (
        estimate_magnetic_dip_degrees(acceleration, magnetometer)
        if config.estimate_magnetic_dip
        else None
    )
    quaternion = run_ahrs(
        acceleration,
        gyroscope,
        magnetometer,
        config,
        magnetic_reference,
    )
    aligned_quaternion, aligned_matrices = align_orientation(
        quaternion, config.axis_mapping_array
    )
    euler = corrected_euler(
        aligned_matrices,
        config.drift_window_samples,
        config.drift_threshold_degrees,
    )
    # MATLAB: a_Bprime = (S' * a_B')' == a_B @ S.
    acceleration_aligned = acceleration @ config.axis_mapping_array
    world_acceleration = np.einsum(
        "nij,nj->ni", aligned_matrices, acceleration_aligned, optimize=True
    )
    world_acceleration -= np.median(world_acceleration, axis=0)
    world_speed = np.cumsum(world_acceleration / config.sample_rate_hz, axis=0)
    b, a = signal.butter(
        config.speed_highpass_order,
        config.speed_highpass_hz * 2.0 / config.sample_rate_hz,
        btype="highpass",
    )
    world_speed = signal.filtfilt(b, a, world_speed, axis=0)
    time_s = np.arange(raw_counts.shape[0], dtype=np.float64) / config.sample_rate_hz
    return FusionResult(
        time_s,
        quaternion,
        aligned_quaternion,
        euler,
        world_acceleration,
        world_speed,
    )
