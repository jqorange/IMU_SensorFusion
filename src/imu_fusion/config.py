"""Typed project configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Change this single value to switch every direct-run entry point to another
# recording. The name must match "<animal>D<day>_<environment>", for example
# "F5D3_outdoor" or "F6D5_outdoor_2".
ACTIVE_SESSION = "F6D9_outdoor"

# Set the exact video for ACTIVE_SESSION. Update this path whenever the active
# session changes; the resolver never scans directories to guess a video.
VIDEO_FILE = Path(
    r"I:\data\FieldRat\2024\F6\day9\F6_outdoor_medium\cam742024-09-19T18_00_13.avi"
)

# Set the exact behavior MAT containing the video timestamps for
# ACTIVE_SESSION. Update this path together with ACTIVE_SESSION and VIDEO_FILE.
TIMESTAMP_MAT_FILE = Path(
    r"I:\data\FieldRat\2024\F6\Merged\day9\3E6_day9\3E6_day9.animal.behavior_corrected.mat"
)

_DEFAULT_MAPPING = (
    (0.0, 0.0, -1.0),
    (0.0, -1.0, 0.0),
    (-1, 0, 0.0),
)


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """All numerical settings needed by preprocessing and fusion."""

    sample_rate_hz: float = 100.0
    raw_sample_rate_hz: float = 1250.0
    channel_count: int = 16
    imu_channels_one_based: tuple[int, ...] = tuple(range(2, 11))
    clip_percentiles: tuple[float, float] = (0.1, 99.9)
    # Genuine rapid turns live in the tails of the gyro distribution. Clipping
    # them with per-session percentiles makes the two turn directions acquire
    # different artificial limits. Accelerometer and magnetometer channels
    # still use percentile clipping for isolated acquisition spikes.
    clip_gyroscope_percentiles: bool = False
    calibrate: bool = True
    ahrs_algorithm: str = "ekf"
    reference_frame: str = "NED"
    ekf_var_gyro: float = 0.3**2
    ekf_var_acc: float = 0.8**2
    # Smaller measurement variance means stronger magnetometer correction.
    # 0.3² trusts magnetic heading about 7.1x more than ahrs' 0.8² default.
    ekf_var_mag: float = 0.05**2
    estimate_magnetic_dip: bool = True
    magnetometer_lowpass_hz: float = 2.0
    magnetometer_lowpass_order: int = 4
    # Adaptive magnetic updates prevent a disturbed field from pulling yaw in
    # one direction during a turn.  All thresholds define smooth ramps rather
    # than hard on/off switches.
    adaptive_magnetometer: bool = True
    # The calibrated, zero-phase low-pass magnetic signal remains useful in
    # ordinary motion. Only fast turns and unreliable observations are gated.
    adaptive_mag_turn_start_dps: float = 10.0
    adaptive_mag_turn_reject_dps: float = 120.0
    adaptive_mag_accel_start_g: float = 0.10
    adaptive_mag_accel_reject_g: float = 0.35
    adaptive_mag_norm_start_fraction: float = 0.10
    adaptive_mag_norm_reject_fraction: float = 0.30
    adaptive_mag_innovation_start_degrees: float = 60.0
    adaptive_mag_innovation_reject_degrees: float = 179.0
    adaptive_mag_min_weight: float = 0.02
    adaptive_mag_fall_time_s: float = 0.10
    adaptive_mag_recovery_time_s: float = 2.0
    sensor_axis_mapping: tuple[tuple[float, float, float], ...] = field(
        default=_DEFAULT_MAPPING
    )
    drift_window_samples: int = 50
    drift_threshold_degrees: float = 300.0
    speed_highpass_hz: float = 0.1
    speed_highpass_order: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0 or self.raw_sample_rate_hz <= 0:
            raise ValueError("Sample rates must be positive.")
        if len(self.imu_channels_one_based) != 9:
            raise ValueError("Exactly nine IMU channels are required.")
        low, high = self.clip_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100.")
        if self.ahrs_algorithm.lower() not in {"ekf", "madgwick"}:
            raise ValueError("ahrs_algorithm must be 'ekf' or 'madgwick'.")
        if self.reference_frame.upper() not in {"NED", "ENU"}:
            raise ValueError("reference_frame must be NED or ENU.")
        if min(self.ekf_variances) <= 0.0:
            raise ValueError("EKF sensor variances must be positive.")
        adaptive_ranges = (
            (
                self.adaptive_mag_turn_start_dps,
                self.adaptive_mag_turn_reject_dps,
            ),
            (
                self.adaptive_mag_accel_start_g,
                self.adaptive_mag_accel_reject_g,
            ),
            (
                self.adaptive_mag_norm_start_fraction,
                self.adaptive_mag_norm_reject_fraction,
            ),
            (
                self.adaptive_mag_innovation_start_degrees,
                self.adaptive_mag_innovation_reject_degrees,
            ),
        )
        if any(start < 0.0 or reject <= start for start, reject in adaptive_ranges):
            raise ValueError(
                "Adaptive magnetometer reject thresholds must exceed non-negative "
                "start thresholds."
            )
        if not 0.0 < self.adaptive_mag_min_weight <= 1.0:
            raise ValueError("adaptive_mag_min_weight must be in (0, 1].")
        if (
            min(
                self.adaptive_mag_fall_time_s,
                self.adaptive_mag_recovery_time_s,
            )
            <= 0.0
        ):
            raise ValueError("Adaptive magnetometer time constants must be positive.")
        mapping = self.axis_mapping_array
        if not np.allclose(mapping.T @ mapping, np.eye(3), atol=1e-9):
            raise ValueError("sensor_axis_mapping must be orthonormal.")
        if not np.isclose(np.linalg.det(mapping), 1.0):
            raise ValueError("sensor_axis_mapping must be a proper rotation (det=1).")
        nyquist = self.sample_rate_hz / 2.0
        if not 0.0 < self.magnetometer_lowpass_hz < nyquist:
            raise ValueError("magnetometer_lowpass_hz must be below Nyquist.")
        if self.magnetometer_lowpass_order < 1:
            raise ValueError("magnetometer_lowpass_order must be positive.")
        if not 0.0 < self.speed_highpass_hz < nyquist:
            raise ValueError("speed_highpass_hz must be below Nyquist.")

    @property
    def axis_mapping_array(self) -> np.ndarray:
        """Return the configured B-prime-to-B rotation matrix."""
        return np.asarray(self.sensor_axis_mapping, dtype=np.float64)

    @property
    def ekf_variances(self) -> tuple[float, float, float]:
        """Return gyroscope, accelerometer, and magnetometer variances."""
        return self.ekf_var_gyro, self.ekf_var_acc, self.ekf_var_mag
