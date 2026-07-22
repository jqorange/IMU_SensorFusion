"""Adapters for MARG algorithms from the Python ``ahrs`` package."""

from __future__ import annotations

import numpy as np
from ahrs.filters import EKF, Madgwick

from imu_fusion.config import FusionConfig

_STANDARD_GRAVITY = 9.80665


def _descending_quality(value: float, start: float, reject: float) -> float:
    """Map an error metric to a continuous quality in the closed interval [0, 1]."""
    return float(np.clip((reject - value) / (reject - start), 0.0, 1.0))


def _magnetic_sample_quality(
    angular_speed_dps: float,
    acceleration_error_g: float,
    magnetic_norm_error_fraction: float,
    innovation_degrees: float,
    config: FusionConfig,
) -> float:
    """Return the weakest of four independent magnetic reliability checks."""
    qualities = (
        _descending_quality(
            angular_speed_dps,
            config.adaptive_mag_turn_start_dps,
            config.adaptive_mag_turn_reject_dps,
        ),
        _descending_quality(
            acceleration_error_g,
            config.adaptive_mag_accel_start_g,
            config.adaptive_mag_accel_reject_g,
        ),
        _descending_quality(
            magnetic_norm_error_fraction,
            config.adaptive_mag_norm_start_fraction,
            config.adaptive_mag_norm_reject_fraction,
        ),
        _descending_quality(
            innovation_degrees,
            config.adaptive_mag_innovation_start_degrees,
            config.adaptive_mag_innovation_reject_degrees,
        ),
    )
    return min(qualities)


def _run_adaptive_ekf(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    magnetometer: np.ndarray,
    config: FusionConfig,
    magnetic_reference: float | np.ndarray | None,
) -> np.ndarray:
    """Run EKF with a smoothly varying magnetometer measurement variance.

    Magnetic confidence drops quickly during fast rotation, non-gravitational
    acceleration, abnormal field magnitude, or a large predicted magnetic
    residual.  It recovers slowly after the motion settles, so a locally
    distorted field cannot immediately pull yaw back in one direction.
    """
    ekf = EKF(
        gyr=gyroscope[:1],
        acc=acceleration[:1],
        mag=magnetometer[:1],
        frequency=config.sample_rate_hz,
        frame=config.reference_frame.upper(),
        noises=np.asarray(config.ekf_variances, dtype=np.float64),
        magnetic_ref=magnetic_reference,
    )
    quaternions = np.empty((len(gyroscope), 4), dtype=np.float64)
    quaternions[0] = ekf.Q[0]
    # EKF._compute_all() does not call update() for a one-sample initializer,
    # while EKF.h() uses ``z`` only to select its six-axis MARG observation.
    ekf.z = np.r_[
        acceleration[0] / np.linalg.norm(acceleration[0]),
        magnetometer[0] / np.linalg.norm(magnetometer[0]),
    ]
    magnetic_norms = np.linalg.norm(magnetometer, axis=1)
    valid_norms = magnetic_norms[np.isfinite(magnetic_norms) & (magnetic_norms > 0.0)]
    if valid_norms.size == 0:
        raise ValueError("Magnetometer contains no finite, non-zero samples.")
    reference_norm = float(np.median(valid_norms))
    angular_speeds_dps = np.rad2deg(np.linalg.norm(gyroscope, axis=1))
    acceleration_errors_g = np.abs(
        np.linalg.norm(acceleration, axis=1) / _STANDARD_GRAVITY - 1.0
    )
    magnetic_norm_errors = np.abs(magnetic_norms / reference_norm - 1.0)
    base_mag_variance = config.ekf_var_mag
    weight = 1.0
    time_step = 1.0 / config.sample_rate_hz

    for index in range(1, len(gyroscope)):
        magnetic_norm = magnetic_norms[index]
        angular_speed_dps = angular_speeds_dps[index]
        if angular_speed_dps >= config.adaptive_mag_turn_reject_dps:
            # This is the overwhelmingly common path during animal movement.
            # Avoid computing a magnetic prediction that will be rejected.
            raw_quality = 0.0
        elif not np.isfinite(magnetic_norm) or magnetic_norm <= 0.0:
            raw_quality = 0.0
        else:
            predicted = ekf.f(quaternions[index - 1], gyroscope[index])
            # EKF.h() selects its model from the previous update's z length.
            # Force the MARG model here even if that update intentionally
            # omitted magnetometer data.
            ekf.z = np.empty(6, dtype=np.float64)
            expected_magnetic = ekf.h(predicted)[3:]
            measured_magnetic = magnetometer[index] / magnetic_norm
            cosine = float(np.clip(expected_magnetic @ measured_magnetic, -1.0, 1.0))
            innovation_degrees = float(np.rad2deg(np.arccos(cosine)))
            raw_quality = _magnetic_sample_quality(
                angular_speed_dps=float(angular_speed_dps),
                acceleration_error_g=float(acceleration_errors_g[index]),
                magnetic_norm_error_fraction=float(magnetic_norm_errors[index]),
                innovation_degrees=innovation_degrees,
                config=config,
            )

        time_constant = (
            config.adaptive_mag_fall_time_s
            if raw_quality < weight
            else config.adaptive_mag_recovery_time_s
        )
        smoothing = 1.0 - np.exp(-time_step / time_constant)
        weight += smoothing * (raw_quality - weight)
        effective_weight = max(weight, config.adaptive_mag_min_weight)
        ekf.noises[2] = base_mag_variance / effective_weight
        # A zero quality is a hard rejection, not merely a very large
        # covariance. Passing None makes the upstream EKF perform a true
        # accelerometer-only update, guaranteeing that magnetic heading cannot
        # oppose either turn direction above the reject threshold.
        magnetic_sample = magnetometer[index] if raw_quality > 0.0 else None
        quaternions[index] = ekf.update(
            quaternions[index - 1],
            gyroscope[index],
            acceleration[index],
            magnetic_sample,
        )
    return quaternions


def run_ahrs(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    magnetometer: np.ndarray,
    config: FusionConfig,
    magnetic_reference: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return scalar-first body-to-world quaternions with shape ``(N, 4)``."""
    kwargs = {
        "gyr": np.asarray(gyroscope, dtype=np.float64),
        "acc": np.asarray(acceleration, dtype=np.float64),
        "mag": np.asarray(magnetometer, dtype=np.float64),
        "frequency": float(config.sample_rate_hz),
    }
    if config.ahrs_algorithm.lower() == "ekf":
        if config.adaptive_magnetometer:
            quaternions = _run_adaptive_ekf(
                kwargs["acc"],
                kwargs["gyr"],
                kwargs["mag"],
                config,
                magnetic_reference,
            )
        else:
            quaternions = EKF(
                frame=config.reference_frame.upper(),
                noises=np.asarray(config.ekf_variances, dtype=np.float64),
                magnetic_ref=magnetic_reference,
                **kwargs,
            ).Q
    else:
        quaternions = Madgwick(**kwargs).Q
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if not np.all(np.isfinite(quaternions)) or np.any(norms == 0.0):
        raise FloatingPointError("AHRS produced invalid quaternions.")
    quaternions /= norms
    return quaternions
