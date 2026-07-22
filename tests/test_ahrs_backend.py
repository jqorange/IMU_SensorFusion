"""Tests for adaptive magnetic confidence used by the EKF backend."""

import numpy as np

from imu_fusion.ahrs_backend import _magnetic_sample_quality, run_ahrs
from imu_fusion.config import FusionConfig


def test_clean_stationary_sample_has_full_magnetic_quality() -> None:
    quality = _magnetic_sample_quality(0.0, 0.0, 0.0, 0.0, FusionConfig())
    assert quality == 1.0


def test_fast_turn_rejects_magnetic_correction() -> None:
    config = FusionConfig()
    quality = _magnetic_sample_quality(
        config.adaptive_mag_turn_reject_dps,
        0.0,
        0.0,
        0.0,
        config,
    )
    assert quality == 0.0


def test_magnetic_quality_uses_smooth_linear_ramp() -> None:
    config = FusionConfig()
    midpoint = (
        config.adaptive_mag_innovation_start_degrees
        + config.adaptive_mag_innovation_reject_degrees
    ) / 2.0
    quality = _magnetic_sample_quality(0.0, 0.0, 0.0, midpoint, config)
    assert quality == 0.5


def test_adaptive_ekf_runs_past_single_sample_initializer() -> None:
    rows = 4
    config = FusionConfig()
    gyroscope = np.zeros((rows, 3))
    # Exercise magnetic rejection followed by recovery. The upstream EKF uses
    # the previous observation length to choose its measurement model.
    gyroscope[1, 2] = np.deg2rad(3.0)
    quaternions = run_ahrs(
        acceleration=np.tile([0.0, 0.0, 9.80665], (rows, 1)),
        gyroscope=gyroscope,
        magnetometer=np.tile([20.0, 0.0, 40.0], (rows, 1)),
        config=config,
        magnetic_reference=np.rad2deg(np.arctan2(40.0, 20.0)),
    )
    assert quaternions.shape == (rows, 4)
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0)
