"""Regression tests for magnetic calibration and local field estimation."""

import numpy as np

from imu_fusion.preprocessing import (
    _ellipsoid_calibrate,
    estimate_magnetic_dip_degrees,
)


def test_ellipsoid_calibration_restores_constant_radius() -> None:
    rng = np.random.default_rng(1234)
    sphere = rng.normal(size=(20_000, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    distortion = np.array([[1.4, 0.12, -0.08], [0.0, 0.75, 0.09], [0.0, 0.0, 1.15]])
    biased = sphere @ distortion + np.array([18.0, -27.0, 9.0])

    corrected = _ellipsoid_calibrate(biased)
    radii = np.linalg.norm(corrected, axis=1)

    assert np.std(radii) / np.mean(radii) < 0.01


def test_magnetic_dip_estimation_uses_acc_mag_angle() -> None:
    dip_degrees = 63.0
    dip_radians = np.deg2rad(dip_degrees)
    acceleration = np.tile([0.0, 0.0, -9.81], (1_000, 1))
    magnetometer = np.tile([np.cos(dip_radians), 0.0, np.sin(dip_radians)], (1_000, 1))

    estimated = estimate_magnetic_dip_degrees(acceleration, magnetometer)

    assert np.isclose(estimated, dip_degrees, atol=1e-10)
