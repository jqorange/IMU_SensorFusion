"""MATLAB-compatible signal conversion, resampling, and calibration."""

from __future__ import annotations

import numpy as np
from scipy import signal


def resample_poly_matlab_like(
    values: np.ndarray, source_hz: float, target_hz: float
) -> np.ndarray:
    """Polyphase resample using the same rational-rate design class as MATLAB."""
    source_i = int(round(source_hz))
    target_i = int(round(target_hz))
    common = np.gcd(source_i, target_i)
    return signal.resample_poly(values, target_i // common, source_i // common, axis=0)


def percentile_clip(values: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    """Clamp each channel in place, preserving the compact input dtype."""
    low, high = np.percentile(values, percentiles, axis=0)
    low = low.astype(values.dtype, copy=False)
    high = high.astype(values.dtype, copy=False)
    np.clip(values, low, high, out=values)
    return values


def percentile_clip_imu(
    values: np.ndarray,
    percentiles: tuple[float, float],
    *,
    clip_gyroscope: bool,
) -> np.ndarray:
    """Clamp IMU channels while optionally preserving genuine gyro peaks."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError("IMU data must have shape (samples, 9).")
    if clip_gyroscope:
        return percentile_clip(values, percentiles)
    clipped_channels = np.array([0, 1, 2, 6, 7, 8])
    clipped = percentile_clip(values[:, clipped_channels].copy(), percentiles)
    values[:, clipped_channels] = clipped
    return values


def raw_to_physical(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ADC counts to m/s², rad/s, and microtesla."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError("Raw IMU data must have shape (samples, 9).")
    acceleration = values[:, 0:3] / 32768.0 * 8.0 * 9.81
    gyroscope = values[:, 3:6] / 32768.0 * np.deg2rad(2000.0)
    magnetometer = values[:, 6:9] / 32768.0 * np.array([1150.0, 1150.0, 2500.0])
    return acceleration, gyroscope, magnetometer


def calibrate_sensors(
    acceleration: np.ndarray,
    gyroscope: np.ndarray,
    magnetometer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same calibration stages as ``calibrateSensors`` in MATLAB.

    Acceleration magnitude is normalized to 9.81, median gyro bias is removed,
    and magnetometer data receives an affine ellipsoid-to-sphere correction.
    """
    acceleration = np.asarray(acceleration, dtype=np.float64).copy()
    gyroscope = np.asarray(gyroscope, dtype=np.float64).copy()
    magnetometer = np.asarray(magnetometer, dtype=np.float64).copy()
    median_norm = np.median(np.linalg.norm(acceleration, axis=1))
    if not np.isfinite(median_norm) or median_norm <= np.finfo(float).eps:
        raise ValueError("Cannot calibrate zero or invalid accelerometer data.")
    acceleration *= 9.81 / median_norm
    gyroscope -= np.median(gyroscope, axis=0)
    magnetometer = _ellipsoid_calibrate(magnetometer)
    return acceleration, gyroscope, magnetometer


def _ellipsoid_calibrate(values: np.ndarray) -> np.ndarray:
    """Fit a quadratic ellipsoid and apply MATLAB ``(mag-b) @ A`` semantics.

    Unlike covariance whitening, this fit uses the geometric surface equation
    and therefore does not interpret frequently visited animal poses as sensor
    gain. A deterministic subset bounds memory for multi-hour sessions.
    """
    samples = np.asarray(values, dtype=np.float64)
    max_fit_samples = 100_000
    if len(samples) > max_fit_samples:
        indices = np.linspace(0, len(samples) - 1, max_fit_samples, dtype=np.int64)
        fit_values = samples[indices]
    else:
        fit_values = samples

    x_axis, y_axis, z_axis = fit_values.T
    design = np.column_stack(
        (
            x_axis**2,
            y_axis**2,
            z_axis**2,
            2.0 * y_axis * z_axis,
            2.0 * x_axis * z_axis,
            2.0 * x_axis * y_axis,
            x_axis,
            y_axis,
            z_axis,
        )
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.ones(len(design)), rcond=None)
    if rank < design.shape[1]:
        raise ValueError("Magnetometer motion does not span a calibratable ellipsoid.")

    quadratic = np.array(
        [
            [coefficients[0], coefficients[5], coefficients[4]],
            [coefficients[5], coefficients[1], coefficients[3]],
            [coefficients[4], coefficients[3], coefficients[2]],
        ]
    )
    linear = coefficients[6:9]
    center = -0.5 * np.linalg.solve(quadratic, linear)
    radius_term = 1.0 + center @ quadratic @ center
    shape = quadratic / radius_term
    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    if np.any(eigenvalues <= 0.0) or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Magnetometer ellipsoid fit is not positive definite.")
    correction = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    corrected = (samples - center) @ correction
    # EKF normalizes each magnetic sample, but retaining microtesla-like scale
    # makes diagnostics and exported intermediate values easier to interpret.
    source_radius = np.median(np.linalg.norm(samples - center, axis=1))
    corrected_radius = np.median(np.linalg.norm(corrected, axis=1))
    return corrected * (source_radius / corrected_radius)


def estimate_magnetic_dip_degrees(
    acceleration: np.ndarray, magnetometer: np.ndarray
) -> float:
    """Estimate NED magnetic dip from the rotation-invariant sensor angle."""
    acceleration = np.asarray(acceleration, dtype=np.float64)
    magnetometer = np.asarray(magnetometer, dtype=np.float64)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    magnetic_norm = np.linalg.norm(magnetometer, axis=1)
    valid = (
        np.isfinite(acceleration_norm)
        & np.isfinite(magnetic_norm)
        & (acceleration_norm > np.finfo(float).eps)
        & (magnetic_norm > np.finfo(float).eps)
        & (np.abs(acceleration_norm - 9.81) < 1.5)
    )
    if np.count_nonzero(valid) < 100:
        raise ValueError("Not enough near-1g samples to estimate magnetic dip.")
    acceleration_unit = acceleration[valid] / acceleration_norm[valid, None]
    magnetic_unit = magnetometer[valid] / magnetic_norm[valid, None]
    # EKF's NED accelerometer reference is [0, 0, -1], so a downward magnetic
    # component has the opposite sign in the measured acc-mag dot product.
    sine_dip = -float(np.median(np.sum(acceleration_unit * magnetic_unit, axis=1)))
    return float(np.rad2deg(np.arcsin(np.clip(sine_dip, -1.0, 1.0))))


def lowpass_magnetometer(
    magnetometer: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """Remove high-frequency magnetic disturbance without phase delay."""
    magnetometer = np.asarray(magnetometer, dtype=np.float64)
    sos = signal.butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=sample_rate_hz,
        output="sos",
    )
    return signal.sosfiltfilt(sos, magnetometer, axis=0)
