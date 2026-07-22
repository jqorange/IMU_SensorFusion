"""Quaternion alignment, Euler extraction, and drift post-processing."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def wxyz_to_rotation(quaternions: np.ndarray) -> Rotation:
    """Convert scalar-first quaternions to SciPy rotations."""
    q = np.asarray(quaternions, dtype=np.float64)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    """Convert SciPy rotations to scalar-first quaternions."""
    q = rotation.as_quat()
    return q[:, [3, 0, 1, 2]]


def align_orientation(
    quaternions: np.ndarray, mapping: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``R_aligned = R_original @ mapping`` to every sample."""
    matrices = wxyz_to_rotation(quaternions).as_matrix()
    aligned_matrices = matrices @ np.asarray(mapping, dtype=np.float64)
    aligned = rotation_to_wxyz(Rotation.from_matrix(aligned_matrices))
    return aligned, aligned_matrices


def corrected_euler(
    aligned_matrices: np.ndarray, window: int, threshold_degrees: float
) -> np.ndarray:
    """Extract MATLAB ZYX Euler angles and apply its local 2π correction.

    Return order is ``[roll, yaw, pitch]`` to match the selected MATLAB
    ``CE32_scaleIMU_gravity.m`` implementation and legacy output CSV.
    """
    yaw_pitch_roll = Rotation.from_matrix(aligned_matrices).as_euler("ZYX")
    yaw = process_axis_fast(np.unwrap(yaw_pitch_roll[:, 0]), window, threshold_degrees)
    pitch = process_axis_fast(
        np.unwrap(yaw_pitch_roll[:, 1]), window, threshold_degrees
    )
    roll = process_axis_fast(np.unwrap(yaw_pitch_roll[:, 2]), window, threshold_degrees)
    return np.column_stack((roll, yaw, pitch))


def process_axis_fast(
    values: np.ndarray, window: int = 50, threshold_degrees: float = 300.0
) -> np.ndarray:
    """Port of MATLAB ``processAxis_fast`` using zero-based intervals."""
    source = np.asarray(values, dtype=np.float64)
    output = source.copy()
    if source.size <= window:
        return output
    flagged = np.flatnonzero(
        np.abs(source[window:] - source[:-window]) > np.deg2rad(threshold_degrees)
    )
    if flagged.size == 0:
        return output
    intervals: list[tuple[int, int]] = []
    start = int(flagged[0])
    end = start + window
    for candidate in flagged[1:]:
        candidate = int(candidate)
        if candidate <= end:
            end = max(end, candidate + window)
        else:
            intervals.append((start, end))
            start, end = candidate, candidate + window
    intervals.append((start, end))
    for start, end in intervals:
        end = min(end, source.size - 1)
        if start >= end:
            continue
        raw_delta = source[end] - source[start]
        # MATLAB round is half-away-from-zero; NumPy uses bankers rounding.
        ratio = raw_delta / (2.0 * np.pi)
        turns = np.sign(ratio) * np.floor(np.abs(ratio) + 0.5)
        total_offset = 2.0 * np.pi * turns - raw_delta
        output[start : end + 1] = source[start : end + 1] + np.linspace(
            0.0, total_offset, end - start + 1
        )
    return output
