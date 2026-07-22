import numpy as np

from imu_fusion.preprocessing import (
    lowpass_magnetometer,
    percentile_clip,
    percentile_clip_imu,
    raw_to_physical,
)


def test_raw_scale_matches_matlab_constants() -> None:
    raw = np.full((1, 9), 32768.0)
    acc, gyr, mag = raw_to_physical(raw)
    np.testing.assert_allclose(acc, [[8.0 * 9.81] * 3])
    np.testing.assert_allclose(gyr, [[np.deg2rad(2000.0)] * 3])
    np.testing.assert_allclose(mag, [[1150.0, 1150.0, 2500.0]])


def test_percentile_clip_is_columnwise() -> None:
    values = np.column_stack((np.arange(101), np.arange(101) * 10.0))
    clipped = percentile_clip(values, (10.0, 90.0))
    np.testing.assert_allclose(clipped[0], [10.0, 100.0])
    np.testing.assert_allclose(clipped[-1], [90.0, 900.0])


def test_imu_clipping_preserves_gyroscope_tails() -> None:
    values = np.tile(np.arange(101, dtype=float)[:, None], (1, 9))
    clipped = percentile_clip_imu(
        values.copy(),
        (10.0, 90.0),
        clip_gyroscope=False,
    )
    np.testing.assert_allclose(clipped[-1, [0, 1, 2, 6, 7, 8]], 90.0)
    np.testing.assert_allclose(clipped[-1, 3:6], 100.0)


def test_magnetometer_lowpass_removes_high_frequency_without_delay() -> None:
    sample_rate = 100.0
    time_s = np.arange(1000) / sample_rate
    slow = np.sin(2.0 * np.pi * 0.5 * time_s)
    fast = 0.5 * np.sin(2.0 * np.pi * 20.0 * time_s)
    values = np.column_stack((slow + fast, slow + fast, slow + fast))
    filtered = lowpass_magnetometer(values, sample_rate, 2.0, 4)
    interior = slice(100, -100)
    np.testing.assert_allclose(filtered[interior, 0], slow[interior], atol=0.03)
