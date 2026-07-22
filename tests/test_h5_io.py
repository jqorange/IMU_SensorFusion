from pathlib import Path

import numpy as np

from imu_fusion.io import read_aligned_imu, read_fusion_result, write_aligned_imu
from imu_fusion.pipeline import FusionResult


def test_aligned_h5_round_trip(tmp_path: Path) -> None:
    source = np.arange(900, dtype=np.float32).reshape(100, 9)
    path = tmp_path / "aligned.h5"
    write_aligned_imu(path, source, sample_rate_hz=100.0)
    restored, time_s = read_aligned_imu(path)
    np.testing.assert_array_equal(restored, source)
    assert time_s is not None
    assert time_s[-1] == 0.99


def test_fusion_h5_round_trip(tmp_path: Path) -> None:
    rows = 20
    result = FusionResult(
        time_s=np.arange(rows) / 100.0,
        quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (rows, 1)),
        quaternion_aligned=np.tile([0.5, 0.5, 0.5, 0.5], (rows, 1)),
        euler_roll_yaw_pitch=np.zeros((rows, 3)),
        world_acceleration=np.ones((rows, 3)),
        world_speed=np.full((rows, 3), 2.0),
    )
    path = tmp_path / "fusion.h5"
    result.write(path)
    restored = read_fusion_result(path)
    assert restored.shape == (rows, 18)
    np.testing.assert_array_equal(restored["speed_z"], 2.0)
