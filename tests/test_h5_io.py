from pathlib import Path

import numpy as np

from imu_fusion.io import (
    read_aligned_imu,
    read_fusion_result,
    read_fusion_view_data,
    write_aligned_imu,
)
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


def test_batch_fusion_csv_is_viewer_compatible(tmp_path: Path) -> None:
    path = tmp_path / "IMU_F5D10_outdoor.csv"
    path.write_text(
        "roll,yaw,pitch,acc_x,acc_y,acc_z,speed_x,speed_y,speed_z\n"
        "0.1,0.2,0.3,0,0,0,0,0,0\n"
        "0.2,0.3,0.4,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    time_s, euler, quaternion = read_fusion_view_data(path)
    np.testing.assert_allclose(time_s, [0.0, 0.01])
    np.testing.assert_allclose(euler[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(np.linalg.norm(quaternion, axis=1), 1.0)
