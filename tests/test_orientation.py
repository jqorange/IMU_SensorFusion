import numpy as np
from scipy.spatial.transform import Rotation

from imu_fusion.orientation import align_orientation, process_axis_fast


def test_axis_mapping_is_applied_on_right() -> None:
    quaternion = np.array([[1.0, 0.0, 0.0, 0.0]])
    mapping = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    _, matrices = align_orientation(quaternion, mapping)
    np.testing.assert_allclose(matrices[0], mapping, atol=1e-12)


def test_scalar_first_quaternion_rotation() -> None:
    scipy_q = Rotation.from_euler("Z", 30.0, degrees=True).as_quat()
    q_wxyz = scipy_q[[3, 0, 1, 2]][None, :]
    _, matrices = align_orientation(q_wxyz, np.eye(3))
    np.testing.assert_allclose(
        matrices[0], Rotation.from_euler("Z", 30.0, degrees=True).as_matrix()
    )


def test_process_axis_leaves_normal_motion_unchanged() -> None:
    values = np.linspace(-1.0, 1.0, 200)
    np.testing.assert_array_equal(process_axis_fast(values), values)


def test_process_axis_corrects_fast_multi_turn_segment() -> None:
    values = np.zeros(120)
    values[10:71] = np.linspace(0.0, 4.0 * np.pi + 0.3, 61)
    values[71:] = values[70]
    corrected = process_axis_fast(values)
    assert not np.array_equal(corrected, values)
    assert np.all(np.isfinite(corrected))
