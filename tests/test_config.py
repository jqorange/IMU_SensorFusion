import numpy as np
import pytest

from imu_fusion.config import FusionConfig


def test_default_mapping_is_a_valid_rotation() -> None:
    mapping = FusionConfig().axis_mapping_array
    np.testing.assert_allclose(mapping.T @ mapping, np.eye(3))
    assert np.isclose(np.linalg.det(mapping), 1.0)


def test_adaptive_threshold_order_is_validated() -> None:
    with pytest.raises(ValueError, match="reject thresholds"):
        FusionConfig(
            adaptive_mag_turn_start_dps=100.0,
            adaptive_mag_turn_reject_dps=50.0,
        )
