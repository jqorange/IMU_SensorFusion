"""Nine-axis IMU preprocessing and orientation fusion."""

from imu_fusion.config import FusionConfig
from imu_fusion.pipeline import FusionResult, fuse_aligned_imu

__all__ = ["FusionConfig", "FusionResult", "fuse_aligned_imu"]
