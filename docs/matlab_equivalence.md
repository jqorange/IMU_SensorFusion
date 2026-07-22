# MATLAB Numerical Equivalence

## Reference Workflow

The Python implementation was organized from these MATLAB responsibilities:

- `batch_IMU_Preprocess.m`: channel reading, resampling, time cropping,
  percentile clipping, and batch layout.
- `CE32_scaleIMU_gravity.m`: physical units, sensor calibration, MARG fusion,
  fixed-axis mapping, Euler angles, world acceleration, and speed.
- `video_vis.m` and related device scripts: video-frame to IMU-time mapping.

## Stage Mapping

| MATLAB stage | Python implementation | Behavior |
|---|---|---|
| `readmulti_frank` | `io.read_analog_channels` | Little-endian, 16-channel interleaved `int16`; selects channels 2-10 |
| `resample(x, 100, 1250)` | `preprocessing.resample_poly_matlab_like` | Rational polyphase FIR resampling |
| `prctile` and clamp | `preprocessing.percentile_clip_imu` | Clips ACC/MAG at 0.1/99.9 percentiles; GYR clipping is configurable |
| `raw2physical` | `preprocessing.raw_to_physical` | Uses the ported ACC/GYR/MAG conversion constants |
| Sensor calibration | `preprocessing.calibrate_sensors` | Gravity scaling, gyro median bias, and magnetic ellipsoid fit |
| MARG filter | `ahrs_backend.estimate_orientation` | NED EKF or optional Madgwick estimator |
| `R_old * S` | `orientation.align_orientation` | Right-multiplies the fixed sensor-axis transform |
| Euler correction | `orientation.corrected_euler` | Unwrap and local integer-revolution correction |
| World acceleration | `pipeline.fuse_aligned_imu` | Rotates mapped acceleration and removes per-axis median |
| Speed | `pipeline.fuse_aligned_imu` | Cumulative integration followed by zero-phase high-pass filtering |

## Closed-Source Boundary

MathWorks `ahrsfilter` and `magcal` are closed implementations. The project uses
the open `ahrs.filters.EKF` and a deterministic quadratic ellipsoid fit. Their
state models, initialization, and numerical details cannot be reproduced
bit-for-bit from the MATLAB APIs.

The default Python path additionally applies a zero-phase magnetic low-pass
filter and adaptive EKF magnetic variance. Quality combines turn rate, dynamic
acceleration, field norm, horizontal field content, and innovation angle. This
retains low-frequency magnetic heading correction while suppressing transient
interference. Disable `adaptive_magnetometer` to use a fixed magnetic variance.

Gyroscope percentile clipping is disabled by default so real high-speed turns
are not truncated independently in each session. Set
`clip_gyroscope_percentiles=True` only when strict legacy clipping is required.

All remaining stages preserve the documented units, channel order, coordinate
operations, and filter settings. Validate the AHRS at trajectory level rather
than expecting sample-identical floating-point output.

## Column and Quaternion Conventions

Fusion Euler output has explicit `roll`, `yaw`, and `pitch` semantics and is
stored in that order. Quaternions are always scalar-first: `w, x, y, z`.
