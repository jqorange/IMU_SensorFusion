# IMU Fusion Python

A pure-Python workflow for preparing CE32 nine-axis IMU recordings, estimating
orientation, inspecting Euler-angle distributions, and comparing fused motion
with synchronized video. The numerical stages follow the original MATLAB
workflow where an equivalent open implementation exists.

## Features

- Memory-mapped reading of interleaved 16-channel `analogin.dat` recordings.
- Parallel session extraction from corrected behavior MAT files.
- MATLAB-like 1250 Hz to 100 Hz polyphase resampling and percentile clipping.
- Accelerometer, gyroscope, and ellipsoid magnetometer calibration.
- NED MARG orientation estimation with the `ahrs` EKF.
- Low-pass and adaptive magnetometer weighting for robust yaw estimation.
- Compressed HDF5 input and output throughout the production workflow.
- A responsive video viewer with background decoding, an LRU frame cache, an
  opaque procedural 3D mouse, and three live two-second Euler plots.
- Standalone polar roll, yaw, and pitch distribution plots.

## Requirements

- Windows 10 or later is recommended for the direct-run launchers.
- Python 3.11 or later.
- [uv](https://docs.astral.sh/uv/) for the locked environment.

Install every runtime and development dependency once:

```powershell
uv sync --extra viewer --extra plotting --extra dev
```

The direct-run scripts also bootstrap their required uv environment
automatically.

## Quick Start

### 1. Prepare sessions from behavior data

Edit the user configuration at the top of
`prepare_sessions_from_behavior.py`, especially `DATA_ROOTS` and
`OUTPUT_ROOT`, then run:

```powershell
python prepare_sessions_from_behavior.py
```

The script locates corrected behavior MAT files, splits their timestamp arrays
at large gaps, matches analog recordings, and prepares sessions concurrently.
Each result is written as `aligned_imu_100hz.h5`; the complete index is stored
in `sessions_manifest.h5`. See
[Session Preparation](docs/session_preparation.md).

### 2. Run sensor fusion

Set `ALIGNED_IMU_FILE` and `FUSION_OUTPUT_FILE` at the top of
`run_sensor_fusion.py`, then run the file directly:

```powershell
python run_sensor_fusion.py
```

The output is `fusion_result.h5`. Set `PREPARE_FROM_ANALOG = True` only when
the entry should first extract a single interval from a raw analog file.

`run_sensor_fusion.bat` provides the same direct entry for Windows Explorer.
The optional `imu-fusion` command remains available for automated jobs and
uses `config/default.yaml`.

### 3. Compare fusion with video

Edit these values at the top of `visualize.py`:

```python
FUSION_RESULT_FILE = Path(r"D:\path\to\fusion_result.h5")
VIDEO_FILE = Path(r"D:\path\to\video.avi")
VIDEO_TIMESTAMP_MAT_FILE = Path(r"D:\path\to\behavior_corrected.mat")
TIMESTAMP_KEY = "behavior.timestamps_corrected"
```

Then run `python visualize.py` or double-click `visualize.bat`. Set
`CHOOSE_FILES_ON_START = True` to select the three files at launch.

The viewer supports play/pause, frame stepping, seeking, 0.25x to 4x playback,
and a temporary synchronization offset. Space toggles playback; the left and
right arrow keys step by one frame. Video frames are decoded on a worker thread
and cached as resized RGB arrays to keep large recordings responsive.

The 3D mouse-to-video alignment is controlled by
`CARTOON_WORLD_YAW_ALIGNMENT_DEGREES`. Camera azimuth and elevation affect only
how the model is viewed, not the fused orientation.

### 4. Plot Euler distributions

Set `FUSION_RESULT_FILE` at the top of `plot_euler_distribution.py`, then run:

```powershell
python plot_euler_distribution.py
```

The script creates polar roll, yaw, and pitch histograms and writes
`output/euler_distribution.png` by default.

## Bundled Example

`examples/f3d5/` contains a compact aligned IMU file, fusion result, video clip,
and matching timestamp MAT file. These files can be assigned to the direct
entries for a self-contained viewer demonstration.

`tools/create_example.py` can rebuild an example from a local analog recording,
video, and timestamp CSV. Its default 60-second interval is intentional:
ellipsoid magnetometer calibration requires motion that spans enough field
directions. A shorter interval may work only when it contains sufficiently
varied head rotation.

## Processing Pipeline

1. Read channels 2 through 10 from little-endian, interleaved `int16` analog
   data.
2. Resample from 1250 Hz to 100 Hz with a rational polyphase FIR filter.
3. Crop to the behavior or video interval and clip configured percentiles.
4. Convert ADC counts to physical accelerometer, gyroscope, and magnetometer
   units.
5. Calibrate all three sensor groups and apply ellipsoid magnetometer
   correction.
6. Remove high-frequency magnetic interference with zero-phase low-pass
   filtering, then adjust magnetic EKF variance from observation quality.
7. Estimate scalar-first quaternions in the NED reference frame.
8. Apply the fixed sensor-axis mapping, calculate roll/yaw/pitch, rotate linear
   acceleration into world coordinates, and estimate high-pass-filtered speed.

MATLAB `ahrsfilter` and `magcal` are proprietary. The Python EKF and deterministic
ellipsoid fit therefore preserve the processing intent and coordinate semantics,
but cannot be bit-for-bit implementations of those closed algorithms. See
[MATLAB Equivalence](docs/matlab_equivalence.md).

## Data Formats

Prepared input:

```text
/imu/raw_counts                 float32 [N, 9]
  attrs/sample_rate_hz          float
  attrs/columns                 bytes [9]
```

Fusion output stores datasets under `/fusion`, including `time_s`, original and
aligned scalar-first quaternions, `euler_roll_yaw_pitch`, world acceleration,
and speed. Angles are radians, acceleration is m/s^2, and speed is m/s.

Additional timestamp and large-file details are documented in
[Data and Synchronization](docs/data_and_sync.md).

## Configuration

Direct-run preparation and fusion use `FusionConfig()` from
`src/imu_fusion/config.py` as their single source of truth. The optional CLI
profile in `config/default.yaml` mirrors these defaults.

The active fixed sensor mapping is:

```python
_DEFAULT_MAPPING = (
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
)
```

## Project Layout

```text
config/                         Optional CLI configuration
docs/                           Data and numerical reference documents
src/imu_fusion/                 Reusable package implementation
tests/                          Unit and integration tests
tools/                          Sample-generation utility
plot_euler_distribution.py      Direct Euler distribution entry
prepare_sessions_from_behavior.py  Direct batch preparation entry
run_sensor_fusion.py            Direct fusion entry
visualize.py                    Direct synchronized viewer entry
```

Generated data belongs in `prepared_sessions/` and `output/`; both are ignored
by version control.

## Development

Run formatting, linting, and tests before committing changes:

```powershell
uv run ruff format .
uv run ruff check .
uv run pytest
```

## License

No license has been declared. Add one before distributing the project outside
its current research environment.
