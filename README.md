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

### Select one session globally

Edit `ACTIVE_SESSION`, `VIDEO_FILE`, and `TIMESTAMP_MAT_FILE` in
`src/imu_fusion/config.py`:

```python
ACTIVE_SESSION = "F5D3_outdoor"
VIDEO_FILE = Path(r"I:\path\to\the\exact_session_video.avi")
TIMESTAMP_MAT_FILE = Path(r"I:\path\to\the\exact_behavior.mat")
```

Names follow `<animal>D<day>_<environment>`, including numbered environments
such as `F5D5_outdoor_1`. The resolver combines this name with
`C:\Users\aalabadmin\Documents\Session infor.txt` to select the recording
number and timestamp segment. The video and behavior MAT always come directly
from `VIDEO_FILE` and `TIMESTAMP_MAT_FILE`; neither input is guessed by scanning
directories. The exact video stem is then used to locate the matching DLC
tracking CSV. The resolver also locates the prepared IMU, behavior-state,
pose-feature, and neural recording paths. `run_sensor_fusion.py`,
`visualize.py`, `visualize_behavior.py`, and `plot_euler_distribution.py` all
use this same selection.

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

After selecting `ACTIVE_SESSION`, run the file directly:

```powershell
python run_sensor_fusion.py
```

The output is `fusion_result.h5`. Set `PREPARE_FROM_ANALOG = True` only when
the entry should first extract a single interval from a raw analog file.

`run_sensor_fusion.bat` provides the same direct entry for Windows Explorer.
The `imu-fusion` command remains available for automated jobs and uses the
same `FusionConfig()` defaults from `src/imu_fusion/config.py`.

### 3. Compare fusion with video

Run `python visualize.py` or double-click `visualize.bat`. Inputs are resolved
from `ACTIVE_SESSION`. Set `CHOOSE_FILES_ON_START = True` to override them with
file dialogs for one launch.

The viewer supports play/pause, frame stepping, stable drag seeking, 0.25x to
4x playback, an editable playback FPS, and a temporary synchronization offset.
The FPS field defaults to the detected source-video frame rate; its value is
multiplied by the selected Speed setting. Hold either frame button or an arrow
key to scan quickly through frames. Space toggles playback. Video frames are
decoded on a worker thread and cached as resized RGB arrays to keep large
recordings responsive.

Click the video canvas and use the mouse wheel to zoom. The enlarged view follows
the frame-matched DeepLabCut `bodyCenter1` position automatically, so manual
panning is disabled. Double-clicking resets the view to fit the full frame. The
compass remains fixed as a screen overlay.

The mouse viewport includes a faint world-horizontal reference plane and a
fixed North/East compass. The video includes a second fixed compass rotated only
by the configured video-to-IMU yaw alignment and fixed camera-image rotation,
never by momentary head motion. Adjust `VIDEO_COMPASS_ROTATION_DEGREES` when a
camera is installed with a different image orientation.

The 3D mouse-to-video alignment is controlled by
`CARTOON_WORLD_YAW_ALIGNMENT_DEGREES`. Camera azimuth and elevation affect only
how the model is viewed, not the fused orientation. The camera uses the fixed
azimuth/elevation basis from the reference `IMU_SensorFusion` project; it does
not derive a session-dependent camera roll.

### 4. Inspect fusion, pose, speed, and behavior together

`visualize_behavior.py` is a separate extended viewer. It preserves the video,
3D mouse, synchronization, seeking, playback, frame stepping, compass, and
wheel-zoom behavior of `visualize.py`. The upper row contains only the video
and reconstructed mouse head. The lower row has three two-panel columns:

- roll/yaw/pitch above `bodyCenter1_v` speed;
- current state and enlarged fading pose above the four behavior scores;
- centered five-second LFP above a dark-themed spike raster.

Set
the session resolver to a recording folder containing a Neuroscope XML file,
an `.lfp` or `.eeg` file, and a CellExplorer
`*cell_metrics.cellinfo.mat` file. The selected LFP channel is copied into
memory once for responsive seeking, but only for the selected session plus its
display margins rather than for the entire day. Neurons are filtered by session
mean firing rate, and the raster is sorted by smoothed peak time in the current
window.

Neural session boundaries follow
`D:\Jiaqi\Projects\Neuro\plot_population_behavior_window.py`. The selected
session start is read from
`event_peth/walk_to_local_search/<session>/walk_to_local_search__meta.json`.
Unlike that reference script's earliest-session subtraction, LFP sample zero
is anchored to `MergePoints.timestamps[0, 0]` from the recording's
`*MergePoints.events.mat`. This is the actual time origin of a full-day merged
LFP. The code validates that the selected session falls inside the merged
time bounds. No numeric LFP anchor is hard-coded. If the event-PETH or
MergePoints metadata is unavailable, the neural panels are disabled instead
of silently assuming an alignment.
When the configured recording directory is unavailable, the viewer remains
usable and shows disabled neural panels with a configuration message.

Select `ACTIVE_SESSION`, then run:

```powershell
python visualize_behavior.py
```

The two viewer entry points read their default IMU result from
`D:\\Jiaqi\\tools\\Batch_sensor_fusion\\results\\IMU_result`, using the
matching `IMU_<session>.csv` file. These batch CSVs contain roll, yaw, pitch,
world acceleration, and world speed at 100 Hz. The viewer reconstructs the
aligned quaternion from the Euler columns so the 3D mouse remains available;
video, timestamp, DLC, behavior, pose, and neural inputs continue to use the
same session resolver as before.

Alternatively, double-click `visualize_behavior.bat`. The behavior and pose
tables are synchronized to elapsed fusion/video time at `BEHAVIOR_SAMPLE_RATE_HZ`.
In the filtered pose format, `bodyCenter1` is the local origin and the missing
`bodyCenter2_y_point` is zero by definition; the viewer reconstructs those two
coordinates before drawing the skeleton.

### 5. Plot Euler distributions

Select `ACTIVE_SESSION`, then run:

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

Every preparation and fusion entry point uses `FusionConfig()` from
`src/imu_fusion/config.py` as its single source of truth.

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
config/                         Session-label data
docs/                           Data and numerical reference documents
src/imu_fusion/                 Reusable package implementation
tests/                          Unit and integration tests
tools/                          Sample-generation utility
plot_euler_distribution.py      Direct Euler distribution entry
prepare_sessions_from_behavior.py  Direct batch preparation entry
run_sensor_fusion.py            Direct fusion entry
visualize.py                    Direct synchronized viewer entry
visualize_behavior.py           Extended fusion/pose/behavior viewer entry
visualize_behavior.bat          Windows launcher for the extended viewer
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
