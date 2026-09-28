# Preparing IMU Sessions from Behavior MAT Files

`prepare_sessions_from_behavior.py` separates time alignment from sensor
fusion. It recursively discovers MAT files whose names contain both `behavior`
and `corrected`, then reads `behavior.timestamps_corrected`.

## Session Detection

A new session starts when adjacent timestamps differ by more than
`SESSION_GAP_SECONDS` (60 seconds by default).

Two analog layouts are supported:

1. A day directory contains one continuous `analogin.dat`. Each behavior segment
   is cropped directly from that recording.
2. A day directory contains numbered ephys folders such as
   `1_yyyyMMdd_HHmmss.SSS` and `2_yyyyMMdd_HHmmss.SSS`. Folders are sorted by
   number. `config/session_labels.txt` selects only the numbered folders that
   represent labeled experimental sessions; the selected folders are then
   matched to behavior segments in order. Their start datetimes and corrected
   timestamp files determine local crop offsets. A count mismatch raises an
   error instead of silently pairing the wrong recordings.

Session-label entries use `F<animal>Day<day>_<folder>: <condition>`, for example:

```text
F5Day2_2: outdoor
F5Day2_5: indoor
F8Day2_6: outdoor_2
```

For a continuous day-level `analogin.dat`, configured folder numbers are paired
with timestamp-derived behavior segments in their configured numeric order.

## Usage

Edit the user configuration at the top of the entry file:

```python
DATA_ROOTS = [Path(r"D:\data\F6\day8")]
OUTPUT_ROOT = Path("prepared_sessions")
SESSION_INFO_FILE = Path("config/session_labels.txt")
SESSION_GAP_SECONDS = 60.0
ANALOG_FILE_PRIORITY = ("analogin.dat", "analogin2.dat")
OVERWRITE_EXISTING = False
MAX_WORKERS = 4
```

Run `python prepare_sessions_from_behavior.py` from any working directory, or
double-click `prepare_sessions_from_behavior.bat`.

Use two workers for a mechanical disk. Four to eight workers are generally
appropriate for SSD or NVMe storage when memory permits. Work is parallelized
across behavior files and, for a single behavior file, across sessions.

## Output

```text
prepared_sessions/
  <behavior-file-stem>/
    session_01/
      aligned_imu_100hz.h5
  sessions_manifest.h5
  preparation_errors.h5        Present only when a job fails
```

Each aligned file stores `/imu/raw_counts` as `float32`, using 8192-row chunks
and LZF compression. Sample rate and column names are dataset attributes. Time
is reconstructed from sample index and sample rate rather than duplicated as a
column.

The manifest records the behavior MAT, matched analog file, behavior bounds,
analog crop offset, duration, output path, source folder number, condition, and
standardized name such as `F5D2_outdoor`. Any generated
`aligned_imu_100hz.h5` can be assigned directly to `ALIGNED_IMU_FILE` in
`run_sensor_fusion.py`.
