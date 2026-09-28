"""Resolve all files belonging to one named FieldRat session."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from imu_fusion.config import TIMESTAMP_MAT_FILE, VIDEO_FILE

_SESSION_NAME_PATTERN = re.compile(
    r"^(F\d+)D(\d+)_(indoor|outdoor)(?:_(\d+))?$",
    re.IGNORECASE,
)
_SESSION_INFO_PATTERN = re.compile(
    r"^(F\d+)Day(\d+)_(\d+)\s*:\s*(\S+)\s*$",
    re.IGNORECASE,
)

SESSION_INFO_FILE = Path(r"C:\Users\aalabadmin\Documents\Session infor.txt")
BEHAVIOR_STATES_ROOT = Path(
    r"D:\Jiaqi\Projects\transition\states\outputs\behavior_transition_states"
)
POSE_FEATURES_ROOT = Path(r"D:\Jiaqi\Projects\DLC_results")
NEURAL_EVENT_PETH_ROOT = Path(r"D:\Jiaqi\Projects\Neuro\event_peth")
NEURAL_REFERENCE_EVENT = "walk_to_local_search"

# Batch_sensor_fusion exports the viewer CSV files here. Keep this separate
# from SessionFiles.fusion_result_file, which remains the normal output target.
BATCH_IMU_RESULT_ROOT = Path(
    r"D:\Jiaqi\tools\Batch_sensor_fusion\results\IMU_result"
)


@dataclass(frozen=True, slots=True)
class SessionFiles:
    """Paths and synchronization metadata shared by all entry points."""

    name: str
    recording_number: int
    timestamp_segment_index: int
    aligned_imu_file: Path
    fusion_result_file: Path
    video_file: Path
    timestamp_mat_file: Path
    dlc_tracking_file: Path
    behavior_states_file: Path
    pose_features_file: Path
    neural_recording_directory: Path
    neural_session_start_seconds: float | None
    lfp_anchor_seconds: float | None


def find_batch_imu_result(session_name: str) -> Path:
    """Find the Batch_sensor_fusion CSV used by the two viewer entry points."""
    canonical_name = _parse_identity(session_name).canonical_name
    expected = BATCH_IMU_RESULT_ROOT / f"IMU_{canonical_name}.csv"
    if expected.is_file():
        return expected
    matches = sorted(BATCH_IMU_RESULT_ROOT.glob(f"IMU_{canonical_name}*.csv"))
    return matches[0] if matches else expected


@dataclass(frozen=True, slots=True)
class _SessionIdentity:
    animal: str
    day: int
    environment: str

    @property
    def canonical_name(self) -> str:
        return f"{self.animal.upper()}D{self.day}_{self.environment.lower()}"


def _parse_identity(session_name: str) -> _SessionIdentity:
    match = _SESSION_NAME_PATTERN.fullmatch(session_name.strip())
    if match is None:
        raise ValueError(
            "Session name must look like 'F5D3_outdoor', "
            "'F5D5_outdoor_1', or 'F6D6_indoor_2'."
        )
    animal, day, environment, occurrence = match.groups()
    label = environment.lower()
    if occurrence is not None:
        label = f"{label}_{int(occurrence)}"
    return _SessionIdentity(animal.upper(), int(day), label)


def _read_session_info(
    identity: _SessionIdentity,
) -> tuple[int, int]:
    if not SESSION_INFO_FILE.is_file():
        raise FileNotFoundError(
            f"Session index file does not exist: {SESSION_INFO_FILE}"
        )
    day_entries: list[tuple[int, str]] = []
    for line in SESSION_INFO_FILE.read_text(encoding="utf-8-sig").splitlines():
        match = _SESSION_INFO_PATTERN.fullmatch(line.strip())
        if match is None:
            continue
        animal, day, recording_number, label = match.groups()
        if animal.upper() == identity.animal and int(day) == identity.day:
            day_entries.append((int(recording_number), label.lower()))
    if not day_entries:
        raise KeyError(
            f"No entries for {identity.animal} day {identity.day} were found "
            f"in {SESSION_INFO_FILE}."
        )
    day_entries.sort(key=lambda item: item[0])
    for segment_index, (recording_number, label) in enumerate(day_entries, start=1):
        if label == identity.environment:
            return recording_number, segment_index
    available = ", ".join(label for _, label in day_entries)
    raise KeyError(
        f"Session {identity.canonical_name!r} is not listed in "
        f"{SESSION_INFO_FILE}. Available labels for this day: {available}."
    )


def _find_first(
    directory: Path,
    patterns: tuple[str, ...],
    fallback_name: str,
    *,
    recursive: bool = False,
) -> Path:
    if directory.is_dir():
        for pattern in patterns:
            matches = sorted(
                directory.rglob(pattern) if recursive else directory.glob(pattern)
            )
            if matches:
                return matches[0]
    return directory / fallback_name


def _find_aligned_imu(
    project_root: Path,
    behavior_mat_file: Path,
    segment_index: int,
    session_name: str,
) -> Path:
    prepared_root = project_root / "prepared_sessions"
    expected_parent_prefix = f"session_{segment_index:02d}"
    behavior_output = prepared_root / behavior_mat_file.stem
    direct_matches = sorted(
        behavior_output.glob(f"{expected_parent_prefix}*/aligned_imu_100hz.h5")
    )
    if direct_matches:
        return direct_matches[0]
    if prepared_root.is_dir():
        day_token = f"day{_parse_identity(session_name).day}".lower()
        for behavior_directory in prepared_root.iterdir():
            if not behavior_directory.is_dir():
                continue
            if day_token not in behavior_directory.name.lower():
                continue
            matches = sorted(
                behavior_directory.glob(
                    f"{expected_parent_prefix}*/aligned_imu_100hz.h5"
                )
            )
            if matches:
                return matches[0]
    return prepared_root / session_name / "aligned_imu_100hz.h5"


def _read_segment_start(meta_path: Path) -> float:
    """Read one event-PETH session boundary used by the neural pipeline."""
    with meta_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    try:
        segment_start = float(metadata["segment_start_sec"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Neural session metadata has no valid segment_start_sec: {meta_path}"
        ) from error
    if not math.isfinite(segment_start):
        raise ValueError(
            f"Neural segment_start_sec must be finite: {meta_path}"
        )
    return segment_start


def _load_merged_time_bounds(
    recording_directory: Path,
) -> tuple[float, float] | None:
    """Read the true merged recording bounds from a MergePoints MAT file."""
    matches = sorted(recording_directory.glob("*MergePoints.events.mat"))
    if not matches:
        return None
    content = loadmat(matches[0], simplify_cells=True)
    merge_points = content.get("MergePoints")
    if not isinstance(merge_points, dict) or "timestamps" not in merge_points:
        raise ValueError(
            f"MergePoints.timestamps is missing from {matches[0]}."
        )
    timestamps = np.asarray(merge_points["timestamps"], dtype=np.float64)
    if timestamps.ndim != 2 or timestamps.shape[1] != 2 or not len(timestamps):
        raise ValueError(
            f"MergePoints.timestamps must have shape (N, 2): {matches[0]}."
        )
    if (
        not np.all(np.isfinite(timestamps))
        or np.any(timestamps[:, 1] <= timestamps[:, 0])
        or np.any(timestamps[1:, 0] < timestamps[:-1, 1])
    ):
        raise ValueError(
            f"MergePoints timestamps are invalid or overlapping: {matches[0]}."
        )
    return float(timestamps[0, 0]), float(timestamps[-1, 1])


def _resolve_neural_alignment(
    session_name: str,
    recording_directory: Path,
) -> tuple[float | None, float | None]:
    """Align event-PETH session seconds to the merged LFP time axis.

    The selected session starts at its event-PETH ``segment_start_sec``.
    LFP sample zero corresponds to the first timestamp in the recording's
    ``MergePoints.events.mat`` file.
    """
    event_root = NEURAL_EVENT_PETH_ROOT / NEURAL_REFERENCE_EVENT
    filename = f"{NEURAL_REFERENCE_EVENT}__meta.json"
    active_meta = event_root / session_name / filename
    if not active_meta.is_file():
        return None, None

    session_start = _read_segment_start(active_meta)
    merged_bounds = _load_merged_time_bounds(recording_directory)
    if merged_bounds is None:
        return None, None
    merged_start, merged_stop = merged_bounds
    if not merged_start <= session_start <= merged_stop:
        raise ValueError(
            f"Session start {session_start:.6f} s for {session_name} is outside "
            f"merged neural bounds [{merged_start:.6f}, {merged_stop:.6f}] s."
        )
    return session_start, merged_start


def resolve_session(session_name: str, project_root: str | Path) -> SessionFiles:
    """Resolve one compact session name into every project input path."""
    root = Path(project_root)
    identity = _parse_identity(session_name)
    canonical_name = identity.canonical_name
    recording_number, segment_index = _read_session_info(identity)

    video_file = VIDEO_FILE
    dlc_tracking_file = _find_first(
        video_file.parent,
        (f"{video_file.stem}*DLC*.csv", "*DLC*.csv"),
        f"{video_file.stem}_DLC.csv",
    )

    timestamp_mat_file = TIMESTAMP_MAT_FILE
    neural_directory = timestamp_mat_file.parent
    aligned_imu_file = _find_aligned_imu(
        root,
        timestamp_mat_file,
        segment_index,
        canonical_name,
    )
    neural_session_start, lfp_anchor = _resolve_neural_alignment(
        canonical_name,
        neural_directory,
    )

    return SessionFiles(
        name=canonical_name,
        recording_number=recording_number,
        timestamp_segment_index=segment_index,
        aligned_imu_file=aligned_imu_file,
        fusion_result_file=root / "output" / "fusion_result.h5",
        video_file=video_file,
        timestamp_mat_file=timestamp_mat_file,
        dlc_tracking_file=dlc_tracking_file,
        behavior_states_file=(
            BEHAVIOR_STATES_ROOT / f"behavior_states_{canonical_name}.csv"
        ),
        pose_features_file=(
            POSE_FEATURES_ROOT
            / canonical_name
            / f"final_filtered_{canonical_name}_50hz.csv"
        ),
        neural_recording_directory=neural_directory,
        neural_session_start_seconds=neural_session_start,
        lfp_anchor_seconds=lfp_anchor,
    )
