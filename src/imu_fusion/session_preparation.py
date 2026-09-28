"""Prepare fusion-ready IMU sessions from behavior MAT timing information."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from imu_fusion.config import FusionConfig
from imu_fusion.io import write_aligned_imu
from imu_fusion.prepare import prepare_aligned_segment


@dataclass(frozen=True, slots=True)
class BehaviorSegment:
    """One continuous behavior interval in ephys-relative seconds."""

    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Return segment duration in seconds."""
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class AnalogSource:
    """An analog file and the folder defining its local time base."""

    path: Path
    folder: Path
    is_continuous_day_file: bool


def read_session_labels(path: str | Path) -> dict[str, str]:
    """Read and normalize ``F3Day5_1: outdoor`` session labels."""
    source = Path(path)
    labels: dict[str, str] = {}
    pattern = re.compile(
        r"^F(\d+)Day(\d+)_(\d+)\s*:\s*((?:indoor|outdoor)(?:_\d+)?)$",
        flags=re.IGNORECASE,
    )
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(
                f"Invalid session label at {source}:{line_number}: {raw_line!r}"
            )
        key = f"F{int(match.group(1))}D{int(match.group(2))}_{int(match.group(3))}"
        condition = match.group(4).lower()
        if key in labels:
            raise ValueError(f"Duplicate session label {key} in {source}.")
        labels[key] = condition
    if not labels:
        raise ValueError(f"Session label file is empty: {source}")
    return labels


def parse_day_id(path: str | Path) -> str:
    """Extract a normalized day ID such as ``F5D10`` from a path."""
    source = Path(path)
    animal_number: int | None = None
    day_number: int | None = None
    for candidate in (source, *source.parents):
        match = re.search(r"F(\d+)Day(\d+)", candidate.name, flags=re.IGNORECASE)
        if match is not None:
            return f"F{int(match.group(1))}D{int(match.group(2))}"
        animal_match = re.fullmatch(r"F(\d+)", candidate.name, flags=re.IGNORECASE)
        day_match = re.fullmatch(r"day(\d+)", candidate.name, flags=re.IGNORECASE)
        if animal_match is not None:
            animal_number = int(animal_match.group(1))
        if day_match is not None:
            day_number = int(day_match.group(1))
        if animal_number is not None and day_number is not None:
            return f"F{animal_number}D{day_number}"
    raise ValueError(f"Cannot determine an F#Day# identifier from {source}.")


def _source_folder_index(source: AnalogSource) -> int | None:
    """Return the numeric prefix of a per-session recording folder."""
    match = re.match(r"^(\d+)_", source.folder.name)
    return int(match.group(1)) if match is not None else None


def _labels_for_day(
    day_id: str, session_labels: Mapping[str, str]
) -> list[tuple[int, str]]:
    """Return configured folder indices and conditions for one day."""
    prefix = f"{day_id}_"
    selected = [
        (int(key.removeprefix(prefix)), condition)
        for key, condition in session_labels.items()
        if key.startswith(prefix)
    ]
    return sorted(selected)


def read_behavior_timestamps(mat_path: str | Path) -> np.ndarray:
    """Read ``behavior.timestamps_corrected`` from classic or v7.3 MAT files."""
    source = Path(mat_path)
    try:
        content = loadmat(source, variable_names=["behavior"], simplify_cells=True)
        behavior = content.get("behavior")
        if isinstance(behavior, dict) and "timestamps_corrected" in behavior:
            values = behavior["timestamps_corrected"]
        elif "timestamps_corrected" in content:
            values = content["timestamps_corrected"]
        else:
            raise KeyError("behavior.timestamps_corrected")
    except NotImplementedError:
        values = _read_v73_timestamps(source)
    timestamps = np.asarray(values, dtype=np.float64).reshape(-1)
    if timestamps.size < 2:
        raise ValueError(f"Not enough behavior timestamps in {source}.")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"Behavior timestamps must be finite and monotonic: {source}")
    return timestamps


def _read_v73_timestamps(mat_path: Path) -> np.ndarray:
    """Read a MATLAB v7.3 HDF5 dataset, including scalar object references."""
    import h5py

    with h5py.File(mat_path, "r") as handle:
        candidates = (
            "behavior/timestamps_corrected",
            "timestamps_corrected",
        )
        dataset = next((handle[name] for name in candidates if name in handle), None)
        if dataset is None:
            raise KeyError("behavior.timestamps_corrected")
        values = dataset[()]
        if values.dtype.kind == "O":
            reference = values.reshape(-1)[0]
            values = handle[reference][()]
        return np.asarray(values).T


def split_behavior_sessions(
    timestamps: np.ndarray, gap_seconds: float
) -> list[BehaviorSegment]:
    """Split timestamps where adjacent values differ by more than the gap."""
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    breaks = np.flatnonzero(np.diff(timestamps) > gap_seconds)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [timestamps.size - 1]))
    return [
        BehaviorSegment(index + 1, float(timestamps[start]), float(timestamps[end]))
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]


def find_behavior_mats(root: str | Path) -> list[Path]:
    """Find corrected behavior MAT files below a file or directory root."""
    root = Path(root)
    if root.is_file():
        return [root] if _is_behavior_mat(root) else []
    return sorted(path for path in root.rglob("*.mat") if _is_behavior_mat(path))


def _is_behavior_mat(path: Path) -> bool:
    lowered = path.name.lower()
    return "behavior" in lowered and (
        "corrected" in lowered or lowered.endswith("behavior_all.mat")
    )


def discover_analog_sources(
    day_folder: str | Path,
    filename_priority: Sequence[str] = ("analogin.dat", "analogin2.dat"),
) -> list[AnalogSource]:
    """Discover per-session files first, then a continuous day-level file."""
    day_folder = Path(day_folder)
    session_folders: list[Path] = []
    for folder in day_folder.rglob("*"):
        if not folder.is_dir() or not re.match(r"^\d+_", folder.name):
            continue
        if _preferred_analog(folder, filename_priority) is not None:
            session_folders.append(folder)
    if session_folders:
        session_folders.sort(key=_session_sort_key)
        return [
            AnalogSource(
                path=_preferred_analog(folder, filename_priority),  # type: ignore[arg-type]
                folder=folder,
                is_continuous_day_file=False,
            )
            for folder in session_folders
        ]

    day_analog = _preferred_analog(day_folder, filename_priority)
    if day_analog is not None:
        return [AnalogSource(day_analog, day_folder, True)]

    fallback_folders = sorted(
        {
            path.parent
            for name in filename_priority
            for path in day_folder.rglob(name)
            if (path.parent / "timestamp_corrected.csv").is_file()
        }
    )
    return [
        AnalogSource(
            path=_preferred_analog(folder, filename_priority),  # type: ignore[arg-type]
            folder=folder,
            is_continuous_day_file=False,
        )
        for folder in fallback_folders
    ]


def _preferred_analog(folder: Path, priority: Sequence[str]) -> Path | None:
    return next((folder / name for name in priority if (folder / name).is_file()), None)


def _session_sort_key(folder: Path) -> tuple[int, str]:
    match = re.match(r"^(\d+)_", folder.name)
    return (int(match.group(1)) if match else 10**9, folder.name.lower())


def session_file_start_offset(source: AnalogSource) -> float:
    """Calculate local behavior start from session folder and timestamp CSV."""
    timestamp_path = source.folder / "timestamp_corrected.csv"
    if not timestamp_path.is_file():
        raise FileNotFoundError(f"Missing timestamp_corrected.csv in {source.folder}")
    table = pd.read_csv(timestamp_path)
    if table.empty:
        raise ValueError(f"Empty timestamp file: {timestamp_path}")
    timestamp_column = next(
        (name for name in table.columns if name.lower() == "timestamp"),
        table.columns[0],
    )
    data_start = pd.to_datetime(table[timestamp_column].iloc[0]).to_pydatetime()
    recording_start = parse_recording_start(source.folder.name)
    return (data_start - recording_start).total_seconds()


def parse_recording_start(folder_name: str) -> datetime:
    """Parse ``<number>_yyyyMMdd_HHmmss.SSS[_...]`` session folders."""
    match = re.match(r"^\d+_(\d{8})_(\d{6})(?:\.(\d+))?", folder_name)
    if match is None:
        raise ValueError(f"Cannot parse recording start from folder: {folder_name}")
    fraction = (match.group(3) or "0")[:6].ljust(6, "0")
    return datetime.strptime(
        f"{match.group(1)}_{match.group(2)}.{fraction}",
        "%Y%m%d_%H%M%S.%f",
    )


def prepare_behavior_day(
    behavior_mat: str | Path,
    output_root: str | Path,
    config: FusionConfig,
    gap_seconds: float = 60.0,
    analog_priority: Sequence[str] = ("analogin.dat", "analogin2.dat"),
    overwrite: bool = False,
    max_workers: int = 1,
    session_labels: Mapping[str, str] | None = None,
    included_standard_sessions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Prepare every matched session for one corrected behavior MAT file."""
    behavior_mat = Path(behavior_mat)
    output_root = Path(output_root)
    timestamps = read_behavior_timestamps(behavior_mat)
    segments = split_behavior_sessions(timestamps, gap_seconds)
    sources = discover_analog_sources(behavior_mat.parent, analog_priority)
    if not sources:
        raise FileNotFoundError(f"No analogin file found below {behavior_mat.parent}")

    continuous = len(sources) == 1 and sources[0].is_continuous_day_file
    day_id = parse_day_id(behavior_mat) if session_labels is not None else ""
    identities: list[tuple[int, str]] = []
    if session_labels is not None:
        configured = _labels_for_day(day_id, session_labels)
        if not configured:
            raise ValueError(f"Session label file has no entries for {day_id}.")
        if continuous:
            if len(configured) != len(segments):
                raise ValueError(
                    f"{day_id} has {len(segments)} behavior segments but "
                    f"{len(configured)} configured session labels."
                )
            identities = configured
        else:
            condition_by_index = dict(configured)
            sources = [
                source
                for source in sources
                if _source_folder_index(source) in condition_by_index
            ]
            identities = [
                (
                    int(_source_folder_index(source)),  # type: ignore[arg-type]
                    condition_by_index[int(_source_folder_index(source))],  # type: ignore[arg-type]
                )
                for source in sources
            ]

    source_count_invalid = (
        len(sources) != len(segments)
        if session_labels is not None
        else len(sources) < len(segments)
    )
    if not continuous and source_count_invalid:
        raise ValueError(
            f"Found {len(segments)} behavior sessions but {len(sources)} matching "
            f"analog session folders below {behavior_mat.parent}."
        )

    day_name = _safe_name(behavior_mat.stem)
    jobs: list[
        tuple[BehaviorSegment, AnalogSource, float, Path, int | None, str | None]
    ] = []
    for index, segment in enumerate(segments):
        source = sources[0] if continuous else sources[index]
        if identities:
            folder_index, condition = identities[index]
        else:
            folder_index = _source_folder_index(source)
            condition = None
        start_offset = (
            segment.start_s if continuous else session_file_start_offset(source)
        )
        session_label = (
            f"session_{segment.index:02d}"
            if continuous
            else f"session_{segment.index:02d}_{_safe_name(source.folder.name)}"
        )
        output_path = output_root / day_name / session_label / "aligned_imu_100hz.h5"
        jobs.append(
            (segment, source, start_offset, output_path, folder_index, condition)
        )

    if included_standard_sessions is not None:
        included = {name.lower() for name in included_standard_sessions}
        jobs = [
            job
            for job in jobs
            if job[5] is not None and f"{day_id}_{job[5]}".lower() in included
        ]
        if not jobs:
            requested = ", ".join(included_standard_sessions)
            raise ValueError(
                f"No prepared session for {day_id} matches requested names: {requested}"
            )

    def prepare_job(
        job: tuple[BehaviorSegment, AnalogSource, float, Path, int | None, str | None],
    ) -> dict[str, object]:
        segment, source, start_offset, output_path, folder_index, condition = job
        status = "skipped_existing"
        if overwrite or not output_path.exists():
            aligned = prepare_aligned_segment(
                source.path,
                config,
                start_offset,
                segment.duration_s,
            )
            write_aligned_imu(output_path, aligned, config.sample_rate_hz)
            status = "prepared"
        return {
            "session_index": segment.index,
            "day_id": day_id,
            "source_folder_index": folder_index,
            "condition": condition or "",
            "standard_session_name": (
                f"{day_id}_{condition}" if condition is not None else ""
            ),
            "behavior_mat": str(behavior_mat),
            "analog_file": str(source.path),
            "behavior_start_s": segment.start_s,
            "behavior_end_s": segment.end_s,
            "analog_start_offset_s": start_offset,
            "duration_s": segment.duration_s,
            "output_file": str(output_path),
            "status": status,
        }

    worker_count = max(1, min(int(max_workers), len(jobs)))
    if worker_count == 1:
        records = [prepare_job(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            records = list(executor.map(prepare_job, jobs))
    return pd.DataFrame.from_records(records)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")


def write_table_h5(frame: pd.DataFrame, path: str | Path, group_name: str) -> None:
    """Write a small manifest/error table without introducing another format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w") as handle:
        group = handle.create_group(group_name)
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                group.create_dataset(column, data=series.to_numpy(), compression="lzf")
            else:
                string_type = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(
                    column,
                    data=series.fillna("").astype(str).to_numpy(dtype=object),
                    dtype=string_type,
                    compression="lzf",
                )
        handle.attrs["format"] = "imu_fusion_table_v1"
