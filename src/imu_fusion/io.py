"""Input/output helpers for aligned IMU files and Intan-style analog data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

RAW_COLUMNS = (
    "acc_x_raw",
    "acc_y_raw",
    "acc_z_raw",
    "gyr_x_raw",
    "gyr_y_raw",
    "gyr_z_raw",
    "mag_x_raw",
    "mag_y_raw",
    "mag_z_raw",
)


def read_analog_channels(
    path: str | Path,
    channel_count: int,
    channels_one_based: tuple[int, ...],
    start_sample: int = 0,
    sample_count: int | None = None,
    output_dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Read selected interleaved int16 channels without loading the whole file.

    This is the NumPy equivalent of ``readmulti_frank(..., 2:10, ...)``.
    Samples are rows and selected channels are columns.
    """
    source = Path(path)
    total_values = source.stat().st_size // np.dtype("<i2").itemsize
    total_samples = total_values // channel_count
    start = max(0, int(start_sample))
    stop = (
        total_samples
        if sample_count is None
        else min(total_samples, start + sample_count)
    )
    if start >= stop:
        raise ValueError(f"Empty sample range [{start}, {stop}) for {source}.")

    interleaved = np.memmap(
        source,
        dtype="<i2",
        mode="r",
        offset=start * channel_count * np.dtype("<i2").itemsize,
        shape=(stop - start, channel_count),
        order="C",
    )
    indices = np.asarray(channels_one_based, dtype=np.int64) - 1
    return np.asarray(interleaved[:, indices], dtype=output_dtype)


def read_aligned_imu(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read an aligned 9-channel HDF5, CSV, or NPY file.

    HDF5 is the native format. CSV remains available for legacy data.
    """
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(source, "r") as handle:
            dataset = handle["imu/raw_counts"]
            values = dataset[()].astype(np.float32, copy=False)
            sample_rate_hz = float(dataset.attrs["sample_rate_hz"])
        time_s = np.arange(values.shape[0], dtype=np.float64) / sample_rate_hz
        return values, time_s
    if source.suffix.lower() == ".npy":
        values = np.asarray(np.load(source), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 9:
            raise ValueError("NPY input must have shape (samples, 9).")
        return values, None
    table = pd.read_csv(source)
    if all(column in table for column in RAW_COLUMNS):
        values = table.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)
    else:
        numeric = table.select_dtypes(include=[np.number])
        if "time_s" in numeric and numeric.shape[1] >= 10:
            numeric = numeric.drop(columns="time_s")
        if numeric.shape[1] < 9:
            raise ValueError("Input CSV needs nine numeric IMU columns.")
        values = numeric.iloc[:, :9].to_numpy(dtype=np.float64)
    time_s = table["time_s"].to_numpy(dtype=np.float64) if "time_s" in table else None
    return values, time_s


def write_aligned_imu(
    path: str | Path, values: np.ndarray, sample_rate_hz: float
) -> None:
    """Write start-aligned raw-count IMU samples to HDF5 or legacy CSV."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 9:
        raise ValueError("values must have shape (samples, 9).")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(target, "w") as handle:
            group = handle.create_group("imu")
            dataset = group.create_dataset(
                "raw_counts",
                data=values.astype(np.float32, copy=False),
                chunks=(min(8192, len(values)), 9),
                compression="lzf",
                shuffle=True,
            )
            dataset.attrs["sample_rate_hz"] = float(sample_rate_hz)
            dataset.attrs["columns"] = np.asarray(RAW_COLUMNS, dtype="S")
            handle.attrs["format"] = "imu_fusion_aligned_v1"
        return
    table = pd.DataFrame(values, columns=RAW_COLUMNS)
    table.insert(0, "time_s", np.arange(len(table)) / sample_rate_hz)
    table.to_csv(target, index=False, float_format="%.10g")


def read_fusion_result(path: str | Path) -> pd.DataFrame:
    """Load a native fusion HDF5 file or a legacy fusion CSV."""
    source = Path(path)
    if source.suffix.lower() not in {".h5", ".hdf5"}:
        return pd.read_csv(source)
    with h5py.File(source, "r") as handle:
        group = handle["fusion"]
        arrays = {
            "time_s": group["time_s"][()],
            "quaternion": group["quaternion"][()],
            "quaternion_aligned": group["quaternion_aligned"][()],
            "euler": group["euler_roll_yaw_pitch"][()],
            "acceleration": group["world_acceleration"][()],
            "speed": group["world_speed"][()],
        }
    return pd.DataFrame(
        np.column_stack(tuple(arrays.values())),
        columns=(
            "time_s",
            "quaternion_w",
            "quaternion_x",
            "quaternion_y",
            "quaternion_z",
            "quaternion_aligned_w",
            "quaternion_aligned_x",
            "quaternion_aligned_y",
            "quaternion_aligned_z",
            "roll",
            "yaw",
            "pitch",
            "acc_x",
            "acc_y",
            "acc_z",
            "speed_x",
            "speed_y",
            "speed_z",
        ),
    )


def read_fusion_view_data(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only time, Euler and aligned quaternion arrays needed by the UI.

    Avoiding acceleration, speed and the unaligned quaternion materially lowers
    startup memory for long sessions.
    """
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(source, "r") as handle:
            group = handle["fusion"]
            time_s = group["time_s"][()].astype(np.float64, copy=False)
            euler = group["euler_roll_yaw_pitch"][()].astype(np.float32, copy=False)
            quaternion = group["quaternion_aligned"][()].astype(np.float32, copy=False)
        return time_s, euler, quaternion
    columns = [
        "time_s",
        "roll",
        "yaw",
        "pitch",
        "quaternion_aligned_w",
        "quaternion_aligned_x",
        "quaternion_aligned_y",
        "quaternion_aligned_z",
    ]
    table = pd.read_csv(source, usecols=columns)
    return (
        table["time_s"].to_numpy(dtype=np.float64),
        table[["roll", "yaw", "pitch"]].to_numpy(dtype=np.float32),
        table[
            [
                "quaternion_aligned_w",
                "quaternion_aligned_x",
                "quaternion_aligned_y",
                "quaternion_aligned_z",
            ]
        ].to_numpy(dtype=np.float32),
    )
