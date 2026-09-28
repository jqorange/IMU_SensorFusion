"""Efficient loading and windowing for synchronized LFP and spike rasters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class NeuralWindow:
    """One centered LFP and raster window ready for display."""

    lfp_time_s: np.ndarray
    lfp: np.ndarray
    raster: np.ndarray


@dataclass
class NeuralData:
    """Session neural data held in a form suitable for responsive slicing."""

    spike_times: tuple[np.ndarray, ...]
    lfp: np.ndarray
    lfp_sample_rate_hz: float
    session_start_seconds: float
    lfp_anchor_seconds: float
    window_seconds: float = 5.0
    raster_bin_seconds: float = 0.05
    raster_smoothing_seconds: float = 0.30

    def centered_window(self, elapsed_seconds: float) -> NeuralWindow:
        """Build a centered window and sort cells by smoothed peak time."""
        absolute_center = self.session_start_seconds + float(elapsed_seconds)
        half_window = self.window_seconds / 2.0
        absolute_start = absolute_center - half_window
        absolute_stop = absolute_center + half_window

        lfp_start = max(
            0,
            int(
                np.floor(
                    (absolute_start - self.lfp_anchor_seconds) * self.lfp_sample_rate_hz
                )
            ),
        )
        lfp_stop = min(
            len(self.lfp),
            int(
                np.ceil(
                    (absolute_stop - self.lfp_anchor_seconds) * self.lfp_sample_rate_hz
                )
            ),
        )
        if lfp_stop > lfp_start:
            lfp_values = self.lfp[lfp_start:lfp_stop].astype(np.float32, copy=False)
            lfp_time = (
                np.arange(lfp_start, lfp_stop, dtype=np.float64)
                / self.lfp_sample_rate_hz
                + self.lfp_anchor_seconds
                - absolute_center
            )
            lfp_values = lfp_values - np.nanmedian(lfp_values)
        else:
            lfp_time = np.empty(0, dtype=np.float64)
            lfp_values = np.empty(0, dtype=np.float32)

        bin_count = max(1, int(round(self.window_seconds / self.raster_bin_seconds)))
        edges = absolute_start + np.arange(bin_count + 1) * self.raster_bin_seconds
        edges[-1] = absolute_stop
        raster = np.zeros((len(self.spike_times), bin_count), dtype=np.uint16)
        for cell_index, spikes in enumerate(self.spike_times):
            left = int(np.searchsorted(spikes, absolute_start, side="left"))
            right = int(np.searchsorted(spikes, absolute_stop, side="left"))
            if right > left:
                raster[cell_index] = np.histogram(spikes[left:right], bins=edges)[0]

        if raster.shape[0] > 1:
            sigma_bins = self.raster_smoothing_seconds / self.raster_bin_seconds
            smoothed = gaussian_filter1d(
                raster.astype(np.float32),
                sigma=sigma_bins,
                axis=1,
                mode="nearest",
            )
            order = np.argsort(np.argmax(smoothed, axis=1), kind="stable")
            raster = raster[order]

        return NeuralWindow(lfp_time_s=lfp_time, lfp=lfp_values, raster=raster)


def load_timestamp_origin(
    mat_path: str | Path,
    key: str | None,
    segment_index: int = 1,
    gap_seconds: float = 60.0,
) -> float:
    """Read the unnormalized first numeric timestamp from a MAT field."""
    if not key:
        raise ValueError(
            "A timestamp key is required to infer the neural session start time."
        )
    value: object = loadmat(mat_path, simplify_cells=True)
    for component in key.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(f"MAT timestamp key {key!r} was not found in {mat_path}.")
        value = value[component]
    timestamps = np.asarray(value)
    if not np.issubdtype(timestamps.dtype, np.number) or timestamps.size == 0:
        raise ValueError(
            "Neural alignment requires a non-empty numeric timestamp array."
        )
    values = timestamps.astype(np.float64).ravel()
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0.0):
        raise ValueError("Video timestamps must be finite and monotonic.")
    breaks = np.flatnonzero(np.diff(values) > gap_seconds)
    starts = np.concatenate(([0], breaks + 1))
    if segment_index < 1 or segment_index > len(starts):
        raise ValueError(
            f"Timestamp array has {len(starts)} segment(s); "
            f"segment {segment_index} was requested."
        )
    origin = float(values[int(starts[segment_index - 1])])
    if not np.isfinite(origin):
        raise ValueError("The first video timestamp is not finite.")
    if abs(origin) > 1e5:
        raise ValueError(
            "MATLAB datenums cannot be aligned directly to neural recording seconds."
        )
    return origin


def _find_single(directory: Path, patterns: tuple[str, ...], label: str) -> Path:
    """Find one recording file using ordered filename patterns."""
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No {label} file was found in {directory}.")


def _read_xml_metadata(xml_path: Path) -> tuple[int, float]:
    """Read channel count and LFP sampling rate from a Neuroscope XML file."""
    root = ElementTree.parse(xml_path).getroot()
    channel_element = root.find(".//acquisitionSystem/nChannels")
    rate_element = root.find(".//fieldPotentials/lfpSamplingRate")
    if channel_element is None or channel_element.text is None:
        raise ValueError(f"XML has no acquisitionSystem/nChannels: {xml_path}")
    if rate_element is None or rate_element.text is None:
        raise ValueError(f"XML has no fieldPotentials/lfpSamplingRate: {xml_path}")
    return int(channel_element.text), float(rate_element.text)


def _load_spike_times(cell_metrics_path: Path) -> tuple[np.ndarray, ...]:
    """Load CellExplorer spike-time arrays from a cell metrics MAT file."""
    content = loadmat(cell_metrics_path, squeeze_me=True, struct_as_record=False)
    if "cell_metrics" not in content:
        raise KeyError(f"{cell_metrics_path} has no cell_metrics variable.")
    metrics = content["cell_metrics"]
    if not hasattr(metrics, "spikes") or not hasattr(metrics.spikes, "times"):
        raise KeyError(f"{cell_metrics_path} has no cell_metrics.spikes.times field.")
    raw_times = metrics.spikes.times
    if isinstance(raw_times, np.ndarray) and raw_times.dtype == object:
        cells = raw_times.ravel().tolist()
    elif isinstance(raw_times, (list, tuple)):
        cells = list(raw_times)
    else:
        cells = [raw_times]
    return tuple(np.sort(np.asarray(cell, dtype=np.float64).ravel()) for cell in cells)


def load_neural_data(
    recording_directory: str | Path,
    *,
    session_start_seconds: float,
    session_duration_seconds: float,
    lfp_anchor_seconds: float | None = None,
    lfp_channel: int = 0,
    minimum_mean_firing_rate_hz: float = 0.2,
    window_seconds: float = 5.0,
    raster_bin_seconds: float = 0.05,
) -> NeuralData:
    """Load one selected LFP channel and session-filtered spike trains."""
    directory = Path(recording_directory)
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Neural recording directory does not exist: {directory}"
        )
    basename = directory.name
    xml_path = _find_single(directory, (f"{basename}.xml", "*.xml"), "XML")
    lfp_path = _find_single(
        directory,
        (f"{basename}.lfp", "*.lfp", f"{basename}.eeg", "*.eeg"),
        "LFP",
    )
    cell_metrics_path = _find_single(
        directory,
        ("*cell_metrics.cellinfo.mat",),
        "cell metrics",
    )

    channel_count, sample_rate_hz = _read_xml_metadata(xml_path)
    if not 0 <= lfp_channel < channel_count:
        raise ValueError(
            f"LFP channel {lfp_channel} is outside [0, {channel_count - 1}]."
        )
    raw_lfp = np.memmap(lfp_path, dtype=np.int16, mode="r")
    if raw_lfp.size % channel_count:
        raise ValueError(
            f"LFP sample count is not divisible by {channel_count} channels: {lfp_path}"
        )
    source_anchor_seconds = float(
        session_start_seconds if lfp_anchor_seconds is None else lfp_anchor_seconds
    )
    total_lfp_samples = raw_lfp.size // channel_count
    half_window = float(window_seconds) / 2.0
    first_lfp_sample = max(
        0,
        int(
            np.floor(
                (session_start_seconds - half_window - source_anchor_seconds)
                * sample_rate_hz
            )
        ),
    )
    last_lfp_sample = min(
        total_lfp_samples,
        int(
            np.ceil(
                (
                    session_start_seconds
                    + session_duration_seconds
                    + half_window
                    - source_anchor_seconds
                )
                * sample_rate_hz
            )
        ),
    )
    if last_lfp_sample <= first_lfp_sample:
        raise RuntimeError(
            "The configured neural session does not overlap the LFP recording."
        )
    lfp = np.asarray(
        raw_lfp.reshape(-1, channel_count)[
            first_lfp_sample:last_lfp_sample, lfp_channel
        ],
        dtype=np.float32,
    ).copy()
    del raw_lfp

    spike_times = _load_spike_times(cell_metrics_path)
    session_stop_seconds = session_start_seconds + session_duration_seconds
    valid_spikes: list[np.ndarray] = []
    for spikes in spike_times:
        start = np.searchsorted(spikes, session_start_seconds, side="left")
        stop = np.searchsorted(spikes, session_stop_seconds, side="left")
        mean_rate = (stop - start) / max(session_duration_seconds, 1e-6)
        if mean_rate > minimum_mean_firing_rate_hz:
            valid_spikes.append(spikes)
    if not valid_spikes:
        raise RuntimeError(
            "No neurons passed the configured session mean firing-rate threshold."
        )

    return NeuralData(
        spike_times=tuple(valid_spikes),
        lfp=lfp,
        lfp_sample_rate_hz=sample_rate_hz,
        session_start_seconds=float(session_start_seconds),
        lfp_anchor_seconds=(source_anchor_seconds + first_lfp_sample / sample_rate_hz),
        window_seconds=float(window_seconds),
        raster_bin_seconds=float(raster_bin_seconds),
    )
