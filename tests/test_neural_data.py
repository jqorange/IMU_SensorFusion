"""Tests for synchronized LFP and raster preparation."""

from __future__ import annotations

import numpy as np
from scipy.io import savemat

from imu_fusion.neural_data import load_neural_data, load_timestamp_origin


def test_load_timestamp_origin(tmp_path) -> None:
    mat_path = tmp_path / "behavior.mat"
    savemat(
        mat_path,
        {"behavior": {"timestamps": np.array([12.5, 12.52, 100.0, 100.02])}},
    )

    origin = load_timestamp_origin(
        mat_path,
        "behavior.timestamps",
        segment_index=2,
        gap_seconds=60.0,
    )

    assert origin == 100.0


def test_load_neural_data_and_centered_window(tmp_path) -> None:
    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "recording.xml").write_text(
        """
        <parameters>
          <acquisitionSystem><nChannels>2</nChannels></acquisitionSystem>
          <fieldPotentials><lfpSamplingRate>100</lfpSamplingRate></fieldPotentials>
        </parameters>
        """,
        encoding="utf-8",
    )
    sample_count = 600
    interleaved = np.column_stack(
        (
            np.arange(sample_count, dtype=np.int16),
            -np.arange(sample_count, dtype=np.int16),
        )
    )
    interleaved.tofile(recording / "recording.lfp")
    spike_cells = np.empty(2, dtype=object)
    spike_cells[0] = np.array([10.1, 10.5, 11.0, 12.0, 13.0])
    spike_cells[1] = np.array([10.2, 11.2, 12.2, 13.2, 13.8])
    savemat(
        recording / "recording.cell_metrics.cellinfo.mat",
        {"cell_metrics": {"spikes": {"times": spike_cells}}},
    )

    neural = load_neural_data(
        recording,
        session_start_seconds=10.0,
        session_duration_seconds=4.0,
        lfp_anchor_seconds=10.0,
        lfp_channel=1,
        minimum_mean_firing_rate_hz=0.2,
        window_seconds=2.0,
        raster_bin_seconds=0.1,
    )
    window = neural.centered_window(2.0)

    assert neural.lfp_sample_rate_hz == 100.0
    assert len(neural.spike_times) == 2
    assert window.lfp.shape == (200,)
    assert window.raster.shape == (2, 20)
    assert int(window.raster.sum()) == 4
