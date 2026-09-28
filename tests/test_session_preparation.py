from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from imu_fusion.config import FusionConfig
from imu_fusion.io import read_aligned_imu
from imu_fusion.session_preparation import (
    parse_day_id,
    parse_recording_start,
    prepare_behavior_day,
    read_session_labels,
    split_behavior_sessions,
)


def test_split_behavior_sessions_at_large_gaps() -> None:
    timestamps = np.array([1.0, 1.1, 1.2, 80.0, 80.1])
    segments = split_behavior_sessions(timestamps, gap_seconds=60.0)
    assert [(item.start_s, item.end_s) for item in segments] == [
        (1.0, 1.2),
        (80.0, 80.1),
    ]


def test_parse_recording_folder_with_suffix() -> None:
    value = parse_recording_start("2_20240918_120411.179_extra")
    assert value.isoformat(timespec="milliseconds") == "2024-09-18T12:04:11.179"


def test_parse_day_id_from_split_directory_names() -> None:
    path = Path(r"I:\data\FieldRat\2024\F5\Merged\day10\121_day10\behavior.mat")
    assert parse_day_id(path) == "F5D10"


def test_prepare_continuous_day_analog(tmp_path: Path) -> None:
    behavior_path = tmp_path / "day.animal.behavior_corrected.mat"
    timestamps = np.arange(0.2, 0.71, 0.01)
    savemat(behavior_path, {"behavior": {"timestamps_corrected": timestamps}})

    raw = np.zeros((200, 16), dtype="<i2")
    raw[:, 1:10] = np.arange(200, dtype=np.int16)[:, None]
    raw.tofile(tmp_path / "analogin.dat")

    config = FusionConfig(
        sample_rate_hz=100.0,
        raw_sample_rate_hz=100.0,
        calibrate=False,
    )
    manifest = prepare_behavior_day(
        behavior_path,
        tmp_path / "prepared",
        config,
    )
    output = Path(manifest.loc[0, "output_file"])
    assert output.is_file()
    assert output.suffix == ".h5"
    values, time_s = read_aligned_imu(output)
    assert values.shape == (50, 9)
    assert time_s is not None
    assert manifest.loc[0, "analog_start_offset_s"] == 0.2
    assert manifest.loc[0, "status"] == "prepared"


def test_session_labels_select_numbered_recording_folders(tmp_path: Path) -> None:
    day_path = tmp_path / "F5Day2"
    day_path.mkdir()
    behavior_path = day_path / "day2.animal.behavior_corrected.mat"
    timestamps = np.array([0.0, 0.1, 80.0, 80.1])
    savemat(behavior_path, {"behavior": {"timestamps_corrected": timestamps}})

    raw = np.zeros((50, 16), dtype="<i2")
    for folder_index in range(1, 6):
        folder = day_path / f"{folder_index}_20240701_12000{folder_index}.000"
        folder.mkdir()
        raw.tofile(folder / "analogin.dat")
        pd.DataFrame({"timestamp": [f"2024-07-01 12:00:0{folder_index}"]}).to_csv(
            folder / "timestamp_corrected.csv", index=False
        )

    label_path = tmp_path / "sessions.txt"
    label_path.write_text("F5Day2_2: outdoor\nF5Day2_5: indoor\n", encoding="utf-8")
    labels = read_session_labels(label_path)
    config = FusionConfig(
        sample_rate_hz=100.0,
        raw_sample_rate_hz=100.0,
        calibrate=False,
    )

    manifest = prepare_behavior_day(
        behavior_path,
        tmp_path / "prepared",
        config,
        session_labels=labels,
    )

    assert manifest["source_folder_index"].tolist() == [2, 5]
    assert manifest["condition"].tolist() == ["outdoor", "indoor"]
    assert manifest["standard_session_name"].tolist() == [
        "F5D2_outdoor",
        "F5D2_indoor",
    ]
    assert "2_20240701" in manifest.loc[0, "analog_file"]
    assert "5_20240701" in manifest.loc[1, "analog_file"]
