"""Tests for global named-session path resolution."""

from pathlib import Path

import numpy as np
from scipy.io import savemat

from imu_fusion import sessions


def test_resolve_named_session(tmp_path: Path, monkeypatch: object) -> None:
    session_info = tmp_path / "Session infor.txt"
    session_info.write_text(
        "F5Day3_2: indoor\nF5Day3_4: outdoor\n",
        encoding="utf-8",
    )
    field_root = tmp_path / "FieldRat"
    recording = field_root / "F5" / "day3" / "day3_outdoorsmall_F5F6"
    recording.mkdir(parents=True)
    video = recording / "cam_test.avi"
    video.touch()
    tracking = recording / "cam_testDLC_model.csv"
    tracking.touch()

    neural = field_root / "F5" / "Merged" / "day3" / "121_day3"
    neural.mkdir(parents=True)
    timestamp_mat = neural / "121_day3.animal.behavior_all.mat"
    timestamp_mat.touch()
    savemat(
        neural / "121_day3.MergePoints.events.mat",
        {
            "MergePoints": {
                "timestamps": np.array(
                    [[0.0, 300.0], [300.1, 800.0]],
                    dtype=np.float64,
                )
            }
        },
    )

    prepared = (
        tmp_path
        / "project"
        / "prepared_sessions"
        / "121_day3.animal.behavior_all"
        / "session_02_recording"
        / "aligned_imu_100hz.h5"
    )
    prepared.parent.mkdir(parents=True)
    prepared.touch()

    behavior_root = tmp_path / "states"
    pose_root = tmp_path / "poses"
    event_root = tmp_path / "event_peth"
    event_name = "walk_to_local_search"
    for session_name, segment_start in (
        ("F5D3_indoor", 120.5),
        ("F5D3_outdoor", 430.25),
    ):
        meta_path = (
            event_root
            / event_name
            / session_name
            / f"{event_name}__meta.json"
        )
        meta_path.parent.mkdir(parents=True)
        meta_path.write_text(
            f'{{"segment_start_sec": {segment_start}}}',
            encoding="utf-8",
        )
    monkeypatch.setattr(sessions, "SESSION_INFO_FILE", session_info)
    monkeypatch.setattr(sessions, "BEHAVIOR_STATES_ROOT", behavior_root)
    monkeypatch.setattr(sessions, "POSE_FEATURES_ROOT", pose_root)
    monkeypatch.setattr(sessions, "NEURAL_EVENT_PETH_ROOT", event_root)
    monkeypatch.setattr(sessions, "VIDEO_FILE", video)
    monkeypatch.setattr(sessions, "TIMESTAMP_MAT_FILE", timestamp_mat)

    resolved = sessions.resolve_session("F5D3_outdoor", tmp_path / "project")

    assert resolved.name == "F5D3_outdoor"
    assert resolved.recording_number == 4
    assert resolved.timestamp_segment_index == 2
    assert resolved.aligned_imu_file == prepared
    assert resolved.video_file == video
    assert resolved.timestamp_mat_file == timestamp_mat
    assert resolved.dlc_tracking_file == tracking
    assert resolved.neural_recording_directory == neural
    assert resolved.neural_session_start_seconds == 430.25
    assert resolved.lfp_anchor_seconds == 0.0
