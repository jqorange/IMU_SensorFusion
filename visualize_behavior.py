"""Launch the synchronized IMU, video, pose, and behavior viewer.

Edit the user configuration below, then run this file directly from an IDE or
use ``visualize_behavior.bat`` on Windows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# =============================================================================
# User configuration
# =============================================================================

TIMESTAMP_KEY: str | None = "behavior.timestamps_corrected"
TIMESTAMP_GAP_SECONDS = 60.0
INITIAL_PLAYBACK_FPS = 40.0
CHOOSE_FILES_ON_START = False

VIDEO_TRACKING_BODYPART = "bodyCenter1"
VIDEO_TRACKING_MIN_LIKELIHOOD = 0.5
BEHAVIOR_SAMPLE_RATE_HZ = 50.0

LFP_CHANNEL = 0
NEURAL_WINDOW_SECONDS = 5.0
RASTER_BIN_SECONDS = 0.05
MINIMUM_MEAN_FIRING_RATE_HZ = 0.2
NEURAL_UPDATE_INTERVAL_SECONDS = 0.05

VIDEO_CACHE_MEGABYTES = 512
VIDEO_PREFETCH_FRAMES = 24
VIDEO_DISPLAY_MAX_SIZE = (1000, 620)
PLOT_WINDOW_SECONDS = 2.0

CARTOON_CAMERA_AZIMUTH_DEGREES = -37.5
# These fixed values match D:\Jiaqi\tools\IMU_SensorFusion.
CARTOON_CAMERA_ELEVATION_DEGREES = 30.0
CARTOON_WORLD_YAW_ALIGNMENT_DEGREES = -120
VIDEO_COMPASS_ROTATION_DEGREES = -90.0


# =============================================================================
# Launcher implementation
# =============================================================================


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    """Resolve relative configuration paths against the project root."""
    return path if path.is_absolute() else project_root / path


def _ensure_viewer_environment(project_root: Path) -> None:
    """Restart once through uv with optional viewer dependencies."""
    if os.environ.get("IMU_BEHAVIOR_VIEWER_BOOTSTRAPPED") == "1":
        return
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError(
            "The launcher could not find uv. Install uv from "
            "https://docs.astral.sh/uv/ and run this file again."
        )
    environment = os.environ.copy()
    environment["IMU_BEHAVIOR_VIEWER_BOOTSTRAPPED"] = "1"
    print("Starting the locked behavior-viewer environment...", flush=True)
    completed = subprocess.run(
        [
            uv_executable,
            "run",
            "--extra",
            "viewer",
            "python",
            str(Path(__file__).resolve()),
        ],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "The behavior viewer exited with code "
            f"{completed.returncode}. See visualize_behavior.log."
        )
    raise SystemExit(0)


def _choose_file(
    title: str,
    initial_path: Path,
    file_filter: str,
) -> Path:
    """Select one required file or keep the configured path."""
    if not CHOOSE_FILES_ON_START:
        return initial_path
    from PySide6 import QtWidgets

    selected, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        title,
        str(initial_path.parent),
        file_filter,
    )
    if not selected:
        raise RuntimeError(f"{title} selection was cancelled.")
    return Path(selected)


def _show_error(message: str) -> None:
    """Show a native error dialog with a standard-error fallback."""
    try:
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        QtWidgets.QMessageBox.critical(None, "Behavior Viewer - Error", message)
        app.processEvents()
    except Exception:
        print(message, file=sys.stderr, flush=True)


def main() -> None:
    """Resolve all inputs and launch the extended viewer."""
    project_root = Path(__file__).resolve().parent
    _ensure_viewer_environment(project_root)

    from PySide6 import QtCore, QtWidgets

    from imu_fusion.behavior_viewer import BehaviorFusionViewer
    from imu_fusion.config import ACTIVE_SESSION
    from imu_fusion.sessions import find_batch_imu_result, resolve_session

    session = resolve_session(ACTIVE_SESSION, project_root)
    print(f"Active session: {session.name}", flush=True)

    configured = {
        "IMU result": (
            find_batch_imu_result(session.name),
            "IMU result files (*.csv *.h5 *.hdf5)",
        ),
        "Video": (session.video_file, "Video files (*.mp4 *.avi *.mov *.mkv)"),
        "Timestamp MAT": (session.timestamp_mat_file, "MATLAB files (*.mat)"),
        "Video DLC tracking": (
            session.dlc_tracking_file,
            "CSV files (*.csv)",
        ),
        "Behavior states": (
            session.behavior_states_file,
            "CSV files (*.csv)",
        ),
        "Pose features": (session.pose_features_file, "CSV files (*.csv)"),
    }
    selected: dict[str, Path] = {}
    for label, (configured_path, file_filter) in configured.items():
        resolved = _resolve_project_path(project_root, configured_path)
        selected[label] = _choose_file(f"Select {label}", resolved, file_filter)
        if not selected[label].is_file():
            raise FileNotFoundError(f"{label} file does not exist: {selected[label]}")

    neural_directory = None
    if session.neural_recording_directory is not None:
        candidate = session.neural_recording_directory
        if (
            candidate.is_dir()
            and session.neural_session_start_seconds is not None
            and session.lfp_anchor_seconds is not None
        ):
            neural_directory = candidate
            print(
                "Neural alignment: "
                f"session start {session.neural_session_start_seconds:.6f} s | "
                f"LFP anchor {session.lfp_anchor_seconds:.6f} s",
                flush=True,
            )
        elif candidate.is_dir():
            print(
                "Neural panels are disabled because matching event-PETH "
                f"session metadata was not found for {session.name}.",
                flush=True,
            )
        else:
            print(
                "Neural recording directory is unavailable; LFP and raster "
                f"panels will remain disabled: {candidate}",
                flush=True,
            )

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BehaviorFusionViewer(
        selected["IMU result"],
        selected["Video"],
        selected["Timestamp MAT"],
        TIMESTAMP_KEY,
        timestamp_segment_index=session.timestamp_segment_index,
        timestamp_gap_seconds=TIMESTAMP_GAP_SECONDS,
        initial_playback_fps=INITIAL_PLAYBACK_FPS,
        behavior_states_path=selected["Behavior states"],
        pose_features_path=selected["Pose features"],
        behavior_sample_rate_hz=BEHAVIOR_SAMPLE_RATE_HZ,
        neural_recording_directory=neural_directory,
        neural_session_start_seconds=session.neural_session_start_seconds,
        lfp_anchor_seconds=session.lfp_anchor_seconds,
        lfp_channel=LFP_CHANNEL,
        neural_window_seconds=NEURAL_WINDOW_SECONDS,
        raster_bin_seconds=RASTER_BIN_SECONDS,
        minimum_mean_firing_rate_hz=MINIMUM_MEAN_FIRING_RATE_HZ,
        neural_update_interval_seconds=NEURAL_UPDATE_INTERVAL_SECONDS,
        tracking_path=selected["Video DLC tracking"],
        tracking_bodypart=VIDEO_TRACKING_BODYPART,
        tracking_min_likelihood=VIDEO_TRACKING_MIN_LIKELIHOOD,
        video_cache_megabytes=VIDEO_CACHE_MEGABYTES,
        video_prefetch_frames=VIDEO_PREFETCH_FRAMES,
        video_display_size=VIDEO_DISPLAY_MAX_SIZE,
        plot_window_seconds=PLOT_WINDOW_SECONDS,
        cartoon_camera_azimuth_degrees=CARTOON_CAMERA_AZIMUTH_DEGREES,
        cartoon_camera_elevation_degrees=CARTOON_CAMERA_ELEVATION_DEGREES,
        cartoon_world_yaw_alignment_degrees=CARTOON_WORLD_YAW_ALIGNMENT_DEGREES,
        video_compass_rotation_degrees=VIDEO_COMPASS_ROTATION_DEGREES,
        show_video_compass=False,
    )
    window.show()
    test_close_ms = int(os.environ.get("IMU_VIEWER_TEST_CLOSE_MS", "0"))
    if test_close_ms > 0:
        QtCore.QTimer.singleShot(test_close_ms, app.quit)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent
    try:
        main()
    except SystemExit:
        raise
    except Exception as error:
        details = traceback.format_exc()
        log_path = root_path / "visualize_behavior.log"
        log_path.write_text(details, encoding="utf-8")
        print(details, file=sys.stderr, flush=True)
        _show_error(
            f"The behavior viewer could not start:\n{error}\n\n"
            f"Details were saved to:\n{log_path}"
        )
        raise SystemExit(1) from None
