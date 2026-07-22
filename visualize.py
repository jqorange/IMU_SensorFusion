"""Launch the synchronized video and IMU fusion viewer without CLI arguments.

Edit the user configuration below, then run this file from an IDE or use
``visualize.bat`` on Windows.
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

FUSION_RESULT_FILE = Path(r"D:\Jiaqi\tools\9_axies_IMU\output\fusion_result.h5")
VIDEO_FILE = Path(
    r"D:\Jiaqi\Datasets\Dataset\Jiaqi_data\F3day5\F3_out_medium_0701\cam742024-07-01T18_36_58.avi"
)
VIDEO_TIMESTAMP_MAT_FILE = Path(
    r"D:\Jiaqi\Datasets\Dataset\Jiaqi_data\F3day5\day5.animal.behavior_corrected.mat"
)
# Set a known MAT key, or use None to select the best timestamp array.
TIMESTAMP_KEY: str | None = "behavior.timestamps_corrected"

# Set True to select the three input files at startup.
CHOOSE_FILES_ON_START = False

# Large-video performance settings. The cache stores resized RGB frames only.
VIDEO_CACHE_MEGABYTES = 512
VIDEO_PREFETCH_FRAMES = 24
VIDEO_DISPLAY_MAX_SIZE = (1000, 620)
# Total duration of each scrolling Euler plot, centered on the current frame.
PLOT_WINDOW_SECONDS = 2.0

# Procedural 3D mouse camera and world-yaw alignment.
CARTOON_CAMERA_AZIMUTH_DEGREES = -37.5
CARTOON_CAMERA_ELEVATION_DEGREES = 30.0
CARTOON_WORLD_YAW_ALIGNMENT_DEGREES = -10


# =============================================================================
# Implementation
# =============================================================================


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    """Resolve relative configuration paths against the project root."""
    return path if path.is_absolute() else project_root / path


def _ensure_viewer_environment(project_root: Path) -> None:
    """Restart once through uv so optional viewer dependencies are available."""
    if os.environ.get("IMU_VIEWER_BOOTSTRAPPED") == "1":
        return
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError(
            "The launcher could not find uv. Install uv from "
            "https://docs.astral.sh/uv/ and run this file again."
        )
    child_environment = os.environ.copy()
    child_environment["IMU_VIEWER_BOOTSTRAPPED"] = "1"
    print("Starting the locked video-viewer environment...", flush=True)
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
        env=child_environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "The video-viewer process failed with exit code "
            f"{completed.returncode}. See visualize.log."
        )
    raise SystemExit(0)


def _choose_files(
    result_path: Path,
    video_path: Path,
    timestamp_path: Path,
) -> tuple[Path, Path, Path]:
    """Optionally select the three viewer inputs with native dialogs."""
    if not CHOOSE_FILES_ON_START:
        return result_path, video_path, timestamp_path

    from PySide6 import QtWidgets

    result_name, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select fusion result",
        str(result_path.parent),
        "Fusion files (*.h5 *.hdf5 *.csv)",
    )
    if not result_name:
        raise RuntimeError("Fusion result selection was cancelled.")
    video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select video",
        str(video_path.parent),
        "Video files (*.mp4 *.avi *.mov *.mkv)",
    )
    if not video_name:
        raise RuntimeError("Video selection was cancelled.")
    timestamp_name, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select video timestamp MAT file",
        str(timestamp_path.parent),
        "MATLAB files (*.mat)",
    )
    if not timestamp_name:
        raise RuntimeError("Timestamp selection was cancelled.")
    return Path(result_name), Path(video_name), Path(timestamp_name)


def _show_error(message: str) -> None:
    """Show a native error box, falling back to standard error."""
    try:
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        QtWidgets.QMessageBox.critical(None, "IMU Viewer - Error", message)
        app.processEvents()
    except Exception:
        print(message, file=sys.stderr, flush=True)


def main() -> None:
    """Resolve configured inputs and launch the synchronized desktop viewer."""
    project_root = Path(__file__).resolve().parent
    _ensure_viewer_environment(project_root)

    from PySide6 import QtCore, QtWidgets

    from imu_fusion.viewer import FusionViewer

    result_path = _resolve_project_path(project_root, FUSION_RESULT_FILE)
    video_path = _resolve_project_path(project_root, VIDEO_FILE)
    timestamp_path = _resolve_project_path(project_root, VIDEO_TIMESTAMP_MAT_FILE)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    result_path, video_path, timestamp_path = _choose_files(
        result_path,
        video_path,
        timestamp_path,
    )
    for label, path in (
        ("Fusion result", result_path),
        ("Video", video_path),
        ("Timestamp MAT", timestamp_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")

    window = FusionViewer(
        result_path,
        video_path,
        timestamp_path,
        TIMESTAMP_KEY,
        video_cache_megabytes=VIDEO_CACHE_MEGABYTES,
        video_prefetch_frames=VIDEO_PREFETCH_FRAMES,
        video_display_size=VIDEO_DISPLAY_MAX_SIZE,
        plot_window_seconds=PLOT_WINDOW_SECONDS,
        cartoon_camera_azimuth_degrees=CARTOON_CAMERA_AZIMUTH_DEGREES,
        cartoon_camera_elevation_degrees=CARTOON_CAMERA_ELEVATION_DEGREES,
        cartoon_world_yaw_alignment_degrees=CARTOON_WORLD_YAW_ALIGNMENT_DEGREES,
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
        log_path = root_path / "visualize.log"
        log_path.write_text(details, encoding="utf-8")
        print(details, file=sys.stderr, flush=True)
        _show_error(
            f"The viewer could not start:\n{error}\n\n"
            f"Details were saved to:\n{log_path}"
        )
        raise SystemExit(1) from None
