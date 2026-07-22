"""Run the complete IMU sensor-fusion pipeline without CLI arguments.

Edit the user configuration below, then run this file from an IDE, execute it
with Python, or use ``run_sensor_fusion.bat`` on Windows. The preferred input
is a start-aligned, resampled nine-axis IMU HDF5 file. Raw ``analogin.dat``
input is also supported through the optional preparation settings.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

# =============================================================================
# User configuration
# =============================================================================

# Preferred input: a start-aligned 100 Hz nine-axis IMU file.
ALIGNED_IMU_FILE = Path(
    r"D:\Jiaqi\tools\9_axies_IMU\prepared_sessions\day5.animal.behavior_corrected\session_01\aligned_imu_100hz.h5"
)

# Fusion output. Missing parent directories are created automatically.
FUSION_OUTPUT_FILE = Path("output/fusion_result.h5")

# Keep False for prepared input. Set True to prepare a raw 16-channel file.
PREPARE_FROM_ANALOG = False

# These settings apply only when PREPARE_FROM_ANALOG is True.
ANALOG_INPUT_FILE = Path(r"D:\data\analogin.dat")
START_OFFSET_SECONDS = 0.0
DURATION_SECONDS = 60.0

# Prepared intermediate file retained for inspection and repeated fusion.
PREPARED_IMU_OUTPUT_FILE = Path("output/aligned_imu_100hz.h5")

# Successful IDE/terminal runs exit without a dialog. Errors still show one.
SHOW_SUCCESS_DIALOG = False
SHOW_ERROR_DIALOG = os.environ.get("IMU_FUSION_NO_DIALOG") != "1"


# =============================================================================
# Implementation
# =============================================================================


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    """Resolve relative configuration paths against the project root."""
    return path if path.is_absolute() else project_root / path


def _ensure_project_environment(project_root: Path) -> None:
    """Restart through the locked project environment when necessary."""
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists() and Path(sys.executable).resolve() == venv_python.resolve():
        return

    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError(
            "The launcher could not find uv. Install uv from "
            "https://docs.astral.sh/uv/ and run this file again."
        )

    print("Starting the locked project environment...", flush=True)
    completed = subprocess.run(
        [uv_executable, "run", "python", str(Path(__file__).resolve())],
        cwd=project_root,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "The project Python process failed with exit code "
            f"{completed.returncode}. See run_sensor_fusion.log."
        )
    # The child process completed the fusion. Stop this bootstrap process
    # without executing the pipeline a second time.
    raise SystemExit(0)


def _write_error_log(project_root: Path, details: str) -> Path:
    """Persist a full traceback where it remains available after exit."""
    log_path = project_root / "run_sensor_fusion.log"
    log_path.write_text(details, encoding="utf-8")
    return log_path


def _show_dialog(title: str, message: str, *, is_error: bool = False) -> None:
    """Show a native result dialog, with a console fallback."""
    if (is_error and not SHOW_ERROR_DIALOG) or (
        not is_error and not SHOW_SUCCESS_DIALOG
    ):
        return
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        if is_error:
            messagebox.showerror(title, message, parent=root)
        else:
            messagebox.showinfo(title, message, parent=root)
        root.destroy()
    except Exception:
        print(message, file=sys.stderr if is_error else sys.stdout, flush=True)


def main() -> Path:
    """Load input, optionally prepare raw data, and run sensor fusion."""
    project_root = Path(__file__).resolve().parent
    _ensure_project_environment(project_root)

    # Import only after environment bootstrapping. This lets any system Python
    # start the entry file even when scientific packages are installed only in
    # the local .venv.
    from imu_fusion.config import FusionConfig
    from imu_fusion.io import read_aligned_imu, write_aligned_imu
    from imu_fusion.pipeline import fuse_aligned_imu
    from imu_fusion.prepare import prepare_aligned_segment

    # Direct-run mode has one source of truth: src/imu_fusion/config.py.
    # In particular, FusionConfig() uses _DEFAULT_MAPPING from that file.
    config = FusionConfig()
    input_started = perf_counter()

    if PREPARE_FROM_ANALOG:
        analog_path = _resolve_project_path(project_root, ANALOG_INPUT_FILE)
        aligned_path = _resolve_project_path(project_root, PREPARED_IMU_OUTPUT_FILE)
        print(f"[1/2] Preparing IMU from analog data: {analog_path}")
        aligned_imu = prepare_aligned_segment(
            analog_path,
            config,
            START_OFFSET_SECONDS,
            DURATION_SECONDS,
        )
        write_aligned_imu(aligned_path, aligned_imu, config.sample_rate_hz)
        print(f"      Aligned IMU saved to: {aligned_path}")
    else:
        aligned_path = _resolve_project_path(project_root, ALIGNED_IMU_FILE)
        print(f"[1/2] Reading aligned IMU: {aligned_path}", flush=True)
        aligned_imu, _ = read_aligned_imu(aligned_path)
    input_seconds = perf_counter() - input_started

    output_path = _resolve_project_path(project_root, FUSION_OUTPUT_FILE)
    print(
        f"[2/2] Running {config.ahrs_algorithm.upper()} sensor fusion...",
        flush=True,
    )
    fusion_started = perf_counter()
    result = fuse_aligned_imu(aligned_imu, config)
    fusion_seconds = perf_counter() - fusion_started
    save_started = perf_counter()
    result.write(output_path)
    save_seconds = perf_counter() - save_started

    duration_s = len(aligned_imu) / config.sample_rate_hz
    output_megabytes = output_path.stat().st_size / (1024 * 1024)
    print("Sensor fusion finished.")
    print(f"  Samples: {len(aligned_imu)}")
    print(f"  Duration: {duration_s:.3f} s")
    print(f"  Result: {output_path}")
    print(f"  Result size: {output_megabytes:.2f} MiB")
    print(
        "  Timing: "
        f"input {input_seconds:.3f} s | "
        f"fusion {fusion_seconds:.3f} s | "
        f"HDF5 save {save_seconds:.3f} s",
        flush=True,
    )
    stale_error_log = project_root / "run_sensor_fusion.log"
    if stale_error_log.exists():
        stale_error_log.unlink()
    return output_path


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent
    try:
        result_path = main()
    except SystemExit:
        raise
    except Exception as error:
        error_details = traceback.format_exc()
        log_file = _write_error_log(root_path, error_details)
        print(error_details, file=sys.stderr, flush=True)
        _show_dialog(
            "IMU Sensor Fusion - Error",
            f"Sensor fusion failed:\n{error}\n\n"
            f"Error details were saved to:\n{log_file}",
            is_error=True,
        )
        raise SystemExit(1) from None
    else:
        _show_dialog(
            "IMU Sensor Fusion - Finished",
            f"Sensor fusion completed successfully.\n\nResult:\n{result_path}",
        )
