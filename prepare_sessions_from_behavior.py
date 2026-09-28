"""Build sensor-fusion inputs from behavior MAT and analog recordings.

Edit the user configuration below and run this file directly. No command-line
arguments are required.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =============================================================================
# User configuration
# =============================================================================

# Each path may be a day directory, a parent directory, or one behavior MAT.
DATA_ROOTS = [
    Path(
        r"I:\data\FieldRat\2024\F5\Merged\day10\121_day10\121_day10.animal.behavior_all.mat"
    ),
]

OUTPUT_ROOT = Path("prepared_sessions")

# Maps each day's numbered recording folder to its experimental condition.
SESSION_INFO_FILE = Path("config/session_labels.txt")

# Prepare only the session currently used by the project entry points.
INCLUDED_STANDARD_SESSIONS = ("F5D10_outdoor",)

# A larger timestamp gap starts a new behavior session.
SESSION_GAP_SECONDS = 60.0

# Search analogin.dat first. Reverse the order for datasets that require analogin2.
ANALOG_FILE_PRIORITY = ("analogin.dat", "analogin2.dat")

# Set True to rebuild and overwrite existing prepared HDF5 files.
OVERWRITE_EXISTING = False

# Concurrent jobs. Use 2 for HDD storage and 4-8 for SSD/NVMe storage.
MAX_WORKERS = 4


# =============================================================================
# Implementation
# =============================================================================


def _ensure_environment(project_root: Path) -> None:
    """Restart once through the locked uv environment."""
    if os.environ.get("IMU_PREPARE_BOOTSTRAPPED") == "1":
        return
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError("uv was not found. Install uv and run this file again.")
    environment = os.environ.copy()
    environment["IMU_PREPARE_BOOTSTRAPPED"] = "1"
    completed = subprocess.run(
        [uv_executable, "run", "python", str(Path(__file__).resolve())],
        cwd=project_root,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Preparation process exited with {completed.returncode}.")
    raise SystemExit(0)


def _absolute(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _show_message(title: str, message: str, *, error: bool = False) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        function = messagebox.showerror if error else messagebox.showinfo
        function(title, message, parent=root)
        root.destroy()
    except Exception:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)


def main() -> Path:
    """Discover behavior files and prepare all of their matched sessions."""
    project_root = Path(__file__).resolve().parent
    _ensure_environment(project_root)

    import pandas as pd

    from imu_fusion.config import FusionConfig
    from imu_fusion.session_preparation import (
        find_behavior_mats,
        prepare_behavior_day,
        read_session_labels,
        write_table_h5,
    )

    # Direct-run mode uses defaults from src/imu_fusion/config.py so session
    # preparation and sensor fusion cannot silently use different YAML values.
    config = FusionConfig()
    output_root = _absolute(project_root, OUTPUT_ROOT)
    session_info_path = _absolute(project_root, SESSION_INFO_FILE)
    session_labels = read_session_labels(session_info_path)
    behavior_files = sorted(
        {
            behavior
            for root in DATA_ROOTS
            for behavior in find_behavior_mats(_absolute(project_root, root))
        }
    )
    if not behavior_files:
        raise FileNotFoundError("No corrected behavior MAT files were found.")

    manifests: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    print(f"Found {len(behavior_files)} behavior MAT file(s).", flush=True)

    def process_behavior(index: int, behavior_file: Path) -> tuple[Path, pd.DataFrame]:
        print(f"[{index}/{len(behavior_files)}] {behavior_file}", flush=True)
        per_day_workers = MAX_WORKERS if len(behavior_files) == 1 else 1
        manifest = prepare_behavior_day(
            behavior_file,
            output_root,
            config,
            gap_seconds=SESSION_GAP_SECONDS,
            analog_priority=ANALOG_FILE_PRIORITY,
            overwrite=OVERWRITE_EXISTING,
            max_workers=per_day_workers,
            session_labels=session_labels,
            included_standard_sessions=INCLUDED_STANDARD_SESSIONS,
        )
        return behavior_file, manifest

    outer_workers = max(1, min(MAX_WORKERS, len(behavior_files)))
    with ThreadPoolExecutor(max_workers=outer_workers) as executor:
        futures = {
            executor.submit(process_behavior, index, behavior_file): behavior_file
            for index, behavior_file in enumerate(behavior_files, start=1)
        }
        for future in as_completed(futures):
            behavior_file = futures[future]
            try:
                _, manifest = future.result()
                manifests.append(manifest)
                for output in manifest["output_file"]:
                    print(f"    {output}", flush=True)
            except Exception as error:
                failures.append(
                    {"behavior_mat": str(behavior_file), "error": str(error)}
                )
                print(f"    ERROR: {error}", file=sys.stderr, flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "sessions_manifest.h5"
    if manifests:
        manifest_frame = pd.concat(manifests, ignore_index=True)
    else:
        manifest_frame = pd.DataFrame()
    write_table_h5(manifest_frame, manifest_path, "sessions")
    if failures:
        write_table_h5(
            pd.DataFrame.from_records(failures),
            output_root / "preparation_errors.h5",
            "errors",
        )
    else:
        stale_error_path = output_root / "preparation_errors.h5"
        if stale_error_path.exists():
            stale_error_path.unlink()
    if not manifests:
        raise RuntimeError("Every behavior file failed. See preparation_errors.h5.")
    print(f"Manifest: {manifest_path}", flush=True)
    return manifest_path


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent
    try:
        result = main()
    except SystemExit:
        raise
    except Exception as error:
        details = traceback.format_exc()
        log_path = root_path / "prepare_sessions_from_behavior.log"
        log_path.write_text(details, encoding="utf-8")
        print(details, file=sys.stderr, flush=True)
        _show_message(
            "IMU Session Preparation - Error",
            f"Preparation failed:\n{error}\n\nDetails:\n{log_path}",
            error=True,
        )
        raise SystemExit(1) from None
    else:
        _show_message(
            "IMU Session Preparation - Finished",
            f"Fusion-ready sessions were generated.\n\nManifest:\n{result}",
        )
