"""Plot polar roll, yaw, and pitch distributions from a fusion result.

Edit the input path in the user configuration and run this file directly.
Zero degrees is at the top and angles increase clockwise.
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

# Native fusion HDF5 or a legacy CSV with roll, yaw, and pitch columns.
FUSION_RESULT_FILE = Path(r"D:\Jiaqi\tools\9_axies_IMU\output\fusion_result.h5")

# Histogram bins per revolution. A value of 144 gives 2.5 degrees per bin.
NUMBER_OF_BINS = 144

# Show probability density instead of sample counts.
SHOW_PROBABILITY_DENSITY = False

# High-resolution output path, resolved against the project root.
SAVE_FIGURE = True
OUTPUT_FIGURE_FILE = Path("output/euler_distribution.png")
FIGURE_DPI = 180

# Show an interactive Matplotlib window after saving.
SHOW_FIGURE = True


# =============================================================================
# Implementation
# =============================================================================


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    """Resolve a configuration path against the project root."""
    return path if path.is_absolute() else project_root / path


def _ensure_plotting_environment(project_root: Path) -> None:
    """Restart through uv so plotting dependencies are available."""
    if os.environ.get("IMU_EULER_PLOT_BOOTSTRAPPED") == "1":
        return
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError(
            "The launcher could not find uv. Install uv from "
            "https://docs.astral.sh/uv/ and run this file again."
        )
    child_environment = os.environ.copy()
    child_environment["IMU_EULER_PLOT_BOOTSTRAPPED"] = "1"
    print("Starting the locked plotting environment...", flush=True)
    completed = subprocess.run(
        [
            uv_executable,
            "run",
            "--extra",
            "plotting",
            "python",
            str(Path(__file__).resolve()),
        ],
        cwd=project_root,
        env=child_environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "The Euler plotting process failed with exit code "
            f"{completed.returncode}. See plot_euler_distribution.log."
        )
    raise SystemExit(0)


def _load_euler_angles(path: Path) -> "object":
    """Read only Euler data instead of loading every fusion dataset."""
    import h5py
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Fusion result does not exist: {path}")
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            dataset_path = "fusion/euler_roll_yaw_pitch"
            if dataset_path not in handle:
                raise KeyError(f"HDF5 dataset is missing: /{dataset_path}")
            euler = handle[dataset_path][()].astype(np.float64, copy=False)
    else:
        import pandas as pd

        table = pd.read_csv(path, usecols=["roll", "yaw", "pitch"])
        euler = table.to_numpy(dtype=np.float64, copy=False)

    if euler.ndim != 2 or euler.shape[1] != 3:
        raise ValueError("Euler data must have shape (samples, 3).")
    if euler.shape[0] == 0:
        raise ValueError("Euler data is empty.")
    return euler


def _centered_circular_values(values: "object", bin_count: int) -> "object":
    """Wrap angles to one revolution with zero at the first-bin center."""
    import numpy as np

    half_bin = np.pi / bin_count
    return (values + half_bin) % (2.0 * np.pi) - half_bin


def plot_euler_distribution(euler: "object", bin_count: int) -> "object":
    """Create the roll, yaw, and pitch polar histograms."""
    import matplotlib.pyplot as plt
    import numpy as np

    if bin_count < 4:
        raise ValueError("NUMBER_OF_BINS must be at least 4.")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.4),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    half_bin = np.pi / bin_count
    bin_edges = np.linspace(-half_bin, 2.0 * np.pi - half_bin, bin_count + 1)
    tick_degrees = np.arange(0, 360, 45)

    for axis, column, name, color in zip(
        axes,
        range(3),
        ("Roll", "Yaw", "Pitch"),
        ("#ef5350", "#43a047", "#1e88e5"),
        strict=True,
    ):
        values = np.asarray(euler[:, column], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"{name} contains no finite values.")
        circular = _centered_circular_values(values, bin_count)
        axis.hist(
            circular,
            bins=bin_edges,
            density=SHOW_PROBABILITY_DENSITY,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
        )
        axis.set_title(f"{name}\n(n = {values.size:,})", pad=18, fontsize=13)
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_xticks(np.deg2rad(tick_degrees))
        axis.set_xticklabels([f"{degree}°" for degree in tick_degrees])
        axis.grid(alpha=0.3)

    figure.suptitle("Roll / Yaw / Pitch Circular Distributions", fontsize=16)
    return figure


def main() -> Path | None:
    """Read the fusion result, render the figure, and save or show it."""
    project_root = Path(__file__).resolve().parent
    _ensure_plotting_environment(project_root)

    import matplotlib.pyplot as plt

    source_path = _resolve_project_path(project_root, FUSION_RESULT_FILE)
    print(f"Reading Euler angles: {source_path}", flush=True)
    euler = _load_euler_angles(source_path)
    print(f"Loaded {len(euler):,} samples. Plotting...", flush=True)
    figure = plot_euler_distribution(euler, NUMBER_OF_BINS)

    output_path: Path | None = None
    if SAVE_FIGURE:
        output_path = _resolve_project_path(project_root, OUTPUT_FIGURE_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Figure saved: {output_path}", flush=True)
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)
    return output_path


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        details = traceback.format_exc()
        log_path = root_path / "plot_euler_distribution.log"
        log_path.write_text(details, encoding="utf-8")
        print(details, file=sys.stderr, flush=True)
        print(f"Error details saved to: {log_path}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
