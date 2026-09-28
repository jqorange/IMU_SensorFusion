"""Extended synchronized viewer for IMU, pose, speed, and behavior states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation

from imu_fusion.neural_data import (
    NeuralData,
    load_neural_data,
    load_timestamp_origin,
)
from imu_fusion.viewer import (
    CartoonMouse3DWidget,
    FusionViewer,
    InteractiveVideoLabel,
)

BEHAVIOR_SAMPLE_RATE_HZ = 50.0
BEHAVIOR_SCORE_COLUMNS = (
    "relocation_score",
    "eating_score",
    "local_search_score",
    "stand_score",
)
BEHAVIOR_NAMES = ("relocation", "eating", "local_search", "stand")
BEHAVIOR_COLORS = {
    "relocation": "#ff9f43",
    "eating": "#50fa7b",
    "local_search": "#57c7ff",
    "stand": "#bd93f9",
}
POSE_KEYPOINTS = (
    "nose",
    "head",
    "leftEar",
    "rightEar",
    "bodyCenter1",
    "bodyCenter2",
    "tailStart",
)
POSE_EDGES = (
    ("nose", "head"),
    ("head", "leftEar"),
    ("head", "rightEar"),
    ("head", "bodyCenter1"),
    ("bodyCenter1", "bodyCenter2"),
    ("bodyCenter2", "tailStart"),
)


@dataclass(frozen=True, slots=True)
class BehaviorData:
    """Required framewise 50 Hz behavior and local-pose arrays."""

    time_s: np.ndarray
    states: np.ndarray
    scores: np.ndarray
    speed: np.ndarray
    poses: np.ndarray


def load_behavior_data(
    states_path: str | Path,
    pose_path: str | Path,
    sample_rate_hz: float = BEHAVIOR_SAMPLE_RATE_HZ,
) -> BehaviorData:
    """Read only the columns used by the extended viewer."""
    if sample_rate_hz <= 0.0:
        raise ValueError("Behavior sample rate must be positive.")
    state_columns = ["frame_idx", "state", *BEHAVIOR_SCORE_COLUMNS]
    states = pd.read_csv(
        states_path,
        usecols=state_columns,
        dtype={
            "frame_idx": np.int64,
            "state": "string",
            **{column: np.float32 for column in BEHAVIOR_SCORE_COLUMNS},
        },
    )
    pose_columns = [
        "bodyCenter1_v",
        "nose_x_point",
        "nose_y_point",
        "head_x_point",
        "head_y_point",
        "leftEar_x_point",
        "leftEar_y_point",
        "rightEar_x_point",
        "rightEar_y_point",
        "bodyCenter2_x_point",
        "tailStart_x_point",
        "tailStart_y_point",
    ]
    pose = pd.read_csv(
        pose_path,
        usecols=pose_columns,
        dtype={column: np.float32 for column in pose_columns},
    )
    row_count = min(len(states), len(pose))
    if row_count < 2:
        raise ValueError("Behavior and pose files must contain at least two rows.")
    frame_indices = states["frame_idx"].to_numpy()[:row_count]
    if not np.array_equal(frame_indices, np.arange(row_count)):
        raise ValueError("Behavior frame_idx must be consecutive and start at zero.")

    poses = np.full((row_count, len(POSE_KEYPOINTS), 2), np.nan, dtype=np.float32)
    keypoint_index = {name: index for index, name in enumerate(POSE_KEYPOINTS)}
    for keypoint in ("nose", "head", "leftEar", "rightEar", "tailStart"):
        index = keypoint_index[keypoint]
        poses[:, index, 0] = pose[f"{keypoint}_x_point"].to_numpy()[:row_count]
        poses[:, index, 1] = pose[f"{keypoint}_y_point"].to_numpy()[:row_count]

    # These local pose features are bodyCenter1-relative. The omitted values
    # are therefore defined by construction rather than missing measurements.
    body_center_1 = keypoint_index["bodyCenter1"]
    body_center_2 = keypoint_index["bodyCenter2"]
    poses[:, body_center_1, :] = 0.0
    poses[:, body_center_2, 0] = pose["bodyCenter2_x_point"].to_numpy()[:row_count]
    poses[:, body_center_2, 1] = 0.0

    return BehaviorData(
        time_s=np.arange(row_count, dtype=np.float64) / sample_rate_hz,
        states=states["state"].fillna("unknown").to_numpy(dtype=str)[:row_count],
        scores=states[list(BEHAVIOR_SCORE_COLUMNS)].to_numpy(dtype=np.float32)[
            :row_count
        ],
        speed=pose["bodyCenter1_v"].to_numpy(dtype=np.float32)[:row_count],
        poses=poses,
    )


class PoseSkeletonWidget(pg.PlotWidget):
    """Draw a short fading pose history with the current skeleton emphasized."""

    _HISTORY_COUNT = 7

    def __init__(self, poses: np.ndarray) -> None:
        super().__init__()
        self.setBackground("#0b0f16")
        self.invertY(True)
        self.hideAxis("left")
        self.hideAxis("bottom")
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self._keypoint_index = {
            name: index for index, name in enumerate(POSE_KEYPOINTS)
        }
        self._history_curves: list[pg.PlotDataItem] = []
        for index in range(self._HISTORY_COUNT):
            fraction = index / max(1, self._HISTORY_COUNT - 1)
            color = QtGui.QColor.fromHsvF(0.51 + 0.22 * fraction, 0.68, 1.0)
            color.setAlphaF(0.025 + 0.86 * fraction**2.4)
            self._history_curves.append(
                self.plot([], [], pen=pg.mkPen(color, width=0.8 + fraction * 2.4))
            )
        self._points = pg.ScatterPlotItem(
            size=7,
            brush=pg.mkBrush("#f8f8f2"),
            pen=pg.mkPen("#20242c", width=1),
        )
        self.addItem(self._points)
        self._set_robust_range(poses)

    def _set_robust_range(self, poses: np.ndarray) -> None:
        values = poses.reshape(-1, 2)
        finite = np.all(np.isfinite(values), axis=1)
        if not np.any(finite):
            self.setRange(xRange=(-10.0, 10.0), yRange=(-10.0, 10.0), padding=0.0)
            self.setAspectLocked(True, ratio=1.0)
            return
        low = np.nanpercentile(values[finite], 2.0, axis=0)
        high = np.nanpercentile(values[finite], 98.0, axis=0)
        center = (low + high) / 2.0
        half_extent = max(float(np.max(high - low)) * 0.52, 1.0)
        limit_extent = half_extent * 4.0
        self.setLimits(
            xMin=float(center[0] - limit_extent),
            xMax=float(center[0] + limit_extent),
            yMin=float(center[1] - limit_extent),
            yMax=float(center[1] + limit_extent),
            minXRange=0.1,
            maxXRange=limit_extent * 2.0,
            minYRange=0.1,
            maxYRange=limit_extent * 2.0,
        )
        self.setRange(
            xRange=(center[0] - half_extent, center[0] + half_extent),
            yRange=(center[1] - half_extent, center[1] + half_extent),
            padding=0.0,
        )
        self.setAspectLocked(True, ratio=1.0)

    def _skeleton_coordinates(
        self, pose: np.ndarray
    ) -> tuple[list[float], list[float]]:
        x_values: list[float] = []
        y_values: list[float] = []
        for start_name, end_name in POSE_EDGES:
            start = pose[self._keypoint_index[start_name]]
            end = pose[self._keypoint_index[end_name]]
            if np.all(np.isfinite(start)) and np.all(np.isfinite(end)):
                x_values.extend((float(start[0]), float(end[0]), np.nan))
                y_values.extend((float(start[1]), float(end[1]), np.nan))
        return x_values, y_values

    def set_pose(
        self, poses: np.ndarray, current_index: int, window_samples: int
    ) -> None:
        """Update the short, quickly fading pose overlay."""
        first = max(0, current_index - window_samples + 1)
        indices = np.linspace(
            first,
            current_index,
            self._HISTORY_COUNT,
            dtype=np.int64,
        )
        for curve, pose_index in zip(self._history_curves, indices, strict=True):
            x_values, y_values = self._skeleton_coordinates(poses[pose_index])
            curve.setData(x_values, y_values)
        current = poses[current_index]
        finite = np.all(np.isfinite(current), axis=1)
        self._points.setData(x=current[finite, 0], y=current[finite, 1])


class BehaviorFusionViewer(FusionViewer):
    """Fusion viewer with combined Euler, speed, pose, and state panels."""

    _POSE_HISTORY_SECONDS = 0.6

    def __init__(
        self,
        result_path: Path,
        video_path: Path,
        timestamp_path: Path,
        timestamp_key: str | None,
        *,
        behavior_states_path: Path,
        pose_features_path: Path,
        behavior_sample_rate_hz: float = BEHAVIOR_SAMPLE_RATE_HZ,
        neural_recording_directory: Path | None = None,
        neural_session_start_seconds: float | None = None,
        lfp_anchor_seconds: float | None = None,
        lfp_channel: int = 0,
        neural_window_seconds: float = 5.0,
        raster_bin_seconds: float = 0.05,
        minimum_mean_firing_rate_hz: float = 0.2,
        neural_update_interval_seconds: float = 0.05,
        **viewer_options: Any,
    ) -> None:
        self._behavior_sample_rate_hz = float(behavior_sample_rate_hz)
        self._behavior = load_behavior_data(
            behavior_states_path,
            pose_features_path,
            self._behavior_sample_rate_hz,
        )
        self._neural: NeuralData | None = None
        self._neural_update_interval_seconds = max(
            0.01, float(neural_update_interval_seconds)
        )
        self._last_neural_update_key: int | None = None
        if neural_recording_directory is not None:
            session_start = (
                load_timestamp_origin(
                    timestamp_path,
                    timestamp_key,
                    segment_index=int(viewer_options.get("timestamp_segment_index", 1)),
                    gap_seconds=float(
                        viewer_options.get("timestamp_gap_seconds", 60.0)
                    ),
                )
                if neural_session_start_seconds is None
                else float(neural_session_start_seconds)
            )
            session_duration = float(
                self._behavior.time_s[-1] + 1.0 / self._behavior_sample_rate_hz
            )
            self._neural = load_neural_data(
                neural_recording_directory,
                session_start_seconds=session_start,
                session_duration_seconds=session_duration,
                lfp_anchor_seconds=lfp_anchor_seconds,
                lfp_channel=lfp_channel,
                minimum_mean_firing_rate_hz=minimum_mean_firing_rate_hz,
                window_seconds=neural_window_seconds,
                raster_bin_seconds=raster_bin_seconds,
            )
        super().__init__(
            result_path,
            video_path,
            timestamp_path,
            timestamp_key,
            **viewer_options,
        )
        self.setWindowTitle("IMU and Behavior Inspector")
        self.resize(1840, 1080)

    @staticmethod
    def _style_plot(plot: pg.PlotWidget, y_label: str) -> None:
        plot.setLabel("bottom", "Time", units="s")
        plot.setLabel("left", y_label)
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.setMouseEnabled(x=False, y=False)
        plot.setMenuEnabled(False)

    @staticmethod
    def _cursor() -> pg.InfiniteLine:
        return pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("#f1fa8c", width=2),
        )

    def _build_ui(self, timestamp_name: str, video_name: str) -> None:
        pg.setConfigOptions(antialias=False, background="#11161f", foreground="#d8dee9")
        central = QtWidgets.QWidget()
        central.setStyleSheet(
            "QWidget { background:#0d1118; color:#d8dee9; }"
            "QPushButton, QComboBox, QDoubleSpinBox {"
            " background:#1b2430; border:1px solid #344050; border-radius:5px;"
            " padding:5px 9px; }"
            "QPushButton:hover { background:#273445; }"
        )
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)

        top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._video = InteractiveVideoLabel()
        self._video.setMinimumSize(720, 400)
        self._video.setToolTip(
            "Mouse wheel: zoom around bodyCenter1  |  Double-click: reset"
        )
        top.addWidget(self._video)
        self._orientation = CartoonMouse3DWidget(
            self._cartoon_camera_azimuth_degrees,
            self._cartoon_camera_elevation_degrees,
            self._cartoon_world_yaw_alignment_degrees,
        )
        top.addWidget(self._orientation)

        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 2)
        root.addWidget(top, 5)

        analytics = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        analytics.setChildrenCollapsible(False)
        analytics.setHandleWidth(8)
        analytics.setStyleSheet(
            "QSplitter::handle:horizontal {"
            " background:#263445; margin:2px 1px; border-radius:3px;"
            "}"
            "QSplitter::handle:horizontal:hover { background:#57c7ff; }"
        )
        signal_column = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        signal_column.setChildrenCollapsible(False)

        self._euler_plot = pg.PlotWidget(title="Orientation · 2 s")
        self._style_plot(self._euler_plot, "Angle (deg)")
        self._euler_plot.setYRange(-180.0, 180.0, padding=0.0)
        self._euler_plot.addLegend(offset=(8, 8))
        self._euler_curves = [
            self._euler_plot.plot([], [], name=name, pen=pg.mkPen(color, width=2))
            for name, color in (
                ("Roll", "#ff5c57"),
                ("Yaw", "#50fa7b"),
                ("Pitch", "#57c7ff"),
            )
        ]
        self._euler_cursor = self._cursor()
        self._euler_plot.addItem(self._euler_cursor)
        signal_column.addWidget(self._euler_plot)

        self._speed_plot = pg.PlotWidget(title="BodyCenter1 speed · 2 s")
        self._style_plot(self._speed_plot, "Speed")
        self._speed_curve = self._speed_plot.plot(
            [],
            [],
            pen=pg.mkPen("#ffb86c", width=2.2),
            fillLevel=0.0,
            brush=pg.mkBrush(255, 184, 108, 28),
        )
        self._speed_cursor = self._cursor()
        self._speed_plot.addItem(self._speed_cursor)
        signal_column.addWidget(self._speed_plot)
        signal_column.setStretchFactor(0, 3)
        signal_column.setStretchFactor(1, 2)
        analytics.addWidget(signal_column)

        behavior_column = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        behavior_column.setChildrenCollapsible(False)
        pose_panel = QtWidgets.QWidget()
        pose_layout = QtWidgets.QVBoxLayout(pose_panel)
        pose_layout.setContentsMargins(0, 0, 0, 0)
        pose_layout.setSpacing(4)
        self._state_label = QtWidgets.QLabel("UNKNOWN")
        self._state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._state_label.setFixedHeight(34)
        pose_layout.addWidget(self._state_label)
        self._pose_plot = PoseSkeletonWidget(self._behavior.poses)
        self._pose_plot.setMinimumHeight(250)
        pose_layout.addWidget(self._pose_plot, 1)
        behavior_column.addWidget(pose_panel)

        score_row = QtWidgets.QWidget()
        score_layout = QtWidgets.QHBoxLayout(score_row)
        score_layout.setContentsMargins(0, 0, 0, 0)
        score_layout.setSpacing(3)
        self._score_plot = pg.PlotWidget(title="Behavior scores · 2 s")
        self._style_plot(self._score_plot, "Score")
        self._score_plot.setYRange(0.0, 1.0, padding=0.02)
        self._score_curves = [
            self._score_plot.plot(
                [],
                [],
                pen=pg.mkPen(BEHAVIOR_COLORS[name], width=1.8),
            )
            for name in BEHAVIOR_NAMES
        ]
        self._score_cursor = self._cursor()
        self._score_plot.addItem(self._score_cursor)
        score_layout.addWidget(self._score_plot, 1)

        score_legend = QtWidgets.QWidget()
        score_legend.setFixedWidth(86)
        legend_layout = QtWidgets.QVBoxLayout(score_legend)
        legend_layout.setContentsMargins(3, 18, 0, 3)
        legend_layout.setSpacing(1)
        for name in BEHAVIOR_NAMES:
            label = QtWidgets.QLabel(f"━ {name.replace('_', ' ').title()}")
            label.setStyleSheet(
                f"color:{BEHAVIOR_COLORS[name]}; font-size:8pt; border:none;"
            )
            legend_layout.addWidget(label)
        legend_layout.addStretch(1)
        score_layout.addWidget(score_legend)
        behavior_column.addWidget(score_row)
        behavior_column.setStretchFactor(0, 3)
        behavior_column.setStretchFactor(1, 2)
        analytics.addWidget(behavior_column)

        neural_column = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        neural_column.setChildrenCollapsible(False)
        self._lfp_plot = pg.PlotWidget(title="LFP · 5 s")
        self._style_plot(self._lfp_plot, "Amplitude")
        self._lfp_plot.setLabel("bottom", "Relative time", units="s")
        self._lfp_curve = self._lfp_plot.plot(
            [],
            [],
            pen=pg.mkPen("#8be9fd", width=1.15),
            skipFiniteCheck=True,
        )
        self._lfp_cursor = self._cursor()
        self._lfp_cursor.setValue(0.0)
        self._lfp_plot.addItem(self._lfp_cursor)
        neural_column.addWidget(self._lfp_plot)

        self._raster_plot = pg.PlotWidget(title="Spike raster · 5 s")
        self._style_plot(self._raster_plot, "Neurons")
        self._raster_plot.setBackground("#0a1018")
        self._raster_plot.showGrid(x=True, y=False, alpha=0.12)
        self._raster_plot.setLabel("bottom", "Relative time", units="s")
        self._raster_plot.getViewBox().invertY(True)
        self._raster_image = pg.ImageItem(axisOrder="row-major")
        raster_colors = ["#0a1018", "#17344a", "#287083", "#6ed6c5"]
        color_map = pg.ColorMap(
            np.linspace(0.0, 1.0, len(raster_colors)), raster_colors
        )
        self._raster_image.setLookupTable(color_map.getLookupTable(nPts=256))
        self._raster_plot.addItem(self._raster_image)
        self._raster_cursor = self._cursor()
        self._raster_cursor.setValue(0.0)
        self._raster_plot.addItem(self._raster_cursor)
        neural_column.addWidget(self._raster_plot)
        neural_column.setStretchFactor(0, 1)
        neural_column.setStretchFactor(1, 1)
        analytics.addWidget(neural_column)

        if self._neural is None:
            message = "Neural alignment metadata or recording is unavailable"
            for plot in (self._lfp_plot, self._raster_plot):
                text_item = pg.TextItem(message, color="#718096", anchor=(0.5, 0.5))
                text_item.setPos(0.0, 0.0)
                plot.addItem(text_item)
                plot.setXRange(-2.5, 2.5, padding=0.0)
                plot.setYRange(-1.0, 1.0, padding=0.0)

        analytics.setStretchFactor(0, 10)
        analytics.setStretchFactor(1, 13)
        analytics.setStretchFactor(2, 11)
        root.addWidget(analytics, 6)

        controls = QtWidgets.QHBoxLayout()
        self._play = QtWidgets.QPushButton("▶ Play")
        self._play.clicked.connect(self._toggle_play)
        controls.addWidget(self._play)
        self._previous = QtWidgets.QPushButton("◀ Frame")
        self._configure_frame_button(self._previous, -1)
        controls.addWidget(self._previous)
        self._following = QtWidgets.QPushButton("Frame ▶")
        self._configure_frame_button(self._following, 1)
        controls.addWidget(self._following)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, max(0, self._available_frames - 1))
        self._slider.sliderPressed.connect(self._begin_slider_drag)
        self._slider.sliderMoved.connect(self._schedule_slider_seek)
        self._slider.sliderReleased.connect(self._commit_slider_seek)
        controls.addWidget(self._slider, 1)
        controls.addWidget(QtWidgets.QLabel("FPS"))
        self._fps = self._create_fps_control()
        controls.addWidget(self._fps)
        controls.addWidget(QtWidgets.QLabel("Speed"))
        self._speed = QtWidgets.QComboBox()
        self._speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self._speed.setCurrentText("1×")
        self._speed.currentTextChanged.connect(self._update_timer)
        controls.addWidget(self._speed)
        controls.addWidget(QtWidgets.QLabel("Offset (s)"))
        self._offset = QtWidgets.QDoubleSpinBox()
        self._offset.setRange(-3600.0, 3600.0)
        self._offset.setDecimals(3)
        self._offset.setSingleStep(0.01)
        self._offset.valueChanged.connect(
            lambda _: self._update_fusion_overlay(self._frame_index)
        )
        controls.addWidget(self._offset)
        root.addLayout(controls)
        self._status = QtWidgets.QLabel(
            f"{video_name}  •  MAT: {timestamp_name}  •  Space: play/pause  "
            "•  ←/→: frame  •  Wheel over video: tracked zoom"
        )
        root.addWidget(self._status)
        self.setCentralWidget(central)

    @staticmethod
    def _nearest_index(time_s: np.ndarray, target_s: float) -> int:
        right = int(np.clip(np.searchsorted(time_s, target_s), 0, len(time_s) - 1))
        left = max(0, right - 1)
        if abs(time_s[left] - target_s) <= abs(time_s[right] - target_s):
            return left
        return right

    @staticmethod
    def _window_bounds(
        time_s: np.ndarray, center_s: float, duration_s: float
    ) -> tuple[float, float, int, int]:
        half = duration_s / 2.0
        start = center_s - half
        end = center_s + half
        if start < time_s[0]:
            start = float(time_s[0])
            end = min(float(time_s[-1]), start + duration_s)
        elif end > time_s[-1]:
            end = float(time_s[-1])
            start = max(float(time_s[0]), end - duration_s)
        first = int(np.searchsorted(time_s, start, side="left"))
        last = int(np.searchsorted(time_s, end, side="right"))
        return start, end, first, last

    def _update_neural_plots(self, fusion_time: float) -> None:
        """Refresh neural panels at a bounded rate during playback and seeking."""
        if self._neural is None:
            return
        update_key = int(
            np.floor(fusion_time / self._neural_update_interval_seconds + 0.5)
        )
        if update_key == self._last_neural_update_key:
            return
        self._last_neural_update_key = update_key
        neural_window = self._neural.centered_window(fusion_time)
        half_window = self._neural.window_seconds / 2.0

        lfp_time = neural_window.lfp_time_s
        lfp_values = neural_window.lfp
        if len(lfp_values) > 4000:
            stride = int(np.ceil(len(lfp_values) / 4000))
            lfp_time = lfp_time[::stride]
            lfp_values = lfp_values[::stride]
        self._lfp_curve.setData(lfp_time, lfp_values)
        self._lfp_plot.setXRange(-half_window, half_window, padding=0.0)
        if len(lfp_values):
            low, high = np.nanpercentile(lfp_values, (1.0, 99.0))
            if np.isfinite(low) and np.isfinite(high):
                if high <= low:
                    low, high = low - 1.0, high + 1.0
                margin = (high - low) * 0.08
                self._lfp_plot.setYRange(low - margin, high + margin, padding=0.0)

        raster = np.clip(neural_window.raster, 0, 3)
        self._raster_image.setImage(raster, autoLevels=False, levels=(0, 3))
        self._raster_image.setRect(
            QtCore.QRectF(
                -half_window,
                0.0,
                self._neural.window_seconds,
                float(raster.shape[0]),
            )
        )
        self._raster_plot.setXRange(-half_window, half_window, padding=0.0)
        self._raster_plot.setYRange(0.0, float(raster.shape[0]), padding=0.0)
        self._raster_plot.setTitle(f"Spike raster · 5 s · {raster.shape[0]} neurons")

    def _update_fusion_overlay(self, index: int, *, render_video: bool = True) -> None:
        fusion_time = self._timestamps[index] + self._offset.value()
        imu_index = self._nearest_index(self._time_s, fusion_time)
        start, end, first, last = self._window_bounds(
            self._time_s, fusion_time, self._plot_window_seconds
        )
        plot_time = self._time_s[first:last]
        wrapped_degrees = (np.rad2deg(self._euler[first:last]) + 180.0) % 360.0 - 180.0
        for axis, curve in enumerate(self._euler_curves):
            values = wrapped_degrees[:, axis].copy()
            values[1:][np.abs(np.diff(values)) > 180.0] = np.nan
            curve.setData(plot_time, values)
        self._euler_plot.setXRange(start, end, padding=0.0)
        self._euler_cursor.setValue(fusion_time)

        q_wxyz = self._quaternion_aligned[imu_index]
        matrix = Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()
        self._current_display_matrix = self._display_yaw_matrix @ matrix
        self._orientation.set_matrix(matrix)
        if render_video and self._last_frame is not None:
            self._render_video_frame(self._last_frame, index)

        behavior_index = self._nearest_index(self._behavior.time_s, fusion_time)
        behavior_start, behavior_end, behavior_first, behavior_last = (
            self._window_bounds(
                self._behavior.time_s,
                fusion_time,
                self._plot_window_seconds,
            )
        )
        behavior_time = self._behavior.time_s[behavior_first:behavior_last]
        self._speed_curve.setData(
            behavior_time,
            self._behavior.speed[behavior_first:behavior_last],
        )
        self._speed_plot.setXRange(behavior_start, behavior_end, padding=0.0)
        self._speed_cursor.setValue(fusion_time)
        current_speed = float(self._behavior.speed[behavior_index])
        self._speed_plot.setTitle(
            f"BodyCenter1 speed: {current_speed:.3f}", color="#ffb86c"
        )

        for score_index, curve in enumerate(self._score_curves):
            curve.setData(
                behavior_time,
                self._behavior.scores[behavior_first:behavior_last, score_index],
            )
        self._score_plot.setXRange(behavior_start, behavior_end, padding=0.0)
        self._score_cursor.setValue(fusion_time)
        self._update_neural_plots(fusion_time)
        pose_window_samples = max(
            2,
            round(self._POSE_HISTORY_SECONDS * self._behavior_sample_rate_hz),
        )
        self._pose_plot.set_pose(
            self._behavior.poses,
            behavior_index,
            pose_window_samples,
        )

        state = self._behavior.states[behavior_index]
        state_color = BEHAVIOR_COLORS.get(state, "#d8dee9")
        self._state_label.setText(state.replace("_", " ").upper())
        self._state_label.setStyleSheet(
            f"font-size:20px; font-weight:700; color:{state_color};"
            "background:#151b25; border-radius:6px; padding:5px;"
        )
        angles = (np.rad2deg(self._euler[imu_index]) + 180.0) % 360.0 - 180.0
        roll, yaw, pitch = angles
        self._euler_plot.setTitle(
            f"R {roll:+.1f}°   Y {yaw:+.1f}°   P {pitch:+.1f}°",
            color="#d8dee9",
        )
        self._status.setText(
            f"Frame {index + 1:,}/{self._available_frames:,}  |  "
            f"video {self._timestamps[index]:.3f} s  |  "
            f"fusion {fusion_time:.3f} s  |  IMU {imu_index:,}  |  "
            f"behavior {behavior_index:,}: {state}  |  speed {current_speed:.3f}"
        )
