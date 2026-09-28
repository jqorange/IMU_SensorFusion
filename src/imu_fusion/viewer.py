"""Responsive Qt viewer for synchronized video and fused IMU results."""

from __future__ import annotations

import argparse
import csv
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
import trimesh
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

from imu_fusion.io import read_fusion_view_data


def _walk_mat(value: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    """Collect one-dimensional arrays from nested ``loadmat`` structures."""
    found: list[tuple[str, np.ndarray]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if not key.startswith("__"):
                found.extend(_walk_mat(child, f"{prefix}.{key}".strip(".")))
    elif isinstance(value, np.ndarray):
        squeezed = np.squeeze(value)
        if squeezed.ndim == 1 and squeezed.size > 1:
            found.append((prefix, squeezed))
    return found


def load_video_timestamps(
    mat_path: str | Path,
    expected_frames: int,
    key: str | None = None,
    segment_index: int = 1,
    gap_seconds: float = 60.0,
) -> tuple[np.ndarray, str]:
    """Load relative seconds from a MATLAB timestamp array.

    Numeric arrays are accepted directly. Datetime strings are converted to
    elapsed seconds. Without ``key``, names containing ``video`` and
    ``timestamp`` are preferred, then the length nearest to the video frame
    count is chosen.
    """
    content = loadmat(mat_path, simplify_cells=True)
    candidates = _walk_mat(content)
    if key:
        candidates = [(name, values) for name, values in candidates if name == key]
        if not candidates:
            names = ", ".join(name for name, _ in _walk_mat(content))
            raise KeyError(f"MAT key {key!r} was not found. Available arrays: {names}")
    if not candidates:
        raise ValueError(f"No one-dimensional timestamp arrays in {mat_path}.")
    if segment_index < 1:
        raise ValueError("Timestamp segment index must be at least 1.")

    def rank(item: tuple[str, np.ndarray]) -> tuple[int, int]:
        name, values = item
        lowered = name.lower()
        preferred = int(not ("timestamp" in lowered or "time" in lowered))
        video_bonus = 0 if "video" in lowered else 1
        return preferred + video_bonus, abs(values.size - expected_frames)

    name, values = min(candidates, key=rank)
    if np.issubdtype(values.dtype, np.number):
        seconds = values.astype(np.float64)
        # MATLAB datenum values are measured in days.
        if np.nanmedian(np.abs(seconds)) > 1e5:
            seconds = (seconds - seconds[0]) * 86400.0
        else:
            seconds = seconds - seconds[0]
    else:
        datetimes = pd.to_datetime(values.astype(str), errors="raise")
        seconds = (datetimes - datetimes[0]).total_seconds().to_numpy()
    if not np.all(np.isfinite(seconds)) or np.any(np.diff(seconds) < 0.0):
        raise ValueError(f"Timestamp array {name!r} must be finite and monotonic.")
    breaks = np.flatnonzero(np.diff(seconds) > gap_seconds)
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks + 1, [len(seconds)]))
    if segment_index > len(starts):
        raise ValueError(
            f"Timestamp array {name!r} has {len(starts)} segment(s); "
            f"segment {segment_index} was requested."
        )
    start = int(starts[segment_index - 1])
    stop = int(stops[segment_index - 1])
    seconds = seconds[start:stop]
    seconds = seconds - seconds[0]
    if abs(len(seconds) - expected_frames) > 1:
        raise ValueError(
            f"Timestamp segment {segment_index} contains {len(seconds):,} values, "
            f"but the video contains {expected_frames:,} frames."
        )
    return seconds, f"{name} [segment {segment_index}]"


def load_dlc_positions(
    csv_path: str | Path,
    bodypart: str = "bodyCenter1",
    min_likelihood: float = 0.5,
) -> np.ndarray:
    """Load and interpolate one DeepLabCut body part as framewise XY pixels."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            _scorer = next(reader)
            bodyparts = next(reader)
            coordinates = next(reader)
        except StopIteration as error:
            raise ValueError(
                f"DLC CSV must contain three header rows: {path}"
            ) from error
    if len(bodyparts) != len(coordinates):
        raise ValueError(f"DLC header rows have different lengths: {path}")

    column_by_coordinate = {
        coordinate.strip().lower(): index
        for index, (part, coordinate) in enumerate(zip(bodyparts, coordinates))
        if part.strip() == bodypart
    }
    missing = {"x", "y"} - column_by_coordinate.keys()
    if missing:
        raise KeyError(
            f"DLC body part {bodypart!r} is missing columns "
            f"{sorted(missing)} in {path}."
        )

    frame_column = 0
    x_column = column_by_coordinate["x"]
    y_column = column_by_coordinate["y"]
    likelihood_column = column_by_coordinate.get("likelihood")
    selected_columns = [frame_column, x_column, y_column]
    if likelihood_column is not None:
        selected_columns.append(likelihood_column)
    table = pd.read_csv(
        path,
        header=None,
        skiprows=3,
        usecols=selected_columns,
        dtype=np.float64,
    )
    if table.empty:
        raise ValueError(f"DLC CSV contains no frames: {path}")
    frame_numbers = table[frame_column].to_numpy()
    expected = np.arange(len(table), dtype=np.float64)
    if not np.array_equal(frame_numbers, expected):
        raise ValueError("DLC frame indices must be consecutive and start at zero.")

    positions = table[[x_column, y_column]].to_numpy(dtype=np.float64, copy=True)
    valid = np.all(np.isfinite(positions), axis=1)
    if likelihood_column is not None:
        likelihood = table[likelihood_column].to_numpy(dtype=np.float64)
        valid &= np.isfinite(likelihood) & (likelihood >= min_likelihood)
    if not np.any(valid):
        raise ValueError(
            f"No valid {bodypart!r} positions remain at likelihood "
            f">= {min_likelihood:.3f}."
        )

    # Interpolation prevents low-confidence detections from making the crop
    # jump away from the animal. Leading/trailing gaps use the nearest valid
    # observation, matching numpy.interp boundary behavior.
    frame_axis = np.arange(len(positions), dtype=np.float64)
    valid_frames = frame_axis[valid]
    for axis in range(2):
        positions[:, axis] = np.interp(
            frame_axis,
            valid_frames,
            positions[valid, axis],
        )
    return positions


class _CleanupCartoonMouse3DWidget(QtWidgets.QWidget):
    """Opaque procedural 3D mouse head with a clear three-quarter view."""

    _BACKGROUND = QtGui.QColor("#11161f")

    def __init__(
        self,
        camera_azimuth_degrees: float,
        camera_elevation_degrees: float,
        world_yaw_alignment_degrees: float,
    ) -> None:
        super().__init__()
        self.setMinimumSize(330, 420)
        self._matrix = np.eye(3)
        self._world_alignment = Rotation.from_euler(
            "z", world_yaw_alignment_degrees, degrees=True
        ).as_matrix()
        self._camera_direction, self._camera_right, self._camera_up = (
            self._camera_basis(camera_azimuth_degrees, camera_elevation_degrees)
        )
        self._parts = self._build_mouse()

    @staticmethod
    def _camera_basis(
        azimuth_degrees: float, elevation_degrees: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return orthonormal camera direction, right, and up vectors."""
        azimuth = np.deg2rad(azimuth_degrees)
        elevation = np.deg2rad(elevation_degrees)
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        right = np.cross(np.array([0.0, 0.0, 1.0]), direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        return direction, right, up

    @staticmethod
    def _ellipsoid(
        center: tuple[float, float, float],
        radii: tuple[float, float, float],
        color: str,
        subdivisions: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, QtGui.QColor]:
        """Create one smooth, opaque ellipsoid component."""
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(radii)
        vertices += np.asarray(center)
        return vertices, np.asarray(mesh.faces, dtype=np.int32), QtGui.QColor(color)

    @classmethod
    def _build_mouse(cls) -> list[tuple[np.ndarray, np.ndarray, QtGui.QColor]]:
        """Construct a stylized mouse head from solid surfaces."""
        return [
            cls._ellipsoid((-0.15, 0.0, 0.0), (1.55, 0.92, 0.88), "#9b8e84"),
            cls._ellipsoid((1.10, 0.0, -0.04), (1.20, 0.68, 0.58), "#aa9b90"),
            cls._ellipsoid((1.72, 0.0, -0.12), (0.58, 0.46, 0.37), "#c2afa2"),
            cls._ellipsoid((-0.45, 0.74, 0.73), (0.39, 0.23, 0.68), "#887971"),
            cls._ellipsoid((-0.45, -0.74, 0.73), (0.39, 0.23, 0.68), "#887971"),
            cls._ellipsoid((-0.40, 0.78, 0.76), (0.25, 0.10, 0.48), "#d49b9b", 1),
            cls._ellipsoid((-0.40, -0.78, 0.76), (0.25, 0.10, 0.48), "#d49b9b", 1),
            cls._ellipsoid((0.82, 0.69, 0.34), (0.18, 0.11, 0.18), "#161414", 1),
            cls._ellipsoid((0.82, -0.69, 0.34), (0.18, 0.11, 0.18), "#161414", 1),
            cls._ellipsoid((2.25, 0.0, -0.08), (0.18, 0.21, 0.16), "#49383a", 1),
        ]

    def set_matrix(self, matrix: np.ndarray) -> None:
        """Set the body-to-world rotation matrix and schedule a repaint."""
        self._matrix = np.asarray(matrix, dtype=np.float64)
        self.update()

    def _project(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotate model vertices and project them orthographically."""
        transform = self._world_alignment @ self._matrix
        world = vertices @ transform.T
        projected = np.column_stack(
            (world @ self._camera_right, -(world @ self._camera_up))
        )
        depth = world @ self._camera_direction
        scale = 0.34 * min(self.width(), self.height())
        screen = projected * scale + np.array([self.width() / 2, self.height() / 2])
        return screen, depth

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paint back-to-front opaque faces and the rotated sensor axes."""
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self._BACKGROUND)

        faces_to_draw: list[tuple[float, QtGui.QPolygonF, QtGui.QColor]] = []
        transform = self._world_alignment @ self._matrix
        light_direction = np.array([0.35, -0.25, 0.90])
        for vertices, faces, base_color in self._parts:
            screen, depth = self._project(vertices)
            world = vertices @ transform.T
            triangles = world[faces]
            normals = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            lengths = np.linalg.norm(normals, axis=1)
            valid = lengths > 1e-9
            normals[valid] /= lengths[valid, None]
            brightness = np.clip(normals @ light_direction, 0.0, 1.0)
            visible = valid & ((normals @ self._camera_direction) > 0.0)
            for face_index in np.flatnonzero(visible):
                polygon = QtGui.QPolygonF(
                    [QtCore.QPointF(*screen[index]) for index in faces[face_index]]
                )
                factor = 0.48 + 0.52 * brightness[face_index]
                color = QtGui.QColor(
                    round(base_color.red() * factor),
                    round(base_color.green() * factor),
                    round(base_color.blue() * factor),
                )
                faces_to_draw.append(
                    (float(np.mean(depth[faces[face_index]])), polygon, color)
                )

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for _, polygon, color in sorted(faces_to_draw, key=lambda item: item[0]):
            painter.setBrush(color)
            painter.drawPolygon(polygon)

        self._draw_axes(painter)
        painter.end()

    def _draw_axes(self, painter: QtGui.QPainter) -> None:
        """Draw positive body axes using the same orientation as the mouse."""
        points = np.vstack((np.zeros(3), np.eye(3) * 1.35))
        screen, _ = self._project(points)
        origin = QtCore.QPointF(*screen[0])
        axis_styles = zip(
            screen[1:],
            ("X", "Y", "Z"),
            ("#ff5c57", "#5af78e", "#57c7ff"),
            strict=True,
        )
        for endpoint, label, color in axis_styles:
            target = QtCore.QPointF(*endpoint)
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 3.0))
            painter.drawLine(origin, target)
            painter.setPen(QtGui.QColor(color))
            painter.drawText(target + QtCore.QPointF(4.0, -4.0), label)


class CartoonMouse3DWidget(QtWidgets.QWidget):
    """Opaque procedural 3D mouse head with a clear three-quarter view."""

    _MATERIALS = np.array(
        [
            [151, 145, 142, 255],  # fur
            [181, 139, 143, 255],  # inner ear
            [40, 35, 35, 255],  # eyes
            [236, 230, 224, 255],  # eye highlight
            [74, 58, 58, 255],  # nose
            [174, 166, 160, 255],  # muzzle highlight
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        camera_azimuth_degrees: float = -37.5,
        camera_elevation_degrees: float = 30.0,
        world_yaw_alignment_degrees: float = 0.0,
    ) -> None:
        super().__init__()
        self.setMinimumSize(320, 320)
        self._matrix = np.eye(3)
        self._world_yaw_matrix = Rotation.from_euler(
            "Z", world_yaw_alignment_degrees, degrees=True
        ).as_matrix()
        azimuth = np.deg2rad(camera_azimuth_degrees)
        elevation = np.deg2rad(camera_elevation_degrees)
        self._camera = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        self._screen_right = np.cross(np.array([0.0, 0.0, 1.0]), self._camera)
        self._screen_right /= np.linalg.norm(self._screen_right)
        self._screen_up = np.cross(self._camera, self._screen_right)
        (
            self._vertices,
            self._faces,
            self._face_normals,
            self._face_materials,
        ) = self._build_mouse_mesh()

    @staticmethod
    def _ellipsoid(
        center: tuple[float, float, float],
        radii: tuple[float, float, float],
        material: int,
        subdivisions: int,
        rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
        local_rotation = Rotation.from_euler(
            "XYZ", rotation_degrees, degrees=True
        ).as_matrix()
        vertices = (sphere.vertices * np.asarray(radii)) @ local_rotation.T
        vertices += np.asarray(center)
        faces = np.asarray(sphere.faces, dtype=np.int32)
        materials = np.full(len(faces), material, dtype=np.int16)
        return vertices, faces, materials

    @classmethod
    def _build_mouse_mesh(
        cls,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble closed cartoon ellipsoids in +X-forward body coordinates."""
        parts = [
            cls._ellipsoid((-1.8, 0.0, 0.0), (6.6, 5.0, 4.8), 0, 3),
            cls._ellipsoid((4.1, 0.0, -0.65), (5.2, 3.65, 3.1), 0, 3),
            cls._ellipsoid((6.2, 0.0, -0.55), (3.4, 2.55, 2.25), 5, 2),
            # Outer and inner ears. Flattened Y radii make upright pinnae.
            cls._ellipsoid(
                (-2.7, -4.25, 3.65),
                (2.35, 1.05, 3.0),
                0,
                2,
                (10.0, -12.0, -6.0),
            ),
            cls._ellipsoid(
                (-2.7, 4.25, 3.65),
                (2.35, 1.05, 3.0),
                0,
                2,
                (-10.0, -12.0, 6.0),
            ),
            cls._ellipsoid((-2.05, -5.28, 3.75), (1.48, 0.18, 2.05), 1, 2),
            cls._ellipsoid((-2.05, 5.28, 3.75), (1.48, 0.18, 2.05), 1, 2),
            # Both eyes are true opaque 3D forms and disappear behind the head.
            cls._ellipsoid((2.55, -3.22, 1.65), (0.78, 0.55, 0.72), 2, 2),
            cls._ellipsoid((2.55, 3.22, 1.65), (0.78, 0.55, 0.72), 2, 2),
            cls._ellipsoid((2.86, -3.68, 2.0), (0.19, 0.13, 0.19), 3, 1),
            cls._ellipsoid((2.86, 3.68, 2.0), (0.19, 0.13, 0.19), 3, 1),
            cls._ellipsoid((9.35, 0.0, -0.55), (0.9, 1.05, 0.78), 4, 2),
        ]
        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        materials: list[np.ndarray] = []
        offset = 0
        for part_vertices, part_faces, part_materials in parts:
            vertices.append(part_vertices)
            faces.append(part_faces + offset)
            materials.append(part_materials)
            offset += len(part_vertices)
        all_vertices = np.vstack(vertices).astype(np.float64)
        all_faces = np.vstack(faces).astype(np.int32)
        all_materials = np.concatenate(materials)
        triangles = all_vertices[all_faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(normal_lengths, np.finfo(float).eps)
        return all_vertices, all_faces, normals, all_materials

    def set_matrix(self, matrix: np.ndarray) -> None:
        self._matrix = self._world_yaw_matrix @ np.asarray(matrix, dtype=np.float64)
        self.update()

    @staticmethod
    def _draw_arrow(
        painter: QtGui.QPainter,
        origin: QtCore.QPointF,
        vector: QtCore.QPointF,
        color: QtGui.QColor,
        label: str,
    ) -> None:
        """Draw one labeled axis arrow."""
        endpoint = origin + vector
        painter.setPen(QtGui.QPen(color, 3.0))
        painter.setBrush(color)
        painter.drawLine(origin, endpoint)
        length = np.hypot(vector.x(), vector.y())
        if length > 1.0:
            unit = QtCore.QPointF(vector.x() / length, vector.y() / length)
            normal = QtCore.QPointF(-unit.y(), unit.x())
            base = endpoint - unit * 8.0
            painter.drawPolygon(
                QtGui.QPolygonF([endpoint, base + normal * 3.5, base - normal * 3.5])
            )
        painter.drawText(endpoint + QtCore.QPointF(4.0, -4.0), label)

    def _draw_axis_inset(self, painter: QtGui.QPainter) -> None:
        origin = QtCore.QPointF(48.0, self.height() - 48.0)
        colors = ("#ff5c57", "#5af78e", "#57c7ff")
        for axis, (color, label) in enumerate(zip(colors, "XYZ", strict=True)):
            world_vector = self._matrix[:, axis]
            vector = QtCore.QPointF(
                -24.0 * float(world_vector @ self._screen_right),
                -24.0 * float(world_vector @ self._screen_up),
            )
            self._draw_arrow(
                painter,
                origin,
                vector,
                QtGui.QColor(color),
                label,
            )

    def _draw_reference_plane(
        self,
        painter: QtGui.QPainter,
        center: QtCore.QPointF,
        scale: float,
    ) -> None:
        """Draw a faint fixed world-horizontal plane behind the mouse."""
        extent = 10.5
        height = -5.2

        def project(world: np.ndarray) -> QtCore.QPointF:
            return QtCore.QPointF(
                center.x() - scale * float(world @ self._screen_right),
                center.y() - scale * float(world @ self._screen_up),
            )

        corners = np.array(
            [
                [-extent, -extent, height],
                [extent, -extent, height],
                [extent, extent, height],
                [-extent, extent, height],
            ]
        )
        painter.setPen(QtGui.QPen(QtGui.QColor(112, 184, 205, 38), 1.0))
        painter.setBrush(QtGui.QColor(112, 184, 205, 18))
        painter.drawPolygon(QtGui.QPolygonF([project(point) for point in corners]))
        for coordinate in np.linspace(-extent, extent, 7):
            painter.drawLine(
                project(np.array([-extent, coordinate, height])),
                project(np.array([extent, coordinate, height])),
            )
            painter.drawLine(
                project(np.array([coordinate, -extent, height])),
                project(np.array([coordinate, extent, height])),
            )

    def _draw_world_compass(self, painter: QtGui.QPainter) -> None:
        """Draw fixed NED North and East directions in the model viewport."""
        origin = QtCore.QPointF(self.width() - 52.0, self.height() - 48.0)
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 226, 235, 90), 1.0))
        painter.setBrush(QtGui.QColor(8, 11, 16, 145))
        painter.drawEllipse(origin, 35.0, 35.0)
        for world_vector, color, label in (
            (np.array([1.0, 0.0, 0.0]), "#f1fa8c", "N"),
            (np.array([0.0, 1.0, 0.0]), "#ff9f43", "E"),
        ):
            vector = QtCore.QPointF(
                -28.0 * float(world_vector @ self._screen_right),
                -28.0 * float(world_vector @ self._screen_up),
            )
            self._draw_arrow(
                painter,
                origin,
                vector,
                QtGui.QColor(color),
                label,
            )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#11161f"))
        center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0 + 5.0)
        scale = min(self.width(), self.height()) / 24.0
        self._draw_reference_plane(painter, center, scale)

        rotated = self._vertices @ self._matrix.T
        horizontal = -(rotated @ self._screen_right)
        vertical = rotated @ self._screen_up
        pixel_x = np.rint(center.x() + horizontal * scale).astype(np.int32)
        pixel_y = np.rint(center.y() - vertical * scale).astype(np.int32)
        projected = np.column_stack((pixel_x, pixel_y))[self._faces]
        rotated_normals = self._face_normals @ self._matrix.T
        facing = rotated_normals @ self._camera > 0.0
        visible = projected[facing]
        material_ids = self._face_materials[facing]
        face_depth = (rotated @ self._camera)[self._faces].mean(axis=1)[facing]
        key_light = self._camera + np.array([0.0, 0.0, 0.8])
        key_light /= np.linalg.norm(key_light)
        lighting = np.clip(rotated_normals[facing] @ key_light, 0.0, 1.0)

        raster = np.zeros((self.height(), self.width(), 4), dtype=np.uint8)
        # Every primitive first contributes to a fully opaque silhouette.
        cv2.fillPoly(raster, projected, (125, 119, 116, 255), lineType=cv2.LINE_8)
        depth_count = 36
        shade_count = 5
        span = float(np.ptp(face_depth))
        depth_ids = np.minimum(
            ((face_depth - face_depth.min()) / max(span, 1e-9) * depth_count).astype(
                np.int32
            ),
            depth_count - 1,
        )
        shade_ids = np.minimum(
            (lighting * shade_count).astype(np.int32), shade_count - 1
        )
        group_ids = (
            depth_ids * (len(self._MATERIALS) * shade_count)
            + material_ids * shade_count
            + shade_ids
        )
        for group in np.unique(group_ids):
            selected = group_ids == group
            material = int((group // shade_count) % len(self._MATERIALS))
            shade = int(group % shade_count)
            brightness = 0.56 + 0.44 * shade / (shade_count - 1)
            rgb = np.clip(
                self._MATERIALS[material, :3] * brightness, 0.0, 255.0
            ).astype(np.uint8)
            cv2.fillPoly(
                raster,
                visible[selected],
                (*rgb.tolist(), 255),
                lineType=cv2.LINE_8,
            )
        mask = raster[:, :, 3]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            raster,
            contours,
            -1,
            (56, 52, 52, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        image = QtGui.QImage(
            raster.data,
            self.width(),
            self.height(),
            raster.strides[0],
            QtGui.QImage.Format.Format_RGBA8888,
        )
        painter.drawImage(0, 0, image)
        self._draw_axis_inset(painter)
        self._draw_world_compass(painter)
        painter.setPen(QtGui.QColor("#d8dee9"))
        painter.drawText(12, 22, "Procedural 3D cartoon mouse • opaque")


class FrameDecoder(QtCore.QThread):
    """Decode and cache resized video frames outside the GUI thread."""

    frame_ready = QtCore.Signal(int, object)
    decode_error = QtCore.Signal(str)

    def __init__(
        self,
        video_path: Path,
        display_size: tuple[int, int] = (1000, 620),
        cache_megabytes: int = 256,
        prefetch_frames: int = 12,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._display_size = display_size
        self._cache_limit = cache_megabytes * 1024 * 1024
        self._prefetch_frames = prefetch_frames
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._condition = threading.Condition()
        self._requested_index: int | None = None
        self._stopping = False

    def request(self, index: int) -> None:
        """Keep only the latest frame request to avoid a seek backlog."""
        with self._condition:
            self._requested_index = index
            self._condition.notify()

    def stop(self) -> None:
        """Stop decoding and wait for the capture to close cleanly."""
        with self._condition:
            self._stopping = True
            self._condition.notify()
        self.wait(3000)

    def _add_cache(self, index: int, frame: np.ndarray) -> None:
        previous = self._cache.pop(index, None)
        if previous is not None:
            self._cache_bytes -= previous.nbytes
        self._cache[index] = frame
        self._cache_bytes += frame.nbytes
        while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
            _, removed = self._cache.popitem(last=False)
            self._cache_bytes -= removed.nbytes

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        max_width, max_height = self._display_size
        factor = min(max_width / width, max_height / height, 1.0)
        if factor < 1.0:
            frame = cv2.resize(
                frame,
                (round(width * factor), round(height * factor)),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def run(self) -> None:
        """Own the OpenCV capture exclusively in this worker thread."""
        capture = cv2.VideoCapture(str(self._video_path))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        if not capture.isOpened():
            self.decode_error.emit(f"Cannot open video: {self._video_path}")
            return
        capture_next = 0
        try:
            while True:
                with self._condition:
                    while self._requested_index is None and not self._stopping:
                        self._condition.wait()
                    if self._stopping:
                        return
                    index = self._requested_index
                    self._requested_index = None

                cached = self._cache.get(index)
                if cached is not None:
                    self._cache.move_to_end(index)
                    self.frame_ready.emit(index, cached)
                    continue

                if index != capture_next:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    self.decode_error.emit(f"Could not decode video frame {index + 1}.")
                    continue
                capture_next = index + 1
                prepared = self._prepare_frame(frame)
                self._add_cache(index, prepared)
                self.frame_ready.emit(index, prepared)

                # Sequential read-ahead makes normal playback memory-backed.
                for ahead in range(self._prefetch_frames):
                    with self._condition:
                        if self._stopping or self._requested_index is not None:
                            break
                    next_index = index + ahead + 1
                    if next_index in self._cache:
                        continue
                    if next_index != capture_next:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, next_index)
                    ok, frame = capture.read()
                    if not ok:
                        break
                    capture_next = next_index + 1
                    self._add_cache(next_index, self._prepare_frame(frame))
        finally:
            capture.release()


class InteractiveVideoLabel(QtWidgets.QWidget):
    """Zoomable video canvas centered automatically on a tracked position."""

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self._pixmap = QtGui.QPixmap()
        self._zoom = 1.0
        self._focus_position: QtCore.QPointF | None = None
        self._compass: list[tuple[QtCore.QPointF, QtGui.QColor, str]] = []

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:  # noqa: N802
        """Replace the current frame without changing the view transform."""
        self._pixmap = pixmap
        self.update()

    def set_focus_position(self, x: float, y: float) -> None:
        """Set the tracked source-pixel position used as the zoom center."""
        self._focus_position = QtCore.QPointF(float(x), float(y))
        self.update()

    def set_compass(
        self,
        directions: list[tuple[QtCore.QPointF, QtGui.QColor, str]],
    ) -> None:
        """Set fixed overlay arrows in normalized screen coordinates."""
        self._compass = directions
        self.update()

    def _fit_scale(self) -> float:
        if self._pixmap.isNull():
            return 1.0
        return min(
            self.width() / max(1, self._pixmap.width()),
            self.height() / max(1, self._pixmap.height()),
        )

    def _display_size(self) -> QtCore.QSizeF:
        scale = self._fit_scale() * self._zoom
        return QtCore.QSizeF(
            self._pixmap.width() * scale,
            self._pixmap.height() * scale,
        )

    def _target_rect(self) -> QtCore.QRectF:
        size = self._display_size()
        center = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        if self._focus_position is not None and self._zoom > 1.001:
            scale = self._fit_scale() * self._zoom
            return QtCore.QRectF(
                center.x() - self._focus_position.x() * scale,
                center.y() - self._focus_position.y() * scale,
                size.width(),
                size.height(),
            )
        return QtCore.QRectF(
            center.x() - size.width() / 2.0,
            center.y() - size.height() / 2.0,
            size.width(),
            size.height(),
        )

    def reset_view(self) -> None:
        """Return to fit-to-window mode."""
        self._zoom = 1.0
        self.update()

    @staticmethod
    def _draw_arrow(
        painter: QtGui.QPainter,
        origin: QtCore.QPointF,
        direction: QtCore.QPointF,
        color: QtGui.QColor,
        label: str,
    ) -> None:
        endpoint = origin + direction * 31.0
        painter.setPen(QtGui.QPen(color, 3.0))
        painter.setBrush(color)
        painter.drawLine(origin, endpoint)
        normal = QtCore.QPointF(-direction.y(), direction.x())
        base = endpoint - direction * 8.0
        painter.drawPolygon(
            QtGui.QPolygonF([endpoint, base + normal * 3.5, base - normal * 3.5])
        )
        painter.drawText(endpoint + QtCore.QPointF(4.0, -4.0), label)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QtGui.QColor("#080b10"))
        if not self._pixmap.isNull():
            painter.drawPixmap(self._target_rect(), self._pixmap, self._pixmap.rect())

        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self._compass:
            origin = QtCore.QPointF(58.0, self.height() - 58.0)
            painter.setPen(QtGui.QPen(QtGui.QColor(235, 239, 245, 130), 1.0))
            painter.setBrush(QtGui.QColor(8, 11, 16, 150))
            painter.drawEllipse(origin, 38.0, 38.0)
            for direction, color, label in self._compass:
                self._draw_arrow(painter, origin, direction, color, label)

        if self._zoom > 1.001:
            painter.setPen(QtGui.QColor(235, 239, 245, 210))
            painter.setBrush(QtGui.QColor(8, 11, 16, 150))
            badge = QtCore.QRectF(self.width() - 76.0, 12.0, 64.0, 25.0)
            painter.drawRoundedRect(badge, 5.0, 5.0)
            painter.drawText(
                badge,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"{self._zoom:.2f}×",
            )
        painter.end()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        if self._pixmap.isNull():
            event.ignore()
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0.0:
            event.accept()
            return
        new_zoom = float(np.clip(self._zoom * (1.18**steps), 1.0, 8.0))
        if np.isclose(new_zoom, self._zoom):
            event.accept()
            return
        self._zoom = new_zoom
        self.update()
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class FusionViewer(QtWidgets.QMainWindow):
    """Seekable synchronized video/fusion browser."""

    def __init__(
        self,
        result_path: Path,
        video_path: Path,
        timestamp_path: Path,
        timestamp_key: str | None,
        *,
        timestamp_segment_index: int = 1,
        timestamp_gap_seconds: float = 60.0,
        tracking_path: Path | None = None,
        tracking_bodypart: str = "bodyCenter1",
        tracking_min_likelihood: float = 0.5,
        video_cache_megabytes: int = 512,
        video_prefetch_frames: int = 24,
        video_display_size: tuple[int, int] = (1000, 620),
        plot_window_seconds: float = 2.0,
        cartoon_camera_azimuth_degrees: float = -37.5,
        cartoon_camera_elevation_degrees: float = 30.0,
        cartoon_world_yaw_alignment_degrees: float = 0.0,
        video_compass_rotation_degrees: float = -90.0,
        show_video_compass: bool = True,
        initial_playback_fps: float | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("IMU Fusion Inspector")
        self.resize(1380, 880)
        self._time_s, self._euler, self._quaternion_aligned = read_fusion_view_data(
            result_path
        )
        self._validate_result()
        probe = cv2.VideoCapture(str(video_path))
        if not probe.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        self._frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps = float(probe.get(cv2.CAP_PROP_FPS)) or 30.0
        self._initial_playback_fps = (
            self._video_fps
            if initial_playback_fps is None
            else float(initial_playback_fps)
        )
        self._video_frame_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_frame_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()
        self._timestamps, timestamp_name = load_video_timestamps(
            timestamp_path,
            self._frame_count,
            timestamp_key,
            segment_index=timestamp_segment_index,
            gap_seconds=timestamp_gap_seconds,
        )
        timestamp_duration = float(self._timestamps[-1] - self._timestamps[0])
        self._timeline_fps = (
            (len(self._timestamps) - 1) / timestamp_duration
            if timestamp_duration > 0.0
            else self._video_fps
        )
        self._available_frames = min(self._frame_count, len(self._timestamps))
        self._tracking_positions: np.ndarray | None = None
        if tracking_path is not None:
            self._tracking_positions = load_dlc_positions(
                tracking_path,
                bodypart=tracking_bodypart,
                min_likelihood=tracking_min_likelihood,
            )
            if len(self._tracking_positions) < self._available_frames:
                raise ValueError(
                    "DLC tracking has fewer frames than the synchronized video: "
                    f"{len(self._tracking_positions):,} < {self._available_frames:,}."
                )
        if self._available_frames <= 0:
            raise ValueError("Video and timestamp file have no usable frames.")
        self._frame_index = 0
        self._requested_frame_index = 0
        self._playing = False
        self._slider_dragging = False
        self._plot_window_seconds = max(0.1, float(plot_window_seconds))
        self._cartoon_camera_azimuth_degrees = float(cartoon_camera_azimuth_degrees)
        self._cartoon_camera_elevation_degrees = float(
            cartoon_camera_elevation_degrees
        )
        self._cartoon_world_yaw_alignment_degrees = float(
            cartoon_world_yaw_alignment_degrees
        )
        self._display_yaw_matrix = Rotation.from_euler(
            "Z", self._cartoon_world_yaw_alignment_degrees, degrees=True
        ).as_matrix()
        video_compass_rotation = Rotation.from_euler(
            "Z", float(video_compass_rotation_degrees), degrees=True
        ).as_matrix()
        self._video_compass_matrix = video_compass_rotation @ self._display_yaw_matrix
        self._show_video_compass = bool(show_video_compass)
        self._current_display_matrix = np.eye(3)
        self._last_frame: np.ndarray | None = None
        self._build_ui(timestamp_name, video_path.name)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._seek_timer = QtCore.QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(80)
        self._seek_timer.timeout.connect(self._request_slider_preview)
        self._pending_seek_index = 0
        self._decoder = FrameDecoder(
            video_path,
            display_size=video_display_size,
            cache_megabytes=video_cache_megabytes,
            prefetch_frames=video_prefetch_frames,
        )
        self._decoder.frame_ready.connect(self._apply_frame)
        self._decoder.decode_error.connect(self._show_decode_error)
        self._decoder.start()
        self._play_clock = QtCore.QElapsedTimer()
        self._play_anchor_time_s = 0.0
        self._show_frame(0)

    def _validate_result(self) -> None:
        rows = len(self._time_s)
        if rows < 2:
            raise ValueError("Fusion result must contain at least two samples.")
        if self._euler.shape != (rows, 3):
            raise ValueError("Fusion Euler dataset must have shape (N, 3).")
        if self._quaternion_aligned.shape != (rows, 4):
            raise ValueError("Aligned quaternion dataset must have shape (N, 4).")

    def _build_ui(self, timestamp_name: str, video_name: str) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._video = InteractiveVideoLabel()
        self._video.setMinimumSize(720, 420)
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
        top.setStretchFactor(0, 5)
        top.setStretchFactor(1, 1)
        root.addWidget(top, 4)

        pg.setConfigOptions(antialias=False, background="#11161f", foreground="#d8dee9")
        plot_row = QtWidgets.QWidget()
        plot_layout = QtWidgets.QHBoxLayout(plot_row)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)
        # Keep only the short synchronized window in the plot. Updating about
        # 200 native-rate samples is both smoother and more informative than
        # drawing a heavily downsampled multi-hour session.
        self._plots: list[pg.PlotWidget] = []
        self._plot_curves: list[pg.PlotDataItem] = []
        self._cursors: list[pg.InfiniteLine] = []
        for column, _axis, color in (
            ("roll", 0, "#ff5c57"),
            ("yaw", 1, "#5af78e"),
            ("pitch", 2, "#57c7ff"),
        ):
            plot = pg.PlotWidget()
            plot.setLabel("bottom", "Time", units="s")
            plot.setLabel("left", column.capitalize(), units="deg")
            plot.showGrid(x=True, y=True, alpha=0.2)
            plot.setYRange(-180.0, 180.0, padding=0.0)
            plot.setTitle(f"{column.capitalize()}: 0.0°", color=color, size="12pt")
            self._plots.append(plot)
            self._plot_curves.append(
                plot.plot(
                    [],
                    [],
                    pen=pg.mkPen(color, width=2),
                    skipFiniteCheck=True,
                )
            )
            cursor = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen("#f1fa8c", width=2),
            )
            self._cursors.append(cursor)
            plot.addItem(cursor)
            plot_layout.addWidget(plot)
        root.addWidget(plot_row, 2)

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
            f"{video_name}  •  MAT array: {timestamp_name}  •  "
            "Space: play/pause  •  ←/→: frame"
        )
        root.addWidget(self._status)
        self.setCentralWidget(central)

    def _configure_frame_button(
        self, button: QtWidgets.QPushButton, direction: int
    ) -> None:
        """Make a frame button accelerate naturally while it is held."""
        button.setAutoRepeat(True)
        button.setAutoRepeatDelay(350)
        button.setAutoRepeatInterval(35)
        button.clicked.connect(lambda: self._step_frame(direction))

    def _speed_factor(self) -> float:
        return float(self._speed.currentText().removesuffix("×"))

    def _create_fps_control(self) -> QtWidgets.QDoubleSpinBox:
        """Create the editable playback frame-rate control."""
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(1.0, 240.0)
        control.setDecimals(2)
        control.setSingleStep(1.0)
        control.setKeyboardTracking(False)
        control.setValue(self._initial_playback_fps)
        control.setToolTip(
            f"Playback frames per second (video source: {self._video_fps:.3f} FPS)"
        )
        control.valueChanged.connect(self._update_timer)
        return control

    def _playback_fps(self) -> float:
        """Return the user-selected base playback frame rate."""
        return float(self._fps.value())

    def _update_timer(self) -> None:
        if self._playing:
            self._play_anchor_time_s = self._timestamps[self._frame_index]
            self._play_clock.restart()
        effective_fps = self._playback_fps() * self._speed_factor()
        interval_ms = max(1, round(1000.0 / effective_fps))
        if self._playing:
            self._timer.start(interval_ms)

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self._play.setText("⏸ Pause" if self._playing else "▶ Play")
        if self._playing:
            self._play_anchor_time_s = self._timestamps[self._frame_index]
            self._play_clock.start()
            self._update_timer()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        playback_time_scale = (
            self._playback_fps() / self._timeline_fps * self._speed_factor()
        )
        target_time = self._play_anchor_time_s + (
            self._play_clock.elapsed() / 1000.0 * playback_time_scale
        )
        target_index = int(np.searchsorted(self._timestamps, target_time))
        target_index = max(self._frame_index + 1, target_index)
        if target_index >= self._available_frames:
            self._toggle_play()
            return
        self._show_frame(target_index)

    def _begin_slider_drag(self) -> None:
        """Freeze playback and prevent decoded frames from moving the thumb."""
        self._slider_dragging = True
        self._seek_timer.stop()
        self._pending_seek_index = self._slider.value()
        if self._playing:
            self._toggle_play()

    def _schedule_slider_seek(self, requested: int) -> None:
        """Preview fusion state and debounce video decoding during a drag."""
        self._pending_seek_index = int(requested)
        # Invalidate any older decode immediately so it cannot repaint a stale
        # frame while the thumb has already moved elsewhere.
        self._requested_frame_index = self._pending_seek_index
        self._update_fusion_overlay(self._pending_seek_index, render_video=False)
        self._seek_timer.start()

    def _commit_slider_seek(self) -> None:
        self._seek_timer.stop()
        self._slider_dragging = False
        self._show_frame(self._pending_seek_index)

    def _request_slider_preview(self) -> None:
        """Decode a settled drag position without releasing the slider."""
        self._show_frame(self._pending_seek_index)

    def _step_frame(self, direction: int) -> None:
        """Request the next frame relative to the latest outstanding request."""
        if self._playing:
            self._toggle_play()
        self._show_frame(self._requested_frame_index + direction)

    def _show_frame(self, requested: int) -> None:
        index = int(np.clip(requested, 0, self._available_frames - 1))
        self._requested_frame_index = index
        self._decoder.request(index)

    @QtCore.Slot(int, object)
    def _apply_frame(self, index: int, image: np.ndarray) -> None:
        """Apply decoded pixels and synchronized state on the main UI thread."""
        if index != self._requested_frame_index:
            return
        self._last_frame = image
        self._frame_index = index
        if not self._slider_dragging:
            self._slider.blockSignals(True)
            self._slider.setValue(index)
            self._slider.blockSignals(False)
        self._update_fusion_overlay(index)

    def _render_video_frame(self, image: np.ndarray, frame_index: int) -> None:
        """Scale a video frame and add a fixed world North/East compass."""
        qt_image = QtGui.QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        )
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        if self._tracking_positions is not None:
            position = self._tracking_positions[frame_index]
            scale_x = image.shape[1] / max(1, self._video_frame_width)
            scale_y = image.shape[0] / max(1, self._video_frame_height)
            self._video.set_focus_position(
                position[0] * scale_x,
                position[1] * scale_y,
            )
        compass: list[tuple[QtCore.QPointF, QtGui.QColor, str]] = []
        if self._show_video_compass:
            for world_vector, color, label in (
                (np.array([1.0, 0.0, 0.0]), "#f1fa8c", "N"),
                (np.array([0.0, 1.0, 0.0]), "#ff9f43", "E"),
            ):
                # The video camera is fixed in the world. Only the constant
                # video-to-IMU yaw alignment rotates its compass.
                display_world = self._video_compass_matrix @ world_vector
                horizontal = display_world[:2]
                norm = float(np.linalg.norm(horizontal))
                if norm < 1e-9:
                    continue
                horizontal /= norm
                compass.append(
                    (
                        QtCore.QPointF(
                            float(horizontal[1]),
                            -float(horizontal[0]),
                        ),
                        QtGui.QColor(color),
                        label,
                    )
                )
        self._video.set_compass(compass)
        self._video.setPixmap(pixmap)

    def _update_fusion_overlay(self, index: int, *, render_video: bool = True) -> None:
        """Update plots and the 3D mouse without requesting a video decode."""
        fusion_time = self._timestamps[index] + self._offset.value()
        right = int(
            np.clip(
                np.searchsorted(self._time_s, fusion_time), 0, len(self._time_s) - 1
            )
        )
        left = max(0, right - 1)
        imu_index = (
            left
            if abs(self._time_s[left] - fusion_time)
            <= abs(self._time_s[right] - fusion_time)
            else right
        )
        for cursor in self._cursors:
            cursor.setValue(fusion_time)
        half_window = self._plot_window_seconds / 2.0
        window_start = fusion_time - half_window
        window_end = fusion_time + half_window
        data_start = float(self._time_s[0])
        data_end = float(self._time_s[-1])
        if window_start < data_start:
            window_start = data_start
            window_end = min(data_end, data_start + self._plot_window_seconds)
        elif window_end > data_end:
            window_end = data_end
            window_start = max(data_start, data_end - self._plot_window_seconds)
        first = int(np.searchsorted(self._time_s, window_start, side="left"))
        last = int(np.searchsorted(self._time_s, window_end, side="right"))
        plot_time = self._time_s[first:last]
        wrapped_degrees = (np.rad2deg(self._euler[first:last]) + 180.0) % 360.0 - 180.0
        for axis, (plot, curve) in enumerate(
            zip(self._plots, self._plot_curves, strict=True)
        ):
            plot_values = wrapped_degrees[:, axis].copy()
            wrap_jumps = np.abs(np.diff(plot_values)) > 180.0
            plot_values[1:][wrap_jumps] = np.nan
            curve.setData(plot_time, plot_values)
            plot.setXRange(window_start, window_end, padding=0.0)
        q_wxyz = self._quaternion_aligned[imu_index]
        matrix = Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()
        self._current_display_matrix = self._display_yaw_matrix @ matrix
        self._orientation.set_matrix(matrix)
        if render_video and self._last_frame is not None:
            self._render_video_frame(self._last_frame, index)
        angles_degrees = (np.rad2deg(self._euler[imu_index]) + 180.0) % 360.0 - 180.0
        for name, color, plot, value in zip(
            ("Roll", "Yaw", "Pitch"),
            ("#ff5c57", "#5af78e", "#57c7ff"),
            self._plots,
            angles_degrees,
            strict=True,
        ):
            plot.setTitle(f"{name}: {value:+.1f}°", color=color, size="12pt")
        roll, yaw, pitch = angles_degrees
        self._status.setText(
            f"Frame {index + 1:,}/{self._available_frames:,}  |  "
            f"video {self._timestamps[index]:.3f} s  |  "
            f"fusion {fusion_time:.3f} s  |  IMU sample {imu_index:,}  |  "
            f"R/Y/P {roll:+.1f}°, {yaw:+.1f}°, {pitch:+.1f}°"
        )

    @QtCore.Slot(str)
    def _show_decode_error(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._toggle_play()
        elif event.key() == QtCore.Qt.Key.Key_Left:
            self._step_frame(-1)
        elif event.key() == QtCore.Qt.Key.Key_Right:
            self._step_frame(1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._timer.stop()
        self._decoder.stop()
        super().closeEvent(event)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path, help="Fused result HDF5")
    parser.add_argument("--video", required=True, type=Path, help="Video file")
    parser.add_argument(
        "--timestamps", required=True, type=Path, help="MAT timestamp file"
    )
    parser.add_argument(
        "--timestamp-key",
        help="Optional nested MAT key, e.g. behavior.timestamps_corrected",
    )
    parser.add_argument(
        "--tracking",
        type=Path,
        help="Optional DeepLabCut CSV used to center the zoomed video",
    )
    parser.add_argument("--tracking-bodypart", default="bodyCenter1")
    parser.add_argument("--tracking-min-likelihood", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Launch the desktop viewer."""
    args = _parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FusionViewer(
        args.result,
        args.video,
        args.timestamps,
        args.timestamp_key,
        tracking_path=args.tracking,
        tracking_bodypart=args.tracking_bodypart,
        tracking_min_likelihood=args.tracking_min_likelihood,
    )
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
