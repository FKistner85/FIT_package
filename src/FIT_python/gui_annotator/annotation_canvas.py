"""Interactive canvas widget for landmark annotation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Landmark:
    """Single annotated landmark with visibility flag."""

    x: float
    y: float
    visible: bool = True
import math

from PyQt5 import QtCore, QtGui, QtWidgets
from FIT_python.utils.transformations import TransformationPipeline

from FIT_python.soft_config import SOFT_CONFIG
from FIT_python import config

CFG = SOFT_CONFIG.get("gui_annotator", {})
REFERENCE_TEMPLATE_DIR = config.RAW_DIR / "reference_templates"
DISPLAY_SIZE = tuple(int(v) for v in CFG.get("display_size", [1280, 720]))


class AnnotationCanvas(QtWidgets.QLabel):
    """Interactive canvas allowing placement of landmarks and scale refs."""

    scale_changed = QtCore.pyqtSignal(float, float)
    landmarks_changed = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.landmarks: List[Landmark] = []
        self.crossfade_opacity = 0.3
        self.image: QtGui.QPixmap | None = None
        self.original_pixmap: QtGui.QPixmap | None = None
        self.references: list[QtGui.QPixmap] = []
        self._reference_pixmaps: list[QtGui.QPixmap] = []
        self.reference_names: list[str] = []
        self._reference_index: int | None = None
        if REFERENCE_TEMPLATE_DIR.exists():
            valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
            for p in sorted(REFERENCE_TEMPLATE_DIR.iterdir()):
                if p.is_file() and p.suffix.lower() in valid_ext:
                    pm = QtGui.QPixmap(str(p))
                    if not pm.isNull():
                        self._reference_pixmaps.append(pm)
                        self.reference_names.append(p.stem)
            if self._reference_pixmaps:
                placeholder = "Leftfront drawing Placeholder"
                if placeholder in self.reference_names:
                    self._reference_index = self.reference_names.index(placeholder)
                else:
                    self._reference_index = 0
        self.setMouseTracking(True)
        self.setStyleSheet("background: white")
        self._pixmap_offset: tuple[int, int] = (0, 0)
        self.rotation_refs: list[Tuple[float, float]] = []
        self.rotation_matrix = QtGui.QTransform()
        self.rotation_angle: float = 0.0
        self.total_rotation_angle: float = 0.0
        self.base_size: tuple[int, int] = (0, 0)
        self.rotation_history: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.transform_history: list[QtGui.QTransform] = []
        self._last_logged_zoom: float = 1.0
        self.orientation_points: list[tuple[float, float]] = []
        self._dragging_orientation: bool = False
        self.selected_orientation_idx: int | None = None
        self.mode: str = "rotate"
        self.scale_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._scale_tmp: list[tuple[float, float]] = []
        self._dragging_scale_handle: tuple[int, int] | None = None
        self.pixels_per_cm: float | None = None
        self.pixels_per_cm_sd: float | None = None
        self._zoom: float = 1.0
        self.selected_idx: int | None = None
        self._dragging: bool = False

    # configuration slots -------------------------------------------------
    def set_crossfade_opacity(self, value: float) -> None:
        """Set overlay opacity between 0 and 1."""
        self.crossfade_opacity = max(0.0, min(1.0, value))
        self.update()

    @property
    def reference_pixmaps(self) -> list[QtGui.QPixmap]:
        """Return the loaded reference templates."""
        return list(self._reference_pixmaps)

    def set_reference_index(self, idx: int) -> None:
        if 0 <= idx < len(self.references):
            self._reference_index = idx
            self.update()


    def set_mode(self, mode: str) -> None:
        """Set interaction mode (rotate, scale, or landmark)."""
        if mode not in {"rotate", "scale", "landmark"}:
            return
        if mode == self.mode:
            return
        self.mode = mode
        self.rotation_refs = []
        self._scale_tmp = []
        self.update()

    def load_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        """Load ``pixmap`` and rescale to the current widget size."""
        self.original_pixmap = pixmap
        self.landmarks = []
        self.landmarks_changed.emit()
        self.rotation_refs = []
        self.rotation_matrix = QtGui.QTransform()
        self.rotation_angle = 0.0
        self.total_rotation_angle = 0.0
        self.base_size = (pixmap.width(), pixmap.height())
        self.rotation_history = []
        self.transform_history = []
        self._last_logged_zoom = 1.0
        self.orientation_points = []
        self.scale_pairs = []
        self._scale_tmp = []
        self.pixels_per_cm = None
        self.pixels_per_cm_sd = None
        self.scale_changed.emit(float("nan"), float("nan"))
        self.mode = "rotate"
        self._zoom = 1.0
        self._rescale_pixmaps()
        self.update()

    # Qt event handlers -------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            return

        w = max(self.image.width() if self.image else self.width(), 1)
        h = max(self.image.height() if self.image else self.height(), 1)
        ox, oy = self._pixmap_offset
        x = event.x() - ox
        y = event.y() - oy
        if x < 0 or y < 0 or x > w or y > h:
            return
        x_norm = x / w
        y_norm = y / h

        if self.mode == "scale":
            threshold = 6
            for idx, (p1, p2) in enumerate(self.scale_pairs):
                if abs(p1[0] - x_norm) * w < threshold and abs(p1[1] - y_norm) * h < threshold:
                    self._dragging_scale_handle = (idx, 0)
                    return
                if abs(p2[0] - x_norm) * w < threshold and abs(p2[1] - y_norm) * h < threshold:
                    self._dragging_scale_handle = (idx, 1)
                    return
            self._scale_tmp.append((x_norm, y_norm))
            if len(self._scale_tmp) == 2:
                self.scale_pairs.append((self._scale_tmp[0], self._scale_tmp[1]))
                self._scale_tmp = []
                self._update_scale()
            self.update()
            return

        if self.mode == "rotate":
            threshold = 6
            for idx, (oxp, oyp) in enumerate(self.orientation_points):
                if abs(oxp - x_norm) * w < threshold and abs(oyp - y_norm) * h < threshold:
                    self.selected_orientation_idx = idx
                    self._dragging_orientation = True
                    return
            self.rotation_refs.append((x_norm, y_norm))
            if len(self.rotation_refs) == 2:
                self._apply_rotation()
            return

        # landmark interaction
        threshold = 6
        found_idx: int | None = None
        for idx, lm in enumerate(self.landmarks):
            if abs(lm.x - x_norm) * w < threshold and abs(lm.y - y_norm) * h < threshold:
                found_idx = idx
                break

        if event.button() == QtCore.Qt.LeftButton:
            if found_idx is not None:
                self.selected_idx = found_idx
                self._dragging = True
                vis = self.landmarks[found_idx].visible
                self.landmarks[found_idx] = Landmark(x_norm, y_norm, vis)
                self.landmarks_changed.emit()
            else:
                self._dragging = False
                self.selected_idx = None
                self._add_or_update_landmark(x_norm, y_norm, True)
            self.update()
            return

        # right button
        self._dragging = False
        self.selected_idx = found_idx
        self._add_or_update_landmark(x_norm, y_norm, False)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        w = max(self.image.width() if self.image else self.width(), 1)
        h = max(self.image.height() if self.image else self.height(), 1)
        ox, oy = self._pixmap_offset
        x = event.x() - ox
        y = event.y() - oy
        if x < 0 or y < 0 or x > w or y > h:
            return
        x_norm = x / w
        y_norm = y / h

        threshold = 6
        found_idx: int | None = None
        for idx, lm in enumerate(self.landmarks):
            if abs(lm.x - x_norm) * w < threshold and abs(lm.y - y_norm) * h < threshold:
                found_idx = idx
                break

        if self._dragging_orientation and self.selected_orientation_idx is not None and self.mode == "rotate":
            self.orientation_points[self.selected_orientation_idx] = (x_norm, y_norm)
            self.update()
            return

        if self._dragging_scale_handle is not None and self.mode == "scale":
            pair_idx, handle_idx = self._dragging_scale_handle
            if 0 <= pair_idx < len(self.scale_pairs):
                p1, p2 = self.scale_pairs[pair_idx]
                if handle_idx == 0:
                    self.scale_pairs[pair_idx] = ((x_norm, y_norm), p2)
                else:
                    self.scale_pairs[pair_idx] = (p1, (x_norm, y_norm))
                self._update_scale()
                self.update()
            return

        if self._dragging and self.selected_idx is not None and self.mode == "landmark":
            vis = self.landmarks[self.selected_idx].visible
            self.landmarks[self.selected_idx] = Landmark(x_norm, y_norm, vis)
            self.landmarks_changed.emit()
            self.update()
        self.selected_idx = found_idx

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
            self._dragging_orientation = False
            self._dragging_scale_handle = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == QtCore.Qt.Key_Delete and self.selected_idx is not None:
            if 0 <= self.selected_idx < len(self.landmarks):
                del self.landmarks[self.selected_idx]
                self.selected_idx = None
                self.landmarks_changed.emit()
                self.update()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        """Zoom the canvas in and out with the mouse wheel."""
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom *= 1.1
        else:
            self._zoom /= 1.1
        self._zoom = max(0.1, min(self._zoom, 10.0))
        self._rescale_pixmaps()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        if self.image:
            ox, oy = self._pixmap_offset
            painter.drawPixmap(ox, oy, self.image)
        if (
            self._reference_index is not None
            and self.image
            and 0 <= self._reference_index < len(self.references)
        ):
            painter.setOpacity(self.crossfade_opacity)
            overlay = self.references[self._reference_index]
            painter.drawPixmap(ox, oy, overlay)
            painter.setOpacity(1.0)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 4))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        if self.image:
            w = self.image.width()
            h = self.image.height()
            for idx, lm in enumerate(self.landmarks):
                if not lm.visible:
                    continue
                point = QtCore.QPoint(ox + int(lm.x * w), oy + int(lm.y * h))
                if self._dragging and idx == self.selected_idx:
                    painter.setPen(QtGui.QPen(QtCore.Qt.blue, 4))
                    painter.drawEllipse(point, 6, 6)
                    painter.setPen(QtGui.QPen(QtCore.Qt.red, 4))
                painter.drawEllipse(point, 4, 4)
                painter.drawText(point + QtCore.QPoint(5, -5), str(idx + 1))
            painter.setPen(QtGui.QPen(QtCore.Qt.green, 2))
            for (x1, y1), (x2, y2) in self.scale_pairs:
                painter.drawLine(
                    ox + int(x1 * w), oy + int(y1 * h), ox + int(x2 * w), oy + int(y2 * h)
                )
                painter.drawEllipse(QtCore.QPoint(ox + int(x1 * w), oy + int(y1 * h)), 3, 3)
                painter.drawEllipse(QtCore.QPoint(ox + int(x2 * w), oy + int(y2 * h)), 3, 3)
            painter.setPen(QtGui.QPen(QtCore.Qt.darkYellow, 2))
            for (x1, y1), (x2, y2) in self.rotation_history:
                painter.drawLine(
                    ox + int(x1 * w), oy + int(y1 * h), ox + int(x2 * w), oy + int(y2 * h)
                )
                painter.drawEllipse(QtCore.QPoint(ox + int(x1 * w), oy + int(y1 * h)), 3, 3)
                painter.drawEllipse(QtCore.QPoint(ox + int(x2 * w), oy + int(y2 * h)), 3, 3)
            if self.mode == "rotate" and self.rotation_refs:
                painter.setPen(QtGui.QPen(QtCore.Qt.darkYellow, 2))
                if len(self.rotation_refs) == 2:
                    (x1, y1), (x2, y2) = self.rotation_refs
                    painter.drawLine(
                        ox + int(x1 * w), oy + int(y1 * h), ox + int(x2 * w), oy + int(y2 * h)
                    )
                    painter.drawEllipse(QtCore.QPoint(ox + int(x1 * w), oy + int(y1 * h)), 3, 3)
                    painter.drawEllipse(QtCore.QPoint(ox + int(x2 * w), oy + int(y2 * h)), 3, 3)
                else:
                    x, y = self.rotation_refs[0]
                    painter.drawEllipse(QtCore.QPoint(ox + int(x * w), oy + int(y * h)), 3, 3)
            if len(self.orientation_points) == 2:
                painter.setPen(QtGui.QPen(QtCore.Qt.magenta, 2))
                (ox1, oy1), (ox2, oy2) = self.orientation_points
                px1 = ox + int(ox1 * w)
                py1 = oy + int(oy1 * h)
                px2 = ox + int(ox2 * w)
                py2 = oy + int(oy2 * h)
                painter.drawLine(px1, py1, px2, py2)
                midx = (px1 + px2) / 2
                midy = (py1 + py2) / 2
                dx = px2 - px1
                dy = py2 - py1
                length = math.hypot(dx, dy)
                if length:
                    ux = -dy / length
                    uy = dx / length
                    px3 = int(midx - ux * length / 2)
                    py3 = int(midy - uy * length / 2)
                    px4 = int(midx + ux * length / 2)
                    py4 = int(midy + uy * length / 2)
                    painter.drawLine(px3, py3, px4, py4)
                painter.drawEllipse(QtCore.QPoint(px1, py1), 3, 3)
                painter.drawEllipse(QtCore.QPoint(px2, py2), 3, 3)
            if self.mode == "scale" and self._scale_tmp:
                x, y = self._scale_tmp[0]
                painter.drawEllipse(QtCore.QPoint(ox + int(x * w), oy + int(y * h)), 3, 3)
        painter.end()

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        w, h = self._screen_constrained_size()
        return QtCore.QSize(w, h)

    def _screen_constrained_size(self) -> Tuple[int, int]:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            sz = screen.availableGeometry().size()
            return min(DISPLAY_SIZE[0], sz.width()), min(
                DISPLAY_SIZE[1], sz.height()
            )
        return DISPLAY_SIZE

    # internal helpers ----------------------------------------------------
    def _rescale_pixmaps(self) -> None:
        """Scale loaded pixmaps to the current widget size."""
        if self.original_pixmap:
            if self._zoom != self._last_logged_zoom:
                ratio = self._zoom / self._last_logged_zoom
                self.transform_history.append(QtGui.QTransform().scale(ratio, ratio))
                self._last_logged_zoom = self._zoom
            target_w = int(self.width() * self._zoom)
            target_h = int(self.height() * self._zoom)
            self.image = self.original_pixmap.scaled(
                target_w,
                target_h,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self._pixmap_offset = (
                (self.width() - self.image.width()) // 2,
                (self.height() - self.image.height()) // 2,
            )
        else:
            self.image = None
            self._pixmap_offset = (0, 0)
        if self._reference_pixmaps:
            if self.image:
                self.references = [
                    pm.scaled(
                        self.image.width(),
                        self.image.height(),
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                    for pm in self._reference_pixmaps
                ]
            else:
                # Without a loaded image we cannot scale using its dimensions.
                # Use the original reference pixmaps without scaling.
                self.references = list(self._reference_pixmaps)
        else:
            self.references = []
        self._update_scale()

    def _add_or_update_landmark(self, x: float, y: float, visible: bool) -> None:
        """Insert or update a landmark near ``(x, y)``."""
        w = max(self.image.width() if self.image else self.width(), 1)
        h = max(self.image.height() if self.image else self.height(), 1)
        threshold = 6  # pixels
        for idx, lm in enumerate(self.landmarks):
            if abs(lm.x - x) * w < threshold and abs(lm.y - y) * h < threshold:
                self.landmarks[idx] = Landmark(x, y, visible)
                self.landmarks_changed.emit()
                return
        if len(self.landmarks) < 11:
            self.landmarks.append(Landmark(x, y, visible))
            self.landmarks_changed.emit()

    def _update_scale(self) -> None:
        """Recompute pixels-per-cm from stored scale pairs."""
        if len(self.scale_pairs) < 2:
            self.pixels_per_cm = None
            self.pixels_per_cm_sd = None
            self.scale_changed.emit(float("nan"), float("nan"))
            return

        w = max(self.image.width() if self.image else self.width(), 1)
        h = max(self.image.height() if self.image else self.height(), 1)
        distances = []
        for (x1, y1), (x2, y2) in self.scale_pairs:
            dx = (x1 - x2) * w
            dy = (y1 - y2) * h
            distances.append(math.hypot(dx, dy))

        mean = sum(distances) / len(distances)
        sd = 0.0
        if len(distances) > 1:
            var = sum((d - mean) ** 2 for d in distances) / (len(distances) - 1)
            sd = math.sqrt(var)

        self.pixels_per_cm = mean
        self.pixels_per_cm_sd = sd
        self.scale_changed.emit(mean, sd)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        self._rescale_pixmaps()
        self._update_scale()
        super().resizeEvent(event)

    def _apply_rotation(self) -> None:
        """Rotate loaded pixmap and existing landmarks using reference points."""
        if not self.original_pixmap or len(self.rotation_refs) != 2:
            return

        w = max(self.image.width() if self.image else self.width(), 1)
        h = max(self.image.height() if self.image else self.height(), 1)
        (x1, y1), (x2, y2) = self.rotation_refs
        p1 = QtCore.QPointF(x1 * w, y1 * h)
        p2 = QtCore.QPointF(x2 * w, y2 * h)

        angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
        self.rotation_angle = angle
        self.rotation_matrix = QtGui.QTransform().rotateRadians(-angle)
        self.transform_history.append(self.rotation_matrix)


        # store reference points in original orientation
        if self.base_size != (0, 0):
            inv = QtGui.QTransform().rotateRadians(self.total_rotation_angle)
            orig_p1 = inv.map(p1)
            orig_p2 = inv.map(p2)
            bw, bh = self.base_size
            self.rotation_history.append(
                ((orig_p1.x() / bw, orig_p1.y() / bh), (orig_p2.x() / bw, orig_p2.y() / bh))
            )

        rotated = self.original_pixmap.transformed(
            self.rotation_matrix, QtCore.Qt.SmoothTransformation
        )
        self.original_pixmap = rotated
        self.total_rotation_angle += angle

        rot_w = rotated.width()
        rot_h = rotated.height()
        p1_rot = self.rotation_matrix.map(p1)
        p2_rot = self.rotation_matrix.map(p2)
        self.orientation_points = [
            (p1_rot.x() / rot_w, p1_rot.y() / rot_h),
            (p2_rot.x() / rot_w, p2_rot.y() / rot_h),
        ]

        if self.landmarks:
            rot_w = rotated.width()
            rot_h = rotated.height()
            new_pts: list[Landmark] = []
            for lm in self.landmarks:
                pt = QtCore.QPointF(lm.x * w, lm.y * h)
                pt = self.rotation_matrix.map(pt)
                new_pts.append(
                    Landmark(pt.x() / rot_w, pt.y() / rot_h, lm.visible)
                )
            self.landmarks = new_pts
            self.landmarks_changed.emit()

        self.rotation_refs = []
        self.scale_pairs = []
        self._scale_tmp = []
        self.pixels_per_cm = None
        self.pixels_per_cm_sd = None
        self.scale_changed.emit(float("nan"), float("nan"))
        self._rescale_pixmaps()
        self.update()

    def get_unrotated_landmarks(self) -> list[Landmark]:
        """Return landmarks transformed back to the original orientation."""
        if not self.landmarks:
            return []
        if not self.image or self.base_size == (0, 0):
            return list(self.landmarks)

        pipeline = TransformationPipeline(self.transform_history)
        inverse = pipeline.inverted()
        w = self.image.width()
        h = self.image.height()
        bw, bh = self.base_size
        pts: list[Landmark] = []
        for lm in self.landmarks:
            pt = QtCore.QPointF(lm.x * w, lm.y * h)
            for t in inverse.transforms:
                pt = t.map(pt)
            pts.append(Landmark(pt.x() / bw, pt.y() / bh, lm.visible))
        return pts

    def get_unrotated_scale_pairs(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return stored scale pairs transformed to the original orientation."""
        if not self.scale_pairs or not self.image or self.base_size == (0, 0):
            return list(self.scale_pairs)

        pipeline = TransformationPipeline(self.transform_history)
        inverse = pipeline.inverted()
        w = self.image.width()
        h = self.image.height()
        bw, bh = self.base_size
        pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for (x1, y1), (x2, y2) in self.scale_pairs:
            p1 = QtCore.QPointF(x1 * w, y1 * h)
            p2 = QtCore.QPointF(x2 * w, y2 * h)
            for t in inverse.transforms:
                p1 = t.map(p1)
                p2 = t.map(p2)
            pairs.append(((p1.x() / bw, p1.y() / bh), (p2.x() / bw, p2.y() / bh)))
        return pairs

    def get_rotation_history(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return recorded rotation reference pairs in original orientation."""
        return list(self.rotation_history)

    def undo_last(self) -> None:
        """Remove the most recently added item depending on the mode."""
        if self.mode == "scale":
            if self._scale_tmp:
                self._scale_tmp.pop()
                self.update()
                return
            if self.scale_pairs:
                self.scale_pairs.pop()
                self._update_scale()
                self.update()
                return

        if self.mode == "rotate" and self.rotation_refs:
            self.rotation_refs.pop()
            self.update()
            return

        if self.landmarks:
            self.landmarks.pop()
            self.landmarks_changed.emit()
            self.update()
