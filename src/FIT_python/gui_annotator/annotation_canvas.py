"""Interactive canvas widget for landmark annotation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import math

from PyQt5 import QtCore, QtGui, QtWidgets

from FIT_python.soft_config import SOFT_CONFIG

CFG = SOFT_CONFIG.get("gui_annotator", {})
REFERENCE_TEMPLATE_DIR = Path(CFG.get("reference_template_dir", ""))
DISPLAY_SIZE = tuple(int(v) for v in CFG.get("display_size", [1280, 720]))


class AnnotationCanvas(QtWidgets.QLabel):
    """Interactive canvas allowing placement of landmarks and scale refs."""

    scale_changed = QtCore.pyqtSignal(float, float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.landmarks: List[Tuple[float, float]] = []
        self.crossfade_opacity = 0.3
        self.image: QtGui.QPixmap | None = None
        self.original_pixmap: QtGui.QPixmap | None = None
        self.references: list[QtGui.QPixmap] = []
        self._reference_pixmaps: list[QtGui.QPixmap] = []
        self.reference_names: list[str] = []
        self._reference_index: int | None = None
        if REFERENCE_TEMPLATE_DIR.exists():
            for p in sorted(REFERENCE_TEMPLATE_DIR.iterdir()):
                if p.is_file():
                    pm = QtGui.QPixmap(str(p))
                    self._reference_pixmaps.append(pm)
                    self.reference_names.append(p.stem)
            if self._reference_pixmaps:
                self._reference_index = 0
        self.setMouseTracking(True)
        self.rotation_refs: list[Tuple[float, float]] = []
        self.rotation_matrix = QtGui.QTransform()
        self.scale_mode = False
        self.scale_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._scale_tmp: list[tuple[float, float]] = []
        self.pixels_per_cm: float | None = None
        self.pixels_per_cm_sd: float | None = None

    # configuration slots -------------------------------------------------
    def set_crossfade_opacity(self, value: float) -> None:
        """Set overlay opacity between 0 and 1."""
        self.crossfade_opacity = max(0.0, min(1.0, value))
        self.update()

    def set_reference_index(self, idx: int) -> None:
        if 0 <= idx < len(self.references):
            self._reference_index = idx
            self.update()

    def set_scale_mode(self, active: bool) -> None:
        """Enable or disable selection of scale reference pairs."""
        self.scale_mode = active
        self._scale_tmp = []

    def load_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        """Load ``pixmap`` and rescale to the current widget size."""
        self.original_pixmap = pixmap
        self.landmarks = []
        self.rotation_refs = []
        self.rotation_matrix = QtGui.QTransform()
        self.scale_pairs = []
        self._scale_tmp = []
        self.pixels_per_cm = None
        self.pixels_per_cm_sd = None
        self.scale_changed.emit(float("nan"), float("nan"))
        self._rescale_pixmaps()
        self.update()

    # Qt event handlers -------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton:
            return

        w = max(self.width(), 1)
        h = max(self.height(), 1)

        if self.scale_mode:
            self._scale_tmp.append((event.x() / w, event.y() / h))
            if len(self._scale_tmp) == 2:
                self.scale_pairs.append((self._scale_tmp[0], self._scale_tmp[1]))
                self._scale_tmp = []
                self._update_scale()
            self.update()
            return

        if len(self.rotation_refs) < 2:
            self.rotation_refs.append((event.x() / w, event.y() / h))
            if len(self.rotation_refs) == 2:
                self._apply_rotation()
            return

        if len(self.landmarks) < 11:
            self.landmarks.append((event.x() / w, event.y() / h))
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        if self.image:
            painter.drawPixmap(0, 0, self.image)
        if (
            self._reference_index is not None
            and self.image
            and 0 <= self._reference_index < len(self.references)
        ):
            painter.setOpacity(self.crossfade_opacity)
            overlay = self.references[self._reference_index]
            painter.drawPixmap(0, 0, overlay)
            painter.setOpacity(1.0)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 4))
        if self.image:
            w = self.width()
            h = self.height()
            for x, y in self.landmarks:
                painter.drawEllipse(QtCore.QPoint(int(x * w), int(y * h)), 4, 4)
            painter.setPen(QtGui.QPen(QtCore.Qt.green, 2))
            for (x1, y1), (x2, y2) in self.scale_pairs:
                painter.drawLine(
                    int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                )
                painter.drawEllipse(QtCore.QPoint(int(x1 * w), int(y1 * h)), 3, 3)
                painter.drawEllipse(QtCore.QPoint(int(x2 * w), int(y2 * h)), 3, 3)
            if self.scale_mode and self._scale_tmp:
                x, y = self._scale_tmp[0]
                painter.drawEllipse(QtCore.QPoint(int(x * w), int(y * h)), 3, 3)
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
            self.image = self.original_pixmap.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        if self._reference_pixmaps:
            self.references = [
                pm.scaled(
                    self.width() // 2,
                    self.height() // 2,
                    QtCore.Qt.IgnoreAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                for pm in self._reference_pixmaps
            ]
        else:
            self.references = []
        self._update_scale()

    def _update_scale(self) -> None:
        """Recompute pixels-per-cm from stored scale pairs."""
        if len(self.scale_pairs) < 2:
            self.pixels_per_cm = None
            self.pixels_per_cm_sd = None
            self.scale_changed.emit(float("nan"), float("nan"))
            return

        w = max(self.width(), 1)
        h = max(self.height(), 1)
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

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        (x1, y1), (x2, y2) = self.rotation_refs
        p1 = QtCore.QPointF(x1 * w, y1 * h)
        p2 = QtCore.QPointF(x2 * w, y2 * h)

        angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
        self.rotation_matrix = QtGui.QTransform().rotateRadians(-angle)

        rotated = self.original_pixmap.transformed(
            self.rotation_matrix, QtCore.Qt.SmoothTransformation
        )
        self.original_pixmap = rotated

        if self.landmarks:
            rot_w = rotated.width()
            rot_h = rotated.height()
            new_pts: list[Tuple[float, float]] = []
            for lx, ly in self.landmarks:
                pt = QtCore.QPointF(lx * w, ly * h)
                pt = self.rotation_matrix.map(pt)
                new_pts.append((pt.x() / rot_w, pt.y() / rot_h))
            self.landmarks = new_pts

        self.rotation_refs = []
        self.scale_pairs = []
        self._scale_tmp = []
        self.pixels_per_cm = None
        self.pixels_per_cm_sd = None
        self.scale_changed.emit(float("nan"), float("nan"))
        self._rescale_pixmaps()
        self.update()
