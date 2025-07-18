"""Interactive canvas widget for landmark annotation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from FIT_python.soft_config import SOFT_CONFIG

CFG = SOFT_CONFIG.get("gui_annotator", {})
REFERENCE_TEMPLATE_DIR = Path(CFG.get("reference_template_dir", ""))
DISPLAY_SIZE = tuple(int(v) for v in CFG.get("display_size", [512, 512]))


class AnnotationCanvas(QtWidgets.QLabel):
    """Simple canvas allowing placement of exactly 11 landmarks."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.landmarks: List[Tuple[float, float]] = []
        self.crossfade_opacity = 0.3
        self.image: QtGui.QPixmap | None = None
        self.references: list[QtGui.QPixmap] = []
        self.reference_names: list[str] = []
        self._reference_index: int | None = None
        if REFERENCE_TEMPLATE_DIR.exists():
            for p in sorted(REFERENCE_TEMPLATE_DIR.iterdir()):
                if p.is_file():
                    pm = QtGui.QPixmap(str(p))
                    pm = pm.scaled(
                        DISPLAY_SIZE[0] // 2,
                        DISPLAY_SIZE[1] // 2,
                        QtCore.Qt.KeepAspectRatio,
                    )
                    self.references.append(pm)
                    self.reference_names.append(p.stem)
            if self.references:
                self._reference_index = 0
        self.setMouseTracking(True)

    # configuration slots -------------------------------------------------
    def set_crossfade_opacity(self, value: float) -> None:
        """Set overlay opacity between 0 and 1."""
        self.crossfade_opacity = max(0.0, min(1.0, value))
        self.update()

    def set_reference_index(self, idx: int) -> None:
        if 0 <= idx < len(self.references):
            self._reference_index = idx
            self.update()

    def load_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self.image = pixmap.scaled(*DISPLAY_SIZE)
        self.setFixedSize(*DISPLAY_SIZE)
        self.landmarks = []
        self.update()

    # Qt event handlers -------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.LeftButton and len(self.landmarks) < 11:
            w = self.image.width() if self.image else 1
            h = self.image.height() if self.image else 1
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
            w = self.image.width()
            h = self.image.height()
            for x, y in self.landmarks:
                painter.drawEllipse(QtCore.QPoint(int(x * w), int(y * h)), 4, 4)
        painter.end()

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        return QtCore.QSize(*DISPLAY_SIZE)
