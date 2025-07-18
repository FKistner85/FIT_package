"""Interactive canvas widget for landmark annotation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from FIT_python.soft_config import SOFT_CONFIG

CFG = SOFT_CONFIG.get("gui_annotator", {})
REFERENCE_IMAGE = Path(CFG.get("reference_image", ""))


class AnnotationCanvas(QtWidgets.QLabel):
    """Simple canvas allowing placement of exactly 11 landmarks."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.landmarks: List[Tuple[int, int]] = []
        self.crossfade_opacity = 0.3
        self.image: QtGui.QPixmap | None = None
        self.reference: QtGui.QPixmap | None = None
        if REFERENCE_IMAGE.exists():
            self.reference = QtGui.QPixmap(str(REFERENCE_IMAGE))
        self.setMouseTracking(True)

    def load_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self.image = pixmap
        self.landmarks = []
        self.update()

    # Qt event handlers -------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.LeftButton and len(self.landmarks) < 11:
            self.landmarks.append((event.x(), event.y()))
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        if self.image:
            painter.drawPixmap(0, 0, self.image)
        if self.reference and self.image:
            painter.setOpacity(self.crossfade_opacity)
            painter.drawPixmap(0, 0, self.reference.scaled(self.image.size()))
            painter.setOpacity(1.0)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 4))
        for x, y in self.landmarks:
            painter.drawEllipse(QtCore.QPoint(x, y), 4, 4)
        painter.end()

    def sizeHint(self) -> QtCore.QSize:  # type: ignore[override]
        if self.image:
            return self.image.size()
        return super().sizeHint()
