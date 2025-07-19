from __future__ import annotations

from typing import Iterable, List
from PyQt5 import QtCore, QtGui

class TransformationPipeline:
    """Manage a sequence of ``QTransform`` objects."""

    def __init__(self, transforms: Iterable[QtGui.QTransform] | None = None) -> None:
        self.transforms: List[QtGui.QTransform] = [QtGui.QTransform(t) for t in transforms] if transforms else []

    def add(self, transform: QtGui.QTransform) -> None:
        """Append a transformation to the pipeline."""
        self.transforms.append(QtGui.QTransform(transform))

    def combined(self) -> QtGui.QTransform:
        """Return a ``QTransform`` representing all steps combined."""
        result = QtGui.QTransform()
        for t in self.transforms:
            result = t * result
        return result

    def inverted(self) -> "TransformationPipeline":
        """Return a pipeline with all transforms inverted and reversed."""
        inv = [t.inverted()[0] for t in reversed(self.transforms)]
        return TransformationPipeline(inv)

    def apply_to_image(self, pixmap: QtGui.QPixmap) -> QtGui.QPixmap:
        """Return ``pixmap`` transformed by all steps."""
        out = pixmap
        for t in self.transforms:
            out = out.transformed(t, QtCore.Qt.SmoothTransformation)
        return out

    def apply_to_points(self, points: Iterable[QtCore.QPointF]) -> List[QtCore.QPointF]:
        """Return list of points transformed by all steps."""
        result = []
        for p in points:
            q = QtCore.QPointF(p)
            for t in self.transforms:
                q = t.map(q)
            result.append(q)
        return result

    def to_json(self) -> List[List[float]]:
        """Serialise transforms into lists of 9 coefficients."""
        data: List[List[float]] = []
        for t in self.transforms:
            data.append([
                t.m11(), t.m12(), t.m13(),
                t.m21(), t.m22(), t.m23(),
                t.m31(), t.m32(), t.m33(),
            ])
        return data

    @classmethod
    def from_json(cls, data: Iterable[Iterable[float]]) -> "TransformationPipeline":
        transforms = [
            QtGui.QTransform(*values) for values in data
        ]
        return cls(transforms)
