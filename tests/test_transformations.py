import pytest
pytest.importorskip("PyQt5")
from PyQt5 import QtCore, QtGui
from FIT_python.utils.transformations import TransformationPipeline


def test_rotation_inversion():
    pipeline = TransformationPipeline()
    t = QtGui.QTransform().rotate(30)
    pipeline.add(t)

    p = QtCore.QPointF(5.0, 2.0)
    q = pipeline.apply_to_points([p])[0]
    r = pipeline.inverted().apply_to_points([q])[0]
    assert abs(r.x() - p.x()) < 1e-6
    assert abs(r.y() - p.y()) < 1e-6
