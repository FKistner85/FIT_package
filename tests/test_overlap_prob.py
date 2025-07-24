import pytest
pytest.importorskip("pandas")
import pandas as pd
from FIT_python.pipeline_individual_id.evaluation import (
    compute_overlap_jsl_style_vec,
    compute_overlap_rhombus,
)

def _build_row(shift: float = 1.5) -> pd.Series:
    xa = [0, 1, 0, 1, 0.5]
    ya = [0, 0, 1, 1, 0.5]
    xb = [x + shift for x in xa]
    yb = ya
    return pd.Series({
        "coords_a_x": str(xa),
        "coords_a_y": str(ya),
        "coords_b_x": str(xb),
        "coords_b_y": str(yb),
    })

def test_overlap_prob_affects_jsl():
    row = _build_row()
    df = pd.DataFrame([row])
    assert not compute_overlap_jsl_style_vec(df, p=0.5)[0]
    assert compute_overlap_jsl_style_vec(df, p=0.99)[0]

def test_overlap_prob_affects_rhombus():
    row = _build_row()
    assert not compute_overlap_rhombus(row, p=0.5)
    assert compute_overlap_rhombus(row, p=0.9)
