import pandas as pd
from FIT_python.pipeline_individual_id.evaluation import compute_bcr

def test_compute_bcr():
    cm = pd.DataFrame(
        [[3, 1], [2, 4]],
        index=["true_same", "true_diff"],
        columns=["pred_same", "pred_diff"],
    )
    bcr = compute_bcr(cm)
    expected = (3 / 4 + 4 / 6) / 2
    assert abs(bcr - expected) < 1e-6

