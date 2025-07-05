import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer


def individual_balanced_sex_score(y_true, y_pred, individual_ids):
    """Balanced accuracy aggregated per individual.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 for female, 1 for male).
    y_pred : array-like
        Predicted labels.
    individual_ids : array-like
        Identifier for the individual each sample belongs to.
    Returns
    -------
    float
        The individual balanced sex score.
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "individual_id": individual_ids,
    })

    per_ind = (
        df.groupby("individual_id")
          .apply(lambda g: (g.y_pred == g.y_true).mean())
          .rename("ind_acc")
    )

    true_sex = (
        df.groupby("individual_id").y_true
          .first().map({0: "f", 1: "m"})
    )

    female_ids = true_sex[true_sex == "f"].index
    male_ids = true_sex[true_sex == "m"].index
    female_acc = per_ind.loc[female_ids].mean() if len(female_ids) > 0 else 0
    male_acc = per_ind.loc[male_ids].mean() if len(male_ids) > 0 else 0

    return 0.5 * (female_acc + male_acc)


individual_balanced_sex = make_scorer(
    individual_balanced_sex_score,
    greater_is_better=True,
    needs_threshold=False,
    needs_proba=False,
)
