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


def individual_balanced_sex_from_df(estimator, X, y_true):
    """Scorer that extracts ``individual_id`` from ``X``.

    This variant allows using ``GridSearchCV`` without having to pass the
    ``individual_ids`` separately via ``fit``. The input ``X`` must be a
    ``pandas.DataFrame`` containing either an ``individual_id`` or ``id``
    column. These identifiers are removed before prediction so that the
    underlying estimator sees only feature columns.
    """

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a DataFrame containing the individual id")

    if "individual_id" in X.columns:
        ids = X["individual_id"].values
    elif "id" in X.columns:
        ids = X["id"].values
    else:
        raise ValueError("X must contain an 'individual_id' or 'id' column")

    # Keep ``X`` intact for the estimator. The pipeline will drop the ID column
    # internally. This avoids feature-name mismatch errors.
    y_pred = estimator.predict(X)
    return individual_balanced_sex_score(y_true, y_pred, ids)


# ``GridSearchCV`` accepts any callable of ``(estimator, X, y_true)`` as a
# scorer. We therefore expose ``individual_balanced_sex_df`` directly without
# ``make_scorer`` so that it can access the ``individual_id`` column in ``X``.
individual_balanced_sex_df = individual_balanced_sex_from_df
