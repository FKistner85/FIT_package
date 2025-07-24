"""Metrics grouped by individual identifiers."""

import pandas as pd


def individual_accuracies(y_true, y_pred, ids):
    """Return female, male and balanced accuracy per individual."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "id": ids})
    acc_per = (df.y_true == df.y_pred).groupby(df.id).mean()
    fem_ids = df.loc[df.y_true == 0, "id"].unique()
    mal_ids = df.loc[df.y_true == 1, "id"].unique()
    fem_acc = acc_per.loc[fem_ids].mean()
    mal_acc = acc_per.loc[mal_ids].mean()
    bal = 0.5 * (fem_acc + mal_acc)
    return fem_acc, mal_acc, bal


def individual_majority_stats(y_true, y_pred, ids):
    """Count individuals with majority correct predictions."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "id": ids})
    pct_per = (df.y_true == df.y_pred).groupby(df.id).mean()
    total = pct_per.shape[0]
    correct = int((pct_per > 0.5).sum())
    wrong = total - correct
    pct = correct / total
    return correct, wrong, pct
