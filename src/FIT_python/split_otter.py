# fit_otter/datasplits/split_sex_model.py

import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import StratifiedKFold, GroupKFold
from fit_otter.config import GLOBAL_RANDOM_SEED

# Define output path for optional split saving
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "datasets" / "processed" / "splits"


def sample_individuals(
    df: pd.DataFrame,
    dataset: str,
    sex: str,
    n: int,
    seed: int = GLOBAL_RANDOM_SEED,
) -> pd.DataFrame:
    """Randomly select ``n`` unique individuals for a dataset/sex combination.

    Called from :func:`create_train_test_split` when constructing the fixed test
    set.
    """
    subset = df[(df["Dataorigin"] == dataset) & (df["Sex"] == sex)]
    inds = subset["individual_id"].unique()
    sampled = pd.Series(inds).sample(min(n, len(inds)), random_state=seed).tolist()
    return df[df["individual_id"].isin(sampled)]


def create_train_test_split(
    df: pd.DataFrame,
    save_csv: bool = False,
    seed: int = GLOBAL_RANDOM_SEED,
    output_dir: Path | None = None,
    save_pickle: bool = False,
):
    """Create deterministic train/test/inference splits.

    Used by the main training notebook and unit tests.  The function enforces
    that the same individual never appears in multiple splits and optionally
    writes the resulting CSV files under ``datasets/processed/splits``.

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset of all footprints.
    save_csv : bool, default ``False``
        If ``True``, write the three CSV files for reproducibility.
    seed : int, default ``GLOBAL_RANDOM_SEED``
        Random seed controlling sampling.

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
        ``(train_df, test_df, inference_df)``.
    """

    # The 'inference' set: all rows where Dataorigin is "Fieldprints Portugal"
    inference_df = df[df["Dataorigin"] == "Fieldprints Portugal"]

    # Remove all 'inference' cases from further splitting
    df_clean = df[df["Dataorigin"] != "Fieldprints Portugal"]

    # The test set is built by fixed rules:
    # - All Lower Saxony fieldprints
    # - 3 random Own Data Collection (female) individuals
    # - 3 random Own Data Collection (male) individuals
    # - 2 random Vetrecova et al (female) individuals
    # - 2 random Vetrecova et al (male) individuals
    test_df = pd.concat([
        df_clean[df_clean["Dataorigin"] == "Fieldprints Lower Saxony"],
        sample_individuals(df_clean, "Own Data Collection", "Female", 3, seed),
        sample_individuals(df_clean, "Own Data Collection", "Male", 3, seed),
        sample_individuals(df_clean, "Vetrecova et al", "Female", 2, seed),
        sample_individuals(df_clean, "Vetrecova et al", "Male", 2, seed),
    ]).drop_duplicates()

    # --- IMPORTANT FIX ---
    # Ensure that no individual can be present in both train and test:
    # Only rows whose 'individual_id' does NOT appear in the test set are allowed in the training set.
    train_df = df_clean[~df_clean["individual_id"].isin(test_df["individual_id"])]

    # Optionally save splits as CSV
    if save_csv or save_pickle:
        out_dir = Path(output_dir or OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_csv:
            train_df.to_csv(out_dir / "train_sex_model.csv", index=False)
            test_df.to_csv(out_dir / "test_sex_model.csv", index=False)
            inference_df.to_csv(out_dir / "inference_portugal.csv", index=False)
        if save_pickle:
            train_df.to_pickle(out_dir / "train_sex_model.pkl")
            test_df.to_pickle(out_dir / "test_sex_model.pkl")
            inference_df.to_pickle(out_dir / "inference_portugal.pkl")

    return train_df, test_df, inference_df



def create_group_kfold(
    df: pd.DataFrame,
    n_splits: int = 3,
    seed: int = GLOBAL_RANDOM_SEED,
    save_csv: bool = False,
    output_dir: Path | None = None,
    save_pickle: bool = False,
) -> pd.DataFrame:
    """Create a fold assignment DataFrame using ``GroupKFold``.

    Used by the hyperparameter tuning pipeline so that each individual only appears
    in a single validation fold.

    Parameters
    ----------
    df : pd.DataFrame
        Data with an ``'individual_id'`` column.
    n_splits : int, default ``3``
        Number of cross-validation folds.
    seed : int, default ``GLOBAL_RANDOM_SEED``
        Random seed (currently unused but kept for API consistency).
    save_csv : bool, default ``False``
        If ``True``, write ``folds_sex_model_<n>fold.csv`` to disk.

    Returns
    -------
    pd.DataFrame
        DataFrame mapping each unique ``individual_id`` to a fold.
    """
    # Get unique individuals
    unique_inds = df["individual_id"].dropna().unique()
    fold_df = pd.DataFrame({"individual_id": unique_inds})
    fold_df["Fold"] = -1  # initialize

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (_, val_idx) in enumerate(gkf.split(X=fold_df, groups=fold_df["individual_id"])):
        fold_df.loc[val_idx, "Fold"] = fold

    if save_csv or save_pickle:
        out_dir = Path(output_dir or OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        if save_csv:
            fold_df.to_csv(out_dir / f"folds_sex_model_{n_splits}fold.csv", index=False)
        if save_pickle:
            fold_df.to_pickle(out_dir / f"folds_sex_model_{n_splits}fold.pkl")
    return fold_df


if __name__ == "__main__":
    try:
        import pandas as pd
    except Exception as exc:
        print("pandas not installed, skipping self-test:", exc)
    else:
        data = pd.DataFrame({
            "individual_id": ["i1", "i1", "i2", "i2", "i3", "i3"],
            "Trail": ["t1", "t2", "t1", "t2", "t1", "t2"],
            "Sex": ["M", "M", "F", "F", "M", "M"],
            "Dataorigin": ["Own", "Own", "Own", "Own", "Own", "Own"],
        })
        tr, te, inf = create_train_test_split(data)
        folds = create_group_kfold(data)
        print("train rows:", len(tr), "test rows:", len(te), "folds:", len(folds))
