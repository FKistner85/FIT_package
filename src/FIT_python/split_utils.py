# src/FIT_python/split_utils.py

import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold, GroupKFold
from FIT_python.config import GLOBAL_RANDOM_SEED, TEST_SIZE, NUM_FOLDS, GROUP_COL, STRATIFY_COL

def train_test_group_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = GLOBAL_RANDOM_SEED,
    group_col: str = GROUP_COL,
    stratify_col: str = STRATIFY_COL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-stratified train/test split:
    - Ensures no group appears in both sets.
    - Stratifies by stratify_col proportion.
    """
    groups = df[group_col].unique()
    y_groups = (
        df.drop_duplicates(group_col)[stratify_col]
          .fillna("unknown")
          .values
    )
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(sss.split(groups.reshape(-1,1), y_groups))
    train_groups = groups[train_idx]
    test_groups  = groups[test_idx]
    return (
        df[df[group_col].isin(train_groups)],
        df[df[group_col].isin(test_groups)]
    )

def group_stratified_kfold(
    df: pd.DataFrame,
    n_splits: int = NUM_FOLDS,
    random_state: int = GLOBAL_RANDOM_SEED,
    group_col: str = GROUP_COL,
    stratify_col: str = STRATIFY_COL
) -> pd.DataFrame:
    """
    Assigns a 'Fold' column via StratifiedGroupKFold on groups,
    stratified by stratify_col.
    """
    individuals = df[[group_col, stratify_col]].drop_duplicates()
    individuals[stratify_col] = individuals[stratify_col].fillna("unknown")
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_series = pd.Series(-1, index=df.index, name="Fold")
    for fold, (_, val_idx) in enumerate(
        sgkf.split(
            individuals,
            individuals[stratify_col],
            groups=individuals[group_col]
        )
    ):
        fold_groups = individuals.iloc[val_idx][group_col]
        fold_series.loc[df[group_col].isin(fold_groups)] = fold
    return df.assign(Fold=fold_series)

def sample_individuals(
    df: pd.DataFrame,
    dataset: str,
    sex: str,
    n: int,
    seed: int = GLOBAL_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Randomly select `n` unique individuals for a dataset/sex combination.
    Used for the fixed Otter test split.
    """
    subset = df[(df["Dataorigin"] == dataset) & (df["sex"].str.lower() == sex.lower())]
    inds = subset["individual_id"].unique()
    sampled = (
        pd.Series(inds).sample(min(n, len(inds)), random_state=seed).tolist()
    )
    return df[df["individual_id"].isin(sampled)]

def create_train_test_split_otter(
    df: pd.DataFrame,
    seed: int = GLOBAL_RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create deterministic train/test/inference splits for Otter.
    Returns (train_df, test_df, inference_df).
    """
    inference_df = df[df["Dataorigin"] == "Fieldprints Portugal"]
    df_clean = df[df["Dataorigin"] != "Fieldprints Portugal"]

    test_df = pd.concat([
        df_clean[df_clean["Dataorigin"] == "Fieldprints Lower Saxony"],
        sample_individuals(df_clean, "Own Data Collection", "Female", 3, seed),
        sample_individuals(df_clean, "Own Data Collection", "Male", 3, seed),
        sample_individuals(df_clean, "Vetrecova et al", "Female", 2, seed),
        sample_individuals(df_clean, "Vetrecova et al", "Male", 2, seed),
    ]).drop_duplicates()

    train_df = df_clean[~df_clean["individual_id"].isin(test_df["individual_id"])]
    return train_df, test_df, inference_df
