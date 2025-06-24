# src/FIT_python/split_utils.py

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import StratifiedGroupKFold
from FIT_python.config import GLOBAL_RANDOM_SEED, TEST_SIZE, NUM_FOLDS, GROUP_COL

def train_test_group_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = GLOBAL_RANDOM_SEED,
    group_col: str = GROUP_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple group-based train/test split."""
    # 1) Gather unique group IDs
    groups = df[group_col].unique()
    # 2) Random shuffle of the group order
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)
    # 3) Determine split index
    split_idx = int(len(groups) * (1 - test_size))
    train_groups = groups[:split_idx]
    test_groups = groups[split_idx:]
    # 4) Mask and return DataFrames
    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    return train_df, test_df

def group_stratified_kfold(
    df: pd.DataFrame,
    n_splits: int = NUM_FOLDS,
    random_state: int = GLOBAL_RANDOM_SEED,
    group_col: str = GROUP_COL,
    stratify_col: str = "sex"
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
    subset = df[(df["dataorigin"] == dataset) & (df["sex"].str.lower() == sex.lower())]
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
    inference_df = df[df["dataorigin"] == "Fieldprints Portugal"]
    df_clean = df[df["dataorigin"] != "Fieldprints Portugal"]

    test_df = pd.concat([
        df_clean[df_clean["dataorigin"] == "Fieldprints Lower Saxony"],
        sample_individuals(df_clean, "Own Data Collection", "Female", 3, seed),
        sample_individuals(df_clean, "Own Data Collection", "Male", 3, seed),
        sample_individuals(df_clean, "Vetrecova et al", "Female", 2, seed),
        sample_individuals(df_clean, "Vetrecova et al", "Male", 2, seed),
    ]).drop_duplicates()

    train_df = df_clean[~df_clean["individual_id"].isin(test_df["individual_id"])]
    return train_df, test_df, inference_df
