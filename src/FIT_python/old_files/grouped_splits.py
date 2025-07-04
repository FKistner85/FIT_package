# src/FIT_python/grouped_splits.py

import pandas as pd
from typing import Tuple
import logging
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold

# Default constants
GROUP_COL       = "individual_id"
STRATIFY_COL    = "sex"
GLOBAL_SEED     = 42
TEST_SIZE       = 0.2
N_SPLITS        = 5

def train_test_group_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = GLOBAL_SEED,
    group_col: str = GROUP_COL,
    stratify_col: str = STRATIFY_COL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-stratified train/test split:
    - Ensures no group appears in both sets.
    - Stratifies by stratify_col proportion.
    """
    groups = df[group_col].unique()
    # label per group
    y_groups = df.drop_duplicates(group_col)[stratify_col].fillna("unknown").values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(groups.reshape(-1,1), y_groups))
    train_groups = groups[train_idx]
    test_groups  = groups[test_idx]
    train_df = df[df[group_col].isin(train_groups)]
    test_df  = df[df[group_col].isin(test_groups)]
    return train_df, test_df

def group_stratified_kfold(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    random_state: int = GLOBAL_SEED,
    group_col: str = GROUP_COL,
    stratify_col: str = STRATIFY_COL
) -> pd.DataFrame:
    """
    Assigns a 'Fold' column via StratifiedGroupKFold on groups,
    stratified by stratify_col.
    """
    logger = logging.getLogger(__name__)

    individuals = df[[group_col, stratify_col]].drop_duplicates()
    individuals[stratify_col] = individuals[stratify_col].fillna("unknown")

    class_counts = individuals[stratify_col].value_counts()
    n_splits = min(n_splits, class_counts.min())
    logger.debug("Initial n_splits=%s based on class counts %s", n_splits, class_counts.to_dict())

    if n_splits < 2:
        raise ValueError("Zu wenige Gruppen in einer Klasse für Stratifizierung")

    seed = random_state
    for attempt in range(5):
        logger.debug("StratifiedGroupKFold attempt %s with seed=%s", attempt + 1, seed)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_series = pd.Series(-1, index=df.index, name="Fold")

        for fold, (_, test_inds) in enumerate(
            sgkf.split(individuals, individuals[stratify_col], groups=individuals[group_col])
        ):
            fold_groups = individuals.iloc[test_inds][group_col]
            fold_series.loc[df[group_col].isin(fold_groups)] = fold
            classes = df[df[group_col].isin(fold_groups)][stratify_col].dropna().unique()
            logger.debug("seed=%s fold=%s classes=%s", seed, fold, classes)

        valid = True
        for fold in range(n_splits):
            cls = df[fold_series == fold][stratify_col].dropna().unique()
            if len(cls) < 2:
                valid = False
                break
        if valid:
            return df.assign(Fold=fold_series)

        seed += 1

    raise RuntimeError("Konnte keine ausgewogene Stratifikation finden")
