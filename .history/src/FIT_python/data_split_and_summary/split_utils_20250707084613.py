# src/FIT_python/split_utils.py

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from sklearn.model_selection import StratifiedGroupKFold
from FIT_python.config import GLOBAL_RANDOM_SEED, TEST_SIZE, NUM_FOLDS, GROUP_COL, SPLITS_DIR
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupKFold,
    KFold,
)

from sklearn.model_selection import train_test_split

from pathlib import Path
import pandas as pd

from FIT_python.config import SPLITS_DIR, NUM_FOLDS, GROUP_COL

def stratified_individual_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "individual_id",   # neu
    stratify_col: str = "sex"           # neu
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df on the level of individuals (group_col), stratified by stratify_col.
    - Alle Zeilen eines Individuums gehen zusammen in Train oder Test
    - Ungültige/fehlende Gruppen landen im inference_df
    """
    # 1) Prüfe Argumente
    if group_col not in df.columns:
        raise ValueError(f"Gruppenspalte '{group_col}' nicht im DataFrame.")
    if stratify_col not in df.columns:
        raise ValueError(f"Stratifizierungs-Spalte '{stratify_col}' nicht im DataFrame.")

    # 2) Metadaten: ein Eintrag pro Individuum mit gültigem Sex
    meta = (
        df[[group_col, stratify_col]]
        .dropna(subset=[group_col, stratify_col])
        .drop_duplicates(subset=[group_col])
    )
    # Nur m/f zulassen
    meta = meta[meta[stratify_col].astype(str).str.lower().isin(["m","f"])].copy()
    meta[stratify_col] = meta[stratify_col].str.lower()

    # 3) Stratified Split auf Gruppen-Ebene
    ids    = meta[group_col].tolist()
    labels = meta[stratify_col].map({"f":0, "m":1}).tolist()
    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # 4) DataFrame-Splits
    train_df     = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    test_df      = df[df[group_col].isin(test_ids)].reset_index(drop=True)
    inference_df = df[~df[group_col].isin(ids)].reset_index(drop=True)

    return train_df, test_df, inference_df

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

        for fold, (_, val_idx) in enumerate(
            sgkf.split(individuals, individuals[stratify_col], groups=individuals[group_col])
        ):
            fold_groups = individuals.iloc[val_idx][group_col]
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
    print("\n🔄 Starte Otter-Splitting...")

    # 1) Inference: alle aus Portugal
    inference_df = df[df["dataorigin"] == "Fieldprints Portugal"]
    print(f"🛰️ Inference-Split – {len(inference_df)} Zeilen")
    print("↪️ Inference 'dataorigin':", inference_df["dataorigin"].unique())
    print("↪️ Inference 'sex' unique:", inference_df['sex'].unique())

    # 2) Alle restlichen Daten
    df_clean = df[df["dataorigin"] != "Fieldprints Portugal"]
    print(f"\n🧹 df_clean – {len(df_clean)} Zeilen (ohne Portugal)")
    print("🧾 dataorigin:", df_clean["dataorigin"].unique())
    print("🧬 sex:", df_clean["sex"].unique())
    print("🆔 individual_id:", df_clean["individual_id"].nunique())

    # 3) Test-Split (bestimmte Individuals)
    print("\n🔬 Erzeuge Test-Split...")
    test_ls = df_clean[df_clean["dataorigin"] == "Fieldprints Lower Saxony"]
    test_own_f = sample_individuals(df_clean, "Own Data Collection", "f", 3, seed)
    test_own_m = sample_individuals(df_clean, "Own Data Collection", "m", 3, seed)
    test_vet_f = sample_individuals(df_clean, "Vetrecova et al", "f", 2, seed)
    test_vet_m = sample_individuals(df_clean, "Vetrecova et al", "m", 2, seed)

    test_df = pd.concat([
        test_ls, test_own_f, test_own_m, test_vet_f, test_vet_m
    ]).drop_duplicates()

    print(f"✅ Test-Split fertig – {len(test_df)} Zeilen")
    print("👤 Test-Individuals:", test_df["individual_id"].nunique())
    print("📊 Test 'dataorigin':", test_df["dataorigin"].value_counts().to_string())
    print("🧬 Test 'sex':", test_df["sex"].value_counts().to_string())

    # 4) Train-Split (alle übrigen)
    train_df = df_clean[~df_clean["individual_id"].isin(test_df["individual_id"])]
    print(f"\n🧠 Train-Split – {len(train_df)} Zeilen")
    print("👤 Train-Individual IDs:", train_df["individual_id"].nunique())
    print("📊 Train 'dataorigin':", train_df["dataorigin"].value_counts().to_string())
    print("🧬 Train 'sex':", train_df["sex"].value_counts().to_string())

    # 5) Fold-Zuordnung
    print("\n🎲 Berechne Fold-Zuordnung...")
    y_train = train_df["sex"].map({"f": 0, "m": 1})
    fold_ids, method = _make_folds(
        train_df, y_train, n_splits=NUM_FOLDS, group_col=GROUP_COL
    )
    train_df["Fold"] = fold_ids

    print(f"\n✅ Fold-Methode verwendet: {method}")
    print("📊 Fold-Verteilung:")
    print(train_df["Fold"].value_counts().sort_index())

    return train_df, test_df, inference_df




def splits_available() -> bool:
    """Prüft, ob mindestens ein gültiges Train/Test-Paar existiert."""
    if not Path(SPLITS_DIR).exists():
        return False
    for d in Path(SPLITS_DIR).iterdir():
        if not d.is_dir():
            continue
        if (d / "train.parquet").exists() and (d / "test.parquet").exists():
            return True
    return False


def _check_valid(fold_ids: np.ndarray, y: pd.Series, n_splits: int) -> bool:
    """Verifiziert, dass in jedem Fold beide Klassen 0 und 1 vorkommen."""
    for fold_i in range(n_splits):
        classes = set(y[fold_ids == fold_i].unique())
        if classes != {0, 1}:
            return False
    return True



def ensure_valid_splits() -> None:
    """
    Prüft, ob für jede Spezies in SPLITS_DIR eine gültige 'Fold'-Spalte existiert,
    d.h. in jedem Fold beide Klassen (0 und 1) vertreten sind.
    Falls nicht, wird SplitWrapper().split_all() ausgeführt.
    """
    from FIT_python.pipeline_sex.split_wrapper import SplitWrapper

    for species_dir in Path(SPLITS_DIR).iterdir():
        if not species_dir.is_dir():
            continue
        train_fp = species_dir / "train.parquet"
        if not train_fp.exists():
            # kein Split da → neu generieren
            SplitWrapper().split_all()
            return

        df = pd.read_parquet(train_fp)
        if "Fold" not in df.columns:
            SplitWrapper().split_all()
            return

        # Labels kodieren
        y = df["sex"].map({"f": 0, "m": 1})
        # Prüfen, ob die vorhandenen Fold-IDs valide sind
        fold_ids = df["Fold"].values.astype(int)
        if not _check_valid(fold_ids, y, NUM_FOLDS):
            SplitWrapper().split_all()
            return

    # wenn wir hier ankommen, sind alle Splits valide


def _make_folds(
    df: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    group_col: str
) -> Tuple[np.ndarray, str]:
    """
    Versucht in folgender Reihenfolge:
      1) vorhandene 'Fold'-Spalte (predefined),
      2) StratifiedGroupKFold via group_stratified_kfold(),
      3) GroupKFold,
      4) KFold.
    Gibt (fold_ids, methode_name) zurück.
    """
    # 1) vorhandene Fold-Spalte?
    if "Fold" in df.columns:
        fold_ids = df["Fold"].values.astype(int)
        if _check_valid(fold_ids, y, n_splits):
            return fold_ids, "predefined"

    # 2) StratifiedGroupKFold via euren Helper (bis zu 3 Versuche)
    for attempt in range(3):
        seed = GLOBAL_RANDOM_SEED + attempt
        try:
            # group_stratified_kfold fügt 'Fold' in df zurück
            df_folds = group_stratified_kfold(
                df, n_splits=n_splits, random_state=seed, group_col=group_col
            )
            fold_ids = df_folds["Fold"].values.astype(int)
            if _check_valid(fold_ids, y, n_splits):
                return fold_ids, "stratified_group"
        except Exception:
            continue

    # 3) GroupKFold (bis zu 3 Versuche)
    for attempt in range(3):
        gkf = GroupKFold(n_splits=n_splits)
        fold_ids = np.empty(len(df), dtype=int)
        for fold, (_, val_idx) in enumerate(
            gkf.split(df, groups=df[group_col])
        ):
            fold_ids[val_idx] = fold
        if _check_valid(fold_ids, y, n_splits):
            return fold_ids, "group"

    # 4) Klassisches KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
    fold_ids = np.empty(len(df), dtype=int)
    for fold, (_, val_idx) in enumerate(kf.split(df)):
        fold_ids[val_idx] = fold
    return fold_ids, "kfold"