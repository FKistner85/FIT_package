# src/FIT_python/split_utils.py

import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from typing import Tuple
from sklearn.model_selection import StratifiedGroupKFold
from FIT_python.config import GLOBAL_RANDOM_SEED, TEST_SIZE, NUM_FOLDS, GROUP_COL, SPLITS_DIR, RESULTS_DATA_DIR
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GroupKFold,
    KFold,
)

import warnings


from sklearn.model_selection import train_test_split

from pathlib import Path
import pandas as pd


from FIT_python.config import SPLITS_DIR, NUM_FOLDS, GROUP_COL

def stratified_individual_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "individual_id",
    stratify_col: str = "sex",
    add_folds: bool = True,
    n_folds: int = NUM_FOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ``df`` by ``group_col`` while stratifying by ``stratify_col``.
    - All rows for one individual are kept together in train or test
    - Invalid or missing groups are placed into ``inference_df``
    - Optionally assign a ``Fold`` column to ``train_df`` using
      stratified group k-fold.
    """
    # 1) Validate arguments
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")
    if stratify_col not in df.columns:
        raise ValueError(f"Stratify column '{stratify_col}' not found in DataFrame.")

    # 2) Metadata: one entry per individual with a valid sex label
    meta = (
        df[[group_col, stratify_col]]
        .dropna(subset=[group_col, stratify_col])
        .drop_duplicates(subset=[group_col])
    )
    # Allow only m/f values
    meta = meta[meta[stratify_col].astype(str).str.lower().isin(["m","f"])].copy()
    meta[stratify_col] = meta[stratify_col].str.lower()

    # 3) Stratified split on the group level
    ids    = meta[group_col].tolist()
    labels = meta[stratify_col].map({"f":0, "m":1}).tolist()
    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # 4) DataFrame splits
    train_df     = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    test_df      = df[df[group_col].isin(test_ids)].reset_index(drop=True)
    inference_df = df[~df[group_col].isin(ids)].reset_index(drop=True)

    if add_folds:
        y_ser = train_df[stratify_col].map({"f": 0, "m": 1})
        fold_ids, _ = _make_folds(
            train_df, y_ser, n_splits=n_folds, group_col=group_col
        )
        train_df = train_df.assign(Fold=fold_ids)

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
        raise ValueError("Not enough groups in one class for stratification")

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

    raise RuntimeError("Could not determine a balanced stratification")

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
    # Spalten großgeschrieben
    subset = df[(df["DATAORIGIN"] == dataset) & (df["SEX"].str.lower() == sex.lower())]
    inds = subset["INDIVIDUAL_ID"].unique()
    sampled = (
        pd.Series(inds).sample(min(n, len(inds)), random_state=seed).tolist()
    )
    return df[df["INDIVIDUAL_ID"].isin(sampled)]


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
    inference_df = df[df["DATAORIGIN"] == "Fieldprints Portugal"]
    print(f"🛰️ Inference-Split – {len(inference_df)} Zeilen")
    print("↪️ Inference 'DATAORIGIN':", inference_df["DATAORIGIN"].unique())
    print("↪️ Inference 'SEX' unique:", inference_df["SEX"].unique())

    # 2) Alle restlichen Daten
    df_clean = df[df["DATAORIGIN"] != "Fieldprints Portugal"]
    print(f"\n🧹 df_clean – {len(df_clean)} Zeilen (ohne Portugal)")
    print("🧾 DATAORIGIN:", df_clean["DATAORIGIN"].unique())
    print("🧬 SEX:", df_clean["SEX"].unique())
    print("🆔 INDIVIDUAL_ID:", df_clean["INDIVIDUAL_ID"].nunique())

    # 3) Test-Split (bestimmte Individuals)
    print("\n🔬 Erzeuge Test-Split...")
    test_ls    = df_clean[df_clean["DATAORIGIN"] == "Fieldprints Lower Saxony"]
    test_own_f = sample_individuals(df_clean, "Own Data Collection", "f", 3, seed)
    test_own_m = sample_individuals(df_clean, "Own Data Collection", "m", 3, seed)
    test_vet_f = sample_individuals(df_clean, "Vetrecova et al", "f", 2, seed)
    test_vet_m = sample_individuals(df_clean, "Vetrecova et al", "m", 2, seed)

    test_df = pd.concat([
        test_ls, test_own_f, test_own_m, test_vet_f, test_vet_m
    ]).drop_duplicates()

    print(f"✅ Test-Split fertig – {len(test_df)} Zeilen")
    print("👤 Test-Individuals:", test_df["INDIVIDUAL_ID"].nunique())
    print("📊 Test 'DATAORIGIN':", test_df["DATAORIGIN"].value_counts().to_string())
    print("🧬 Test 'SEX':", test_df["SEX"].value_counts().to_string())

    # 4) Train split with the remaining data
    train_df = df_clean[~df_clean["INDIVIDUAL_ID"].isin(test_df["INDIVIDUAL_ID"])]
    print(f"\n🧠 Train-Split – {len(train_df)} Zeilen")
    print("👤 Train-Individual IDs:", train_df["INDIVIDUAL_ID"].nunique())
    print("📊 Train 'DATAORIGIN':", train_df["DATAORIGIN"].value_counts().to_string())
    print("🧬 Train 'SEX':", train_df["SEX"].value_counts().to_string())

    # 5) Assign folds
    print("\n🎲 Berechne Fold-Zuordnung...")
    y_train = train_df["SEX"].map({"f": 0, "m": 1})
    fold_ids, method = _make_folds(
        train_df, y_train, n_splits=NUM_FOLDS, group_col=GROUP_COL
    )
    train_df["FOLD"] = fold_ids

    print(f"\n✅ Fold method used: {method}")
    print("📊 Fold-Verteilung:")
    print(train_df["FOLD"].value_counts().sort_index())

    return train_df, test_df, inference_df





def splits_available() -> bool:
    """Return ``True`` if at least one valid train/test pair exists."""
    if not Path(SPLITS_DIR).exists():
        return False
    for d in Path(SPLITS_DIR).iterdir():
        if not d.is_dir():
            continue
        if (d / "train.parquet").exists() and (d / "test.parquet").exists():
            return True
    return False


def _check_valid(fold_ids: np.ndarray, y: pd.Series, n_splits: int) -> bool:
    """Check that each fold contains both classes 0 and 1."""
    for fold_i in range(n_splits):
        classes = set(y[fold_ids == fold_i].unique())
        if classes != {0, 1}:
            return False
    return True



def ensure_valid_splits() -> None:
    """Validate that each species in ``SPLITS_DIR`` has a valid ``Fold`` column
    with both classes present in every fold.  Warns if any split is invalid."""
    for species_dir in Path(SPLITS_DIR).iterdir():
        if not species_dir.is_dir():
            continue

        train_fp = species_dir / "train.parquet"
        if not train_fp.exists():
            warnings.warn(
                f"Missing train.parquet in {species_dir} – split not found.",
                UserWarning,
            )
            continue

        df = pd.read_parquet(train_fp)
        if "Fold" not in df.columns:
            warnings.warn(
                f"Column 'Fold' missing in {species_dir}.",
                UserWarning,
            )
            continue

        # Labels kodieren
        y = df["sex"].map({"f": 0, "m": 1})
        fold_ids = df["Fold"].values.astype(int)
        if not _check_valid(fold_ids, y, NUM_FOLDS):
            warnings.warn(
                f"Invalid fold distribution in {species_dir} (not all folds contain both classes).",
                UserWarning,
            )
            continue

    # reaching this point means all splits are valid


def _make_folds(
    df: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    group_col: str
) -> Tuple[np.ndarray, str]:
    """Return ``(fold_ids, method_name)`` using the following strategy:
      1) existing ``Fold`` column (predefined)
      2) ``StratifiedGroupKFold`` via :func:`group_stratified_kfold`
      3) ``GroupKFold``
      4) ``KFold``
    """
    # 1) vorhandene Fold-Spalte?
    if "Fold" in df.columns:
        fold_ids = df["Fold"].values.astype(int)
        if _check_valid(fold_ids, y, n_splits):
            return fold_ids, "predefined"

    # 2) Try ``StratifiedGroupKFold`` up to three times
    for attempt in range(3):
        seed = GLOBAL_RANDOM_SEED + attempt
        try:
            # ``group_stratified_kfold`` adds the ``Fold`` column to ``df``
            df_folds = group_stratified_kfold(
                df, n_splits=n_splits, random_state=seed, group_col=group_col
            )
            fold_ids = df_folds["Fold"].values.astype(int)
            if _check_valid(fold_ids, y, n_splits):
                return fold_ids, "stratified_group"
        except Exception:
            continue

    # 3) Try ``GroupKFold`` up to three times
    for attempt in range(3):
        gkf = GroupKFold(n_splits=n_splits)
        fold_ids = np.empty(len(df), dtype=int)
        for fold, (_, val_idx) in enumerate(
            gkf.split(df, groups=df[group_col])
        ):
            fold_ids[val_idx] = fold
        if _check_valid(fold_ids, y, n_splits):
            return fold_ids, "group"

    # 4) Plain ``KFold``
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_RANDOM_SEED)
    fold_ids = np.empty(len(df), dtype=int)
    for fold, (_, val_idx) in enumerate(kf.split(df)):
        fold_ids[val_idx] = fold
    return fold_ids, "kfold"

def prepare_all_splits(
    species_filter: Optional[List[str]] = None
) -> None:
    """
    Lädt alle CSVs in RAW_DIR und erstellt für jede Art Splits:

    - Wenn `species_filter` angegeben ist, nur für diese Arten.
    - Für 'Eurasian Otter.csv' wird `create_train_test_split_otter` verwendet.
    - Für alle anderen Arten `stratified_individual_split`.
    """

    for csv_fp in RAW_DIR.glob("*.csv"):
        species = csv_fp.stem  # z.B. "Eurasian Otter"
        # Filter?
        if species_filter and species not in species_filter:
            continue

        # 1) Import
        importer = DataImporter(RAW_DIR, target_cols=DEFAULT_TARGETS)
        dfs = importer.run()
        # Schlüssel finden, der zur aktuellen Datei passt
        # (DataImporter kann mehrere Tabs/Sheets liefern)
        key = next(k for k in dfs if species.lower() in k.lower())
        df = dfs[key]

        # 2) Splits erstellen
        if species.lower() == "eurasian otter":
            train_df, test_df, inf_df = create_train_test_split_otter(df)
        else:
            # generischer Stratified split nach individual_id (oder GROUP_COL)
            train_df, test_df, inf_df = stratified_individual_split(
                df,
                id_col=GROUP_COL,
                n_splits=NUM_FOLDS,
                random_state=0
            )

        # 3) Folds ins train_df schreiben, falls nicht schon geschehen
        if "Fold" not in train_df.columns:
            y_train = train_df["sex"].map({"f": 0, "m": 1}) \
                      if species.lower()=="eurasian otter" else \
                      train_df[GROUP_COL]
            folds, _ = _make_folds(
                train_df,
                y_train,
                n_splits=NUM_FOLDS,
                group_col=GROUP_COL
            )
            train_df = train_df.assign(Fold=folds)

        # 4) Abspeichern
        out_dir = SPLITS_DIR / species.replace(" ", "_").lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(out_dir / "train.parquet", index=False)
        test_df.to_parquet (out_dir / "test.parquet",  index=False)
        inf_df.to_parquet  (out_dir / "inference.parquet", index=False)

        # 5) Optional: Zusammenfassung & Plots
        run_summary(
            out_dir,
            RESULTS_DATA_DIR / f"{species.replace(' ','_').lower()}_summary.csv",
            RESULTS_DATA_DIR / f"{species.replace(' ','_').lower()}_fig"
        )