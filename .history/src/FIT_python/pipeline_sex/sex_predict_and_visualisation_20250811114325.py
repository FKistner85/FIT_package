from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix

from FIT_python.config import DATA_DIR, PATHS, SEX_PREDICT_METRIC, CONFIG
from FIT_python.utils import get_species_paths
from FIT_python.Visualisations.plot_style import apply_style, map_sex


# =============================================================================
# Constants / Mappings
# =============================================================================
SEX_TO_INT = {"f": 0, "m": 1}
INT_TO_SEX = {0: "f", 1: "m"}
SEX_VALUE_MAP = CONFIG["visualisation"]["sex"]["value_map"]


# =============================================================================
# Small helpers
# =============================================================================
def _ensure_out_dir(out_dir: str | Path | None) -> Path:
    p = Path(out_dir) if out_dir is not None else (PATHS["sex_modelling"] / "figures")
    p.mkdir(parents=True, exist_ok=True)
    return p

def _savefig(path: Path, caption: str | None = None) -> None:
    from FIT_python.caption_utils import save_caption  # lazy import
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if caption:
        save_caption(path, caption)


# =============================================================================
# Small I/O helper
# =============================================================================
def _load_split(path: Path) -> pd.DataFrame:
    if Path(path).exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


# =============================================================================
# Baseline prediction (simple model)
# =============================================================================
def predict_simple_baseline(
    species: str,
    exp_dir: Path | None = None,
    models_dir: Path | None = None,
    *,
    include_inference: bool = True,
    reuse_csv: bool = True,
) -> pd.DataFrame:
    """Return predictions of the simple baseline model for ``species``."""
    paths = get_species_paths(section="sex_modelling", species=species)
    splits_dir = PATHS["splits"] / species

    pred_dir = (
        Path(paths["predictions"])
        if "predictions" in paths
        else (PATHS["sex_modelling"] / species / "tables")
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pred_dir / f"{species}_baseline_predictions.csv"

    if models_dir is not None:
        model_dir = Path(models_dir)
    elif exp_dir is not None:
        model_dir = Path(exp_dir) / "models"
    else:
        model_dir = Path(paths["models"])
    model_path = model_dir / f"{species}.joblib"

    if reuse_csv:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Predictions CSV not found: {csv_path}. "
            "Set reuse_csv=False to recompute predictions."
        )

    if not splits_dir.is_dir():
        warnings.warn(
            f"Split directory for {species!r} not found – skipping.",
            UserWarning,
        )
        return pd.DataFrame()

    clf = joblib.load(model_path)

    split_names = ["train", "test"]
    if include_inference and (splits_dir / "inference.parquet").exists():
        split_names.append("inference")

    dfs: list[pd.DataFrame] = []
    for name in split_names:
        df = _load_split(splits_dir / f"{name}.parquet")
        if df.empty:
            continue

        drop_cols = ["sex"]
        if "individual_id" in df.columns:
            drop_cols.append("individual_id")
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=["number"]).copy()
        if "Fold" in X.columns:
            X = X.drop(columns=["Fold"])

        if X.empty:
            continue

        df["__split__"] = name
        df["pred_baseline_sex"] = clf.predict(X)
        proba = clf.predict_proba(X)
        df["pred_baseline_proba_f"] = proba[:, 0]
        df["pred_baseline_proba_m"] = proba[:, 1]
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    if not reuse_csv:
        all_df.to_csv(csv_path, index=False)
    return all_df


# =============================================================================
# Best-model predictions (per species)
# =============================================================================
def predict_all(
    species: str,
    metric_key: str = SEX_PREDICT_METRIC,  # e.g. "balanced_accuracy"
    prefer_generic: bool = False,          # kept for API compatibility
    models_dir: str | Path | None = None,
    include_inference: bool = True,
    reuse_csv: bool = False,
    use_cv_train_predictions: bool = True,
) -> pd.DataFrame:
    """Return dataframe with best-model predictions for Train/Test/Inference.

    Train: Out-of-fold predictions (OOF) using 'Fold' column.
    Test + Inference: Predictions from model retrained on full Train.
    """
    paths = get_species_paths(section="sex_modelling", species=species)
    splits_dir = PATHS["splits"] / species

    models_dir = Path(models_dir) if models_dir is not None else Path(paths["models"])

    pred_dir = (
        Path(paths["predictions"])
        if "predictions" in paths
        else (PATHS["sex_modelling"] / species / "tables")
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pred_dir / f"{species}_all_predictions.csv"

    if reuse_csv:
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"Predictions CSV not found: {csv_path}. "
            "Set reuse_csv=False to recompute predictions."
        )

    if not splits_dir.is_dir():
        warnings.warn(f"Split directory for {species!r} not found – skipping.")
        return pd.DataFrame()

    split_names = ["train", "test"]
    if include_inference and (splits_dir / "inference.parquet").exists():
        split_names.append("inference")
    splits = {n: splits_dir / f"{n}.parquet" for n in split_names}
    dfs = {name: pd.read_parquet(p) for name, p in splits.items()}

    for name, df in dfs.items():
        df.columns = (
            df.columns.str.replace(r"[.\-]", "_", regex=True).str.replace("T", "t")
        )
        df["__split__"] = name

    best_model_path = models_dir / metric_key / f"{species}.joblib"
    if not best_model_path.exists():
        raise FileNotFoundError(f"No model found at {best_model_path}")
    best_model = joblib.load(best_model_path)

    num_cols = dfs["train"].select_dtypes(include=np.number).columns
    feature_cols = [c for c in num_cols if not c.startswith("pred_") and c != "Fold"]

    if use_cv_train_predictions and "Fold" in dfs["train"].columns:
        n_samples = len(dfs["train"])
        oof_preds = np.empty(n_samples, dtype=object)
        oof_proba_f = np.empty(n_samples, dtype=float)
        oof_proba_m = np.empty(n_samples, dtype=float)

        for fold in sorted(dfs["train"]["Fold"].unique()):
            tr_idx = dfs["train"].index[dfs["train"]["Fold"] != fold]
            val_idx = dfs["train"].index[dfs["train"]["Fold"] == fold]

            X_tr = dfs["train"].loc[tr_idx, feature_cols]
            y_tr = dfs["train"].loc[tr_idx, "sex"].map(SEX_TO_INT)
            X_val = dfs["train"].loc[val_idx, feature_cols]

            mdl = clone(best_model).fit(X_tr, y_tr)
            preds = mdl.predict(X_val)
            proba = mdl.predict_proba(X_val)

            oof_preds[val_idx] = np.vectorize(INT_TO_SEX.get)(preds)
            oof_proba_f[val_idx] = proba[:, 0]
            oof_proba_m[val_idx] = proba[:, 1]

        dfs["train"]["pred_sex"] = oof_preds
        dfs["train"]["pred_proba_f"] = oof_proba_f
        dfs["train"]["pred_proba_m"] = oof_proba_m
    else:
        X_train = dfs["train"][feature_cols]
        y_train = dfs["train"]["sex"].map(SEX_TO_INT)
        mdl = clone(best_model).fit(X_train, y_train)
        proba = mdl.predict_proba(X_train)
        train_preds = mdl.predict(X_train)
        dfs["train"]["pred_sex"] = np.vectorize(INT_TO_SEX.get)(train_preds)
        dfs["train"]["pred_proba_f"] = proba[:, 0]
        dfs["train"]["pred_proba_m"] = proba[:, 1]

    X_train_full = dfs["train"][feature_cols]
    y_train_full = dfs["train"]["sex"].map(SEX_TO_INT)
    final_model = clone(best_model).fit(X_train_full, y_train_full)

    for split in ["test", "inference"]:
        if split in dfs:
            X_split = dfs[split][feature_cols]
            proba = final_model.predict_proba(X_split)
            preds = final_model.predict(X_split)
            dfs[split]["pred_sex"] = np.vectorize(INT_TO_SEX.get)(preds)
            dfs[split]["pred_proba_f"] = proba[:, 0]
            dfs[split]["pred_proba_m"] = proba[:, 1]

    all_df = pd.concat(dfs.values(), ignore_index=True)
    all_df.to_csv(csv_path, index=False)
    return all_df


# =============================================================================
# Hyperparameter search visualisation
# =============================================================================
def plot_hyperparam_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot a heatmap visualising mean CV accuracy across preprocessing options."""
    if df.empty:
        raise ValueError("DataFrame for heatmap is empty")

    col_map = {
        "select__method": "fs_method",
        "reduce_pre__method": "reduce_pre_method",
        "mean_test_score": "cv_balanced_accuracy",
        "mean_test_balanced_accuracy": "cv_balanced_accuracy",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "fs_method" not in df.columns:
        df["fs_method"] = "None"
    if "reduce_pre_method" not in df.columns:
        df["reduce_pre_method"] = "None"
    if "cv_balanced_accuracy" not in df.columns:
        raise KeyError("Missing column 'cv_balanced_accuracy' for heatmap")

    pivot = (
        df.pivot_table(
            index="fs_method",
            columns="reduce_pre_method",
            values="cv_balanced_accuracy",
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    apply_style()
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("Pre-dimensionality reducer")
    plt.ylabel("Feature selector")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hyperparam_heatmap.png"
    _savefig(out_file, "Mean CV accuracy by selector and reducer")
    plt.close()
    return out_file


# =============================================================================
# Batch prediction for all species
# =============================================================================
def predict_all_species(species_list: list[str] | None = None) -> pd.DataFrame:
    """Predict sex for all species and combine into a single CSV."""
    if species_list is None:
        species_list = [p.name for p in (DATA_DIR / "splits").iterdir() if p.is_dir()]

    valid_species: list[str] = []
    for species in species_list:
        if (DATA_DIR / "splits" / species).is_dir():
            valid_species.append(species)
        else:
            warnings.warn(
                f"Split directory for {species!r} not found – skipping.",
                UserWarning,
            )

    dfs = [predict_all(sp, prefer_generic=True, reuse_csv=False) for sp in valid_species]
    if not dfs:
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)
    out_csv = PATHS["sex_modelling"] / "tables" / "all_species_all_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)
    return all_df


# =============================================================================
# Confusion matrices (train/test)
# =============================================================================
def plot_confusion(df: pd.DataFrame, out_dir: str | Path | None = None) -> Path | None:
    """Plot confusion matrices for train (CV) and test predictions incl. counts."""
    apply_style()

    if "pred_sex" not in df.columns:
        raise KeyError("DataFrame contains no 'pred_sex' column")
    if "__split__" not in df.columns:
        raise KeyError("DataFrame is missing the '__split__' column")

    mapping = {"f": "F", "m": "M", "female": "F", "male": "M", "F": "F", "M": "M", 0: "F", 1: "M"}

    df_mapped = df.copy()
    df_mapped["true_label"] = df_mapped["sex"].map(mapping)
    df_mapped["pred_label"] = df_mapped["pred_sex"].map(mapping)

    train = df_mapped[
        (df_mapped["__split__"] == "train")
        & df_mapped["true_label"].isin(["F", "M"])
        & df_mapped["pred_label"].isin(["F", "M"])
    ]
    test = df_mapped[
        (df_mapped["__split__"] == "test")
        & df_mapped["true_label"].isin(["F", "M"])
        & df_mapped["pred_label"].isin(["F", "M"])
    ]
    if train.empty or test.empty:
        return None

    y_true_train, y_pred_train = train["true_label"], train["pred_label"]
    y_true_test, y_pred_test = test["true_label"], test["pred_label"]

    cm_train = confusion_matrix(y_true_train, y_pred_train, labels=["F", "M"])
    cm_test = confusion_matrix(y_true_test, y_pred_test, labels=["F", "M"])

    height = plt.rcParams["figure.figsize"][1] * 0.6
    fig, axes = plt.subplots(1, 2, figsize=(8, height), sharey=True)
    mats = [(cm_train, "a)"), (cm_test, "b)")]

    for ax, (cm, title) in zip(axes, mats):
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100

        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                c = cm[i, j]
                p = cm_perc[i, j]
                annot[i, j] = f"{c} ({p:.0f}%)"

        sns.heatmap(
            cm_perc,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=["Female", "Male"],
            yticklabels=["Female", "Male"],
            ax=ax,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Predicted Sex")
        if ax is axes[0]:
            ax.set_ylabel("True Sex")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticklabels(["Female", "Male"], rotation=0)
        ax.set_yticklabels(["Female", "Male"], rotation=0)

    outp = _ensure_out_dir(out_dir) / "confusion_train_test.png"
    _savefig(outp, "Confusion matrices for train (CV) and test")
    plt.close(fig)
    return outp


# =============================================================================
# Helper for majority/pivot sorting (kept for compatibility)
# =============================================================================
def build_pivot_sorted(sub: pd.DataFrame, col: str) -> tuple[pd.DataFrame, int]:
    tmp = sub.copy()
    tmp[col] = tmp[col].map(SEX_VALUE_MAP)

    if "sex" not in tmp.columns:
        raise KeyError("DataFrame lacks 'sex' column for true sex sorting")
    tmp["true_female"] = tmp["sex"].map(SEX_VALUE_MAP) == "Female"

    pivot = tmp.pivot_table(index="individual_id", columns=col, aggfunc="size", fill_value=0)
    pivot = pivot.reindex(columns=["Female", "Male"], fill_value=0)

    total_counts = pivot.sum(axis=1)
    female_ratio = pivot["Female"] / total_counts
    male_ratio = pivot["Male"] / total_counts

    true_female_ids = tmp.loc[tmp["true_female"], "individual_id"].unique()
    true_male_ids = tmp.loc[~tmp["true_female"], "individual_id"].unique()

    female_df = (
        pivot.loc[pivot.index.isin(true_female_ids)]
        .assign(ratio=female_ratio)
        .sort_values(by="ratio", ascending=False)
        .drop(columns="ratio")
    )
    male_df = (
        pivot.loc[pivot.index.isin(true_male_ids)]
        .assign(ratio=male_ratio)
        .sort_values(by="ratio", ascending=True)
        .drop(columns="ratio")
    )

    pivot_sorted = pd.concat([female_df, male_df])
    female_count = len(female_df)
    return pivot_sorted, female_count


def plot_hist_predsex_train_test(df: pd.DataFrame) -> None:
    """Placeholder kept for compatibility (no-op if unused)."""
    apply_style()
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols or "__split__" not in df.columns:
        return
    # no plot generated → nothing to save


# =============================================================================
# Inference bar plot (per location/trail)
# =============================================================================
def plot_inference(df: pd.DataFrame, out_dir: str | Path | None = None) -> list[Path]:
    """Plot predicted sex counts for inference split with Location + Trail labels."""
    apply_style()
    outputs: list[Path] = []

    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols or "inference" not in df["__split__"].unique():
        return outputs

    sub = df[df["__split__"] == "inference"].copy()
    outp_dir = _ensure_out_dir(out_dir)

    for col in pred_cols:
        sub[col] = sub[col].map(CONFIG["visualisation"]["sex"]["value_map"])

        # Extract trailing number from 'trail'
        sub["trail_num"] = sub["trail"].str.extract(r"(\d+)$")
        sub["loc_trail"] = sub["location"] + " " + sub["trail_num"].fillna("")

        pivot = sub.pivot_table(index="loc_trail", columns=col, aggfunc="size", fill_value=0)

        colors = [CONFIG["visualisation"]["sex"]["colors"].get(c, "#333333") for c in pivot.columns]
        ax = pivot.plot.bar(stacked=True, figsize=(10, 4), color=colors)

        plt.xlabel("Location & Trail")
        plt.ylabel("Count")
        ax.legend(title="Predicted", labels=pivot.columns)
        plt.xticks(rotation=45, ha="right")

        model = col[len("pred_") : -len("_sex")] or "pred"
        outfile = outp_dir / f"inference_bar_{model}.png"
        _savefig(outfile, f"Inference counts — {model}")
        plt.close()
        outputs.append(outfile)

    return outputs


def plot_confusion_and_inference(df: pd.DataFrame, out_dir: str | Path | None = None) -> dict:
    """Wrapper calling :func:`plot_confusion` and :func:`plot_inference` and saving figs."""
    out = {}
    out["confusion"] = plot_confusion(df, out_dir=out_dir)
    out["inference"] = plot_inference(df, out_dir=out_dir)
    return out


# =============================================================================
# Quality heatmaps
# =============================================================================
def plot_quality_grouped(df: pd.DataFrame, out_dir: str | Path | None = None) -> list[Path]:
    """Plot quality heatmaps based on majority decision per Trail or Individual."""
    apply_style()
    outputs: list[Path] = []
    outp_dir = _ensure_out_dir(out_dir)

    splits = ["train", "test"] if "__split__" in df.columns else [None]
    for split in splits:
        df_sub = df if split is None else df[df["__split__"] == split]
        if df_sub.empty:
            continue

        df_plot = df_sub[df_sub["sex"].isin(["f", "m"])].copy()
        df_plot["true_label"] = map_sex(df_plot["sex"])

        pred_cols = [c for c in df_plot if c.startswith("pred_") and c.endswith("_sex")]
        if not pred_cols:
            raise KeyError("No prediction label columns found in dataframe")

        for col in pred_cols:
            key = col.split("_")[1]
            df_plot[f"pred_label_{key}"] = df_plot[col].map({0: "Female", 1: "Male"})

        def classify_majority(group: pd.DataFrame) -> str:
            correct_ratio = (group[pred_label_cols].eq(group["true_label"], axis=0).any(axis=1)).mean()
            if correct_ratio >= 0.9:
                return "High"
            elif correct_ratio >= 0.7:
                return "Moderate"
            elif correct_ratio >= 0.5:
                return "Low"
            else:
                return "Misclassified"

        pred_label_cols = [c for c in df_plot if c.startswith("pred_label_")]

        cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
        height = plt.rcParams["figure.figsize"][1] * 0.75
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(plt.rcParams["figure.figsize"][0], height]),
            sharey=True,
        )

        for ax, (tag, cols) in zip(
            axes,
            [
                ("c)", ["trail", "true_label"]),
                ("d)", ["individual_id", "true_label"]),
            ],
        ):
            acc = df_plot.groupby(cols).apply(classify_majority).reset_index(name="Class")

            pivot = (
                acc.pivot_table(
                    index="true_label",
                    columns="Class",
                    values=cols[0],
                    aggfunc="count",
                    fill_value=0,
                )
                .reindex(columns=["High", "Moderate", "Low", "Misclassified"], fill_value=0)
            )

            proportions = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
            counts = pivot.astype(int)
            total = counts.values.sum()

            annot = counts.copy().astype(str)
            for i in counts.index:
                for j in counts.columns:
                    v = counts.at[i, j]
                    annot.at[i, j] = f"{v} ({v/total:.0%})" if v > 0 else ""

            sns.heatmap(
                proportions,
                annot=annot,
                fmt="",
                cmap=cmap,
                vmin=0,
                vmax=1,
                linewidths=0.5,
                linecolor="gray",
                ax=ax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(["Female", "Male"], rotation=0)
            ax.set_title(tag, loc="left", fontweight="bold")
            ax.set_xlabel("Prediction Confidence")
            if ax is axes[0]:
                ax.set_ylabel("Sex")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

        fname = f"quality_grouped_{split or 'all'}.png"
        outfile = outp_dir / fname
        _savefig(outfile, f"Prediction quality grouped — {split or 'all'}")
        plt.close(fig)
        outputs.append(outfile)

    return outputs


def _plot_quality_heatmaps_single(
    df_sub: pd.DataFrame,
    pred_col: str,
    proba_cols: list[str],
    title: str | None = None,
    group_by: str | None = None,
    *,
    out_dir: str | Path | None = None,
    filename: str | None = None,
) -> Path:
    """Plot prediction quality heatmaps for a single model and split."""
    df = df_sub.copy()
    df["pred_label"] = df[pred_col].map({0: "Female", 1: "Male"})
    df["Correct"] = df["pred_label"] == df["true_label"]
    df["Max_Prob"] = df[proba_cols].max(axis=1)
    df["Quality"] = df["Max_Prob"].apply(lambda p: "High" if p > 0.9 else ("Moderate" if p > 0.7 else "Low"))

    if group_by is not None:
        grouped = df.groupby([group_by, "true_label"])
        df = grouped.agg(correct_rate=("Correct", "mean")).reset_index()
        df["Correct"] = df["correct_rate"] > 0.5
        df["Quality"] = df["correct_rate"].apply(
            lambda p: (
                "High"
                if p > 0.9
                else ("Moderate" if p > 0.7 else ("Low" if p > 0.5 else "Misclassified"))
            )
        )

    idx = pd.MultiIndex.from_product([["Female", "Male"], [True, False]], names=["true_label", "Correct"])
    columns = ["High", "Moderate", "Low"]
    if group_by is not None:
        columns.append("Misclassified")
    counts = (
        df.groupby(["true_label", "Correct", "Quality"]).size().unstack(fill_value=0)
    ).reindex(index=idx, columns=columns, fill_value=0)

    if counts.values.sum() == 0:
        # still create empty figure to be explicit
        apply_style()
        plt.figure()
        outp = _ensure_out_dir(out_dir) / (filename or "quality_heatmap_empty.png")
        _savefig(outp, title or "No data to plot")
        plt.close()
        return outp

    normed = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    def make_annot(block: pd.DataFrame) -> pd.DataFrame:
        total = block.values.sum()
        return block.applymap(lambda x: (f"{int(x)}\n({int(round(x / total * 100))}%)" if total > 0 else "0\n(0%)"))

    annot_corr = make_annot(counts.xs(True, level="Correct"))
    annot_incorr = make_annot(counts.xs(False, level="Correct"))

    green_cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
    red_cmap = LinearSegmentedColormap.from_list("red", ["white", "crimson"])

    apply_style()
    height = plt.rcParams["figure.figsize"][1] * 0.75
    fig, axes = plt.subplots(1, 2, figsize=(plt.rcParams["figure.figsize"][0], height), sharey=True)
    sns.heatmap(
        normed.xs(True, level="Correct"),
        annot=annot_corr,
        fmt="",
        cmap=green_cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        ax=axes[0],
        cbar=True,
    )
    axes[0].set_title("a)", loc="left", fontweight="bold")
    axes[0].set(ylabel="Sex")
    sns.heatmap(
        normed.xs(False, level="Correct"),
        annot=annot_incorr,
        fmt="",
        cmap=red_cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title("b)", loc="left", fontweight="bold")
    axes[1].set(ylabel="")
    for ax in axes:
        ax.set_xlabel("Prediction Confidence")
        labels = ["High", "Moderate", "Low"]
        if group_by is not None:
            labels.append("Misclassified")
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(["Female", "Male"], rotation=0)

    if title:
        fig.suptitle(title, y=1.02)

    outp = _ensure_out_dir(out_dir) / (filename or "quality_heatmap.png")
    _savefig(outp, title or "Prediction quality heatmaps")
    plt.close(fig)
    return outp


def plot_quality_heatmaps(
    df: pd.DataFrame,
    pred_col: str | None = None,
    proba_cols: list[str] | None = None,
    title: str | None = None,
    group_by: str | None = None,
    *,
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Plot prediction-quality heatmaps and save each figure."""
    apply_style()
    outputs: list[Path] = []

    if pred_col is not None and proba_cols is not None:
        splits = ["train", "test"] if "__split__" in df.columns else [None]
        for split in splits:
            df_sub = df if split is None else df[df["__split__"] == split]
            if df_sub.empty:
                continue
            df_sub = df_sub.copy()
            df_sub["true_label"] = map_sex(df_sub["sex"])
            sub_title = title if title is not None else (f"{split} set" if split else None)
            fname = f"quality_heatmap_{split or 'all'}.png"
            outp = _plot_quality_heatmaps_single(
                df_sub,
                pred_col,
                proba_cols,
                sub_title,
                group_by,
                out_dir=out_dir,
                filename=fname,
            )
            outputs.append(outp)
        return outputs

    # Automatic generation for all models and splits
    df = df.copy()
    df["true_label"] = map_sex(df["sex"])

    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")

    split_order = [s for s in ["train", "test"] if s in df["__split__"].unique()]
    for split in split_order:
        for col in pred_cols:
            model = col[len("pred_") : -len("_sex")] or "pred"
            model_prefix = f"{model}_" if model else ""
            probs = [f"pred_{model_prefix}proba_f", f"pred_{model_prefix}proba_m"]
            df_sub = df[df["__split__"] == split].copy()
            df_sub = df_sub[df_sub["sex"].isin(["f", "m"])]
            df_sub = df_sub[df_sub[col].isin([0, 1])]
            if df_sub.empty:
                continue
            fname = f"quality_heatmap_{model}_{split}.png"
            outp = _plot_quality_heatmaps_single(
                df_sub,
                col,
                probs,
                title=f"{model} — {split}",
                group_by=group_by,
                out_dir=out_dir,
                filename=fname,
            )
            outputs.append(outp)
    return outputs


def plot_model_quality_heatmaps(df: pd.DataFrame, out_dir: str | Path | None = None) -> list[Path]:
    """Deprecated wrapper for :func:`plot_quality_heatmaps` (still saves)."""
    warnings.warn(
        "plot_model_quality_heatmaps is deprecated; use plot_quality_heatmaps",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_quality_heatmaps(df, out_dir=out_dir)


# =============================================================================
# Individual probability histograms
# =============================================================================
def plot_individual_probabilities(df: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Plot distribution of predicted sex probabilities for train and test."""
    apply_style()
    out_path = _ensure_out_dir(out_dir)
    outputs: list[Path] = []

    proba_cols = [c for c in df.columns if c.endswith("_proba_m")]
    if not proba_cols:
        raise ValueError("DataFrame contains no probability columns ending with '_proba_m'.")

    palette = {
        "Female": CONFIG["visualisation"]["sex"]["colors"]["Female"],
        "Male": CONFIG["visualisation"]["sex"]["colors"]["Male"],
        "Unknown": CONFIG["visualisation"]["sex"]["colors"].get("Unknown", "#333333"),
    }

    splits = ["train", "test"] if "__split__" in df.columns else [None]
    for split in splits:
        df_sub = df if split is None else df[df["__split__"] == split]
        if df_sub.empty:
            continue
        agg = {c: "mean" for c in proba_cols}
        agg["sex"] = "first"
        grouped = df_sub.groupby("individual_id").agg(agg).reset_index()
        grouped["sex_std"] = map_sex(grouped["sex"])

        for col in proba_cols:
            model = col[len("pred_") : -len("_proba_m")] or "pred"
            height = plt.rcParams["figure.figsize"][1] * 0.75
            plt.figure(figsize=(plt.rcParams["figure.figsize"][0], height))
            sns.histplot(
                grouped,
                x=col,
                hue="sex_std",
                element="step",
                stat="density",
                common_norm=False,
                palette=palette,
            )
            plt.xticks(rotation=0)
            plt.xlabel("Predicted probability male")
            plt.ylabel("Density")

            suffix = f"_{split}" if split is not None else ""
            file = out_path / f"{model}_individual_probabilities{suffix}.png"
            _savefig(file, f"Predicted male probability for {model}{suffix.replace('_', ' ')}".strip())
            plt.close()
            outputs.append(file)

    return outputs
