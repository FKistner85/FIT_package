import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import LinearSegmentedColormap
from FIT_python.config import DATA_DIR, RESULTS_DATA_DIR
from FIT_python.plot_style import TEST_COLORS
from FIT_python.plot_style import apply_style


DEFAULT_SPECIES = "eurasian_otter"


def _base_paths(
    species: str = DEFAULT_SPECIES, prefer_generic: bool = False
) -> tuple[Path, Path, Path]:
    """Return split dir, model dir and output csv for a species.

    If ``prefer_generic`` is ``True`` or a species specific directory does
    not exist, fall back to ``random_search_standard_metrics``. This enables
    notebooks that work on a single species to load custom models while the
    all-species notebook can still rely on the generic directory.
    """
    splits = DATA_DIR / "splits" / species
    specific = RESULTS_DATA_DIR / f"{species}_random_search_standard_metrics"
    generic = RESULTS_DATA_DIR / "random_search_standard_metrics"
    models = generic if prefer_generic else specific
    if not models.exists():
        models = specific if prefer_generic else generic
    csv = models / f"{species}_all_predictions.csv"
    return splits, models, csv


# === Modelle definieren ===
MODELS = {
    "balanced_acc": "best_balanced_test_acc",
    "maj_pct": "best_maj_test_pct",
    "neg_log_loss": "best_mean_test_neg_log_loss",
    "accuracy": "best_accuracy_test",
}


# === CSV erzeugen (einmal laufen lassen) ===
def predict_all(
    species: str = DEFAULT_SPECIES,
    prefer_generic: bool = False,
    include_inference: bool = True,
) -> pd.DataFrame:
    """Return dataframe with model predictions for ``species``.

    Parameters
    ----------
    species:
        The species folder under ``data/splits``.
    prefer_generic:
        If ``True``, models are loaded from the shared
        ``random_search_standard_metrics`` directory when present.
    include_inference:
        Include the ``inference`` split if the corresponding parquet exists.
    """
    splits_dir, models_dir, csv_path = _base_paths(species, prefer_generic)

    split_names = ["train", "test"]
    if include_inference and (splits_dir / "inference.parquet").exists():
        split_names.append("inference")

    splits = {n: splits_dir / f"{n}.parquet" for n in split_names}
    dfs = {name: pd.read_parquet(p) for name, p in splits.items()}
    # Säubere Spaltennamen
    for name, df in dfs.items():
        df.columns = df.columns.str.replace(r"[.\-]", "_", regex=True).str.replace(
            "T", "t"
        )
        df["__split__"] = name

    # Vorhersagen pro Modell
    for key, subdir in MODELS.items():
        clf = load(models_dir / subdir / f"{species}.joblib")
        for df in dfs.values():
            X = df.select_dtypes(include=np.number)
            df[f"pred_{key}_sex"] = clf.predict(X)
            proba = clf.predict_proba(X)
            df[f"pred_{key}_proba_f"] = proba[:, 0]
            df[f"pred_{key}_proba_m"] = proba[:, 1]

    # CSV speichern
    all_df = pd.concat(dfs.values(), ignore_index=True)
    all_df.to_csv(csv_path, index=False)
    return all_df


def predict_all_species(species_list: list[str] | None = None) -> pd.DataFrame:
    """Predict sex for all species and combine into a single CSV."""
    if species_list is None:
        species_list = [p.name for p in (DATA_DIR / "splits").iterdir() if p.is_dir()]

    dfs = [predict_all(species, prefer_generic=True) for species in species_list]
    all_df = pd.concat(dfs, ignore_index=True)
    out_csv = RESULTS_DATA_DIR / "all_species_all_predictions.csv"
    all_df.to_csv(out_csv, index=False)
    return all_df


# === Plots für Confusion & Inference ===
def plot_confusion_and_inference(df: pd.DataFrame) -> None:
    """Visualise predictions for train/test and inference splits."""
    apply_style()
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")

    # Only compute confusion matrices on train and test sets
    split_order = [s for s in ["train", "test"] if s in df["__split__"].unique()]
    for col in pred_cols:
        model = col.split("_")[1]
        for split in split_order:
            sub = df[df["__split__"] == split]
            if sub.empty:
                continue
            y_true = sub["sex"].map({"f": "F", "m": "M"})
            y_pred = sub[col].map({0: "F", 1: "M"})
            cm = confusion_matrix(y_true, y_pred, labels=["F", "M"])
            acc = accuracy_score(y_true, y_pred)
            plt.figure(figsize=(4, 4))
            sns.heatmap(
                cm / cm.sum(axis=1, keepdims=True),
                annot=True,
                fmt=".2f",
                cmap="Blues",
            )
            plt.title(f"{model} — {split} (Acc {acc:.1%})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.show()

        # Bar plot for inference predictions if present
        if "inference" in df["__split__"].unique():
            sub = df[df["__split__"] == "inference"]
            pivot = sub.pivot_table(
                index="trail", columns=col, aggfunc="size", fill_value=0
            ).rename(columns={0: "F", 1: "M"})
            pivot.plot.bar(stacked=True, figsize=(6, 3), color=TEST_COLORS)
            plt.title(f"{model} — inference")
            plt.xlabel("Trail")
            plt.ylabel("Count")
            plt.legend(title="Predicted")
            plt.tight_layout()
            plt.show()


# === Plots für Qualitäts-Heatmaps ===
def plot_quality(df):
    apply_style()
    # true_label
    df = df.copy()
    df["true_label"] = df["sex"].map({"f": "F", "m": "M"})
    # pred_label & Correct pro Modell
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    for col in pred_cols:
        key = col.split("_")[1]
        df[f"pred_label_{key}"] = df[col].map({0: "F", 1: "M"})

    # True wenn irgendein Modell richtig war

    pred_label_cols = [c for c in df if c.startswith("pred_label_")]
    if not pred_label_cols:
        raise KeyError("No prediction label columns found in dataframe")

    df["Correct"] = np.any([df[c] == df["true_label"] for c in pred_label_cols], axis=0)

    # Klassifizierung
    def classify(group):
        acc = group["Correct"].mean()
        if acc >= 0.9:
            return "High"
        if acc >= 0.7:
            return "Moderate"
        if acc >= 0.5:
            return "Low"
        return "Misclassified"

    # Erstelle Heatmaps a) und b)
    cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
    for tag, cols in [
        ("a)", ["trail", "true_label"]),
        ("b)", ["individual_id", "true_label"]),
    ]:
        kvals = (
            df.groupby(cols, group_keys=False).apply(classify).reset_index(name="Class")
        )
        pivot = kvals.pivot_table(
            index="true_label",
            columns="Class",
            values=cols[0],
            aggfunc="count",
            fill_value=0,
        ).reindex(columns=["High", "Moderate", "Low", "Misclassified"], fill_value=0)
        proportions = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
        counts = pivot.astype(int)
        total = counts.values.sum()

        # Annotation per cell
        annot = counts.copy().astype(str)
        for i in counts.index:
            for j in counts.columns:
                v = counts.at[i, j]
                annot.at[i, j] = f"{v}\n({v/total:.0%})" if v > 0 else ""

        plt.figure(figsize=(4, 4))
        sns.heatmap(
            proportions,
            annot=annot,
            fmt="",
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(tag, loc="left", fontweight="bold")
        plt.xlabel("Quality")
        plt.ylabel("True Sex")
        # Kommentar rechts
        plt.gca().text(
            1.02,
            0.5,
            "(trail)" if tag == "a)" else "(animal)",
            transform=plt.gca().transAxes,
            va="center",
            color="gray",
        )
        plt.tight_layout()
        plt.show()


def plot_individual_probabilities(df: pd.DataFrame, out_dir: str | Path):
    """Plot distribution of predicted sex probabilities for each individual."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    proba_cols = [c for c in df if c.startswith("pred_") and c.endswith("_proba_m")]
    if not proba_cols:
        raise ValueError(
            "DataFrame contains no probability columns ending with '_proba_m'."
        )
    agg = {c: "mean" for c in proba_cols}
    agg["sex"] = "first"
    grouped = df.groupby("individual_id").agg(agg).reset_index()
    for col in proba_cols:
        model = col.split("_")[1]
        plt.figure(figsize=(5, 3))
        sns.histplot(
            grouped, x=col, hue="sex", element="step", stat="density", common_norm=False
        )
        plt.xlabel("Predicted probability male")
        plt.ylabel("Density")
        plt.title(model)
        plt.tight_layout()
        plt.savefig(out_path / f"{model}_individual_probabilities.png")
        plt.close()


def plot_quality_heatmaps(
    df_sub: pd.DataFrame,
    pred_col: str,
    proba_cols: list[str],
    title: str | None = None,
) -> None:
    """Plot prediction quality heatmaps for a single model and split."""

    df = df_sub.copy()
    df["pred_label"] = df[pred_col].map({0: "F", 1: "M"})
    df["Correct"] = df["pred_label"] == df["true_label"]
    df["Max_Prob"] = df[proba_cols].max(axis=1)
    df["Quality"] = df["Max_Prob"].apply(
        lambda p: "High" if p > 0.9 else ("Moderate" if p > 0.7 else "Low")
    )

    idx = pd.MultiIndex.from_product(
        [["F", "M"], [True, False]], names=["true_label", "Correct"]
    )
    counts = (
        df.groupby(["true_label", "Correct", "Quality"]).size().unstack(fill_value=0)
    ).reindex(index=idx, columns=["High", "Moderate", "Low"], fill_value=0)

    if counts.values.sum() == 0:
        print("    → No data to plot")
        return

    normed = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    def make_annot(block: pd.DataFrame) -> pd.DataFrame:
        total = block.values.sum()
        return block.applymap(
            lambda x: f"{int(x)}\n({int(round(x / total * 100))}%)" if total > 0 else "0\n(0%)"
        )

    annot_corr = make_annot(counts.xs(True, level="Correct"))
    annot_incorr = make_annot(counts.xs(False, level="Correct"))

    green_cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
    red_cmap = LinearSegmentedColormap.from_list("red", ["white", "crimson"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
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
    axes[0].set(title="Correct Predictions", ylabel="True Label")
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
    axes[1].set(title="Incorrect Predictions")
    for ax in axes:
        ax.set_xlabel("Prediction Quality")
        ax.set_xticklabels(["High", "Moderate", "Low"], rotation=0)
        ax.set_yticklabels(["F", "M"], rotation=0)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_model_quality_heatmaps(df: pd.DataFrame) -> None:
    """Plot prediction-quality heatmaps for each best model and split."""

    apply_style()

    df = df.copy()
    df["true_label"] = df["sex"].map({"f": "F", "m": "M"})

    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")

    split_order = [s for s in ["train", "test"] if s in df["__split__"].unique()]
    for split in split_order:
        for pred_col in pred_cols:
            model = pred_col[len("pred_") : -len("_sex")]
            proba_cols = [f"pred_{model}_proba_f", f"pred_{model}_proba_m"]
            df_sub = df[df["__split__"] == split].copy()
            df_sub = df_sub[df_sub[pred_col].isin([0, 1])]
            if df_sub.empty:
                continue
            plot_quality_heatmaps(
                df_sub,
                pred_col,
                proba_cols,
                title=f"{model} — {split}",
            )

