import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import LinearSegmentedColormap
from FIT_python.config import DATA_DIR, RESULTS_DATA_DIR


DEFAULT_SPECIES = "eurasian_otter"

def _base_paths(species: str = DEFAULT_SPECIES) -> tuple[Path, Path, Path]:
    """Return split dir, model dir and output csv for a species."""
    splits = DATA_DIR / "splits" / species
    models = RESULTS_DATA_DIR / f"{species}_random_search_standard_metrics"
    csv    = models / f"{species}_all_predictions.csv"
    return splits, models, csv

# === Modelle definieren ===
MODELS = {
    "balanced_acc": "best_balanced_test_acc",
    "maj_pct":      "best_maj_test_pct",
    "neg_log_loss": "best_mean_test_neg_log_loss",
    "accuracy":     "best_accuracy_test",
}

# === CSV erzeugen (einmal laufen lassen) ===
def predict_all(species: str = DEFAULT_SPECIES) -> pd.DataFrame:
    """Load models for ``species`` and return dataframe with predictions."""
    splits_dir, models_dir, csv_path = _base_paths(species)

    splits = {n: splits_dir / f"{n}.parquet" for n in ["train", "test", "inference"]}
    dfs = {name: pd.read_parquet(p) for name, p in splits.items()}
    # Säubere Spaltennamen
    for name, df in dfs.items():
        df.columns = (
            df.columns.str.replace(r"[.\-]", "_", regex=True)
                       .str.replace("T", "t")
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

    dfs = [predict_all(species) for species in species_list]
    all_df = pd.concat(dfs, ignore_index=True)
    out_csv = RESULTS_DATA_DIR / "all_species_all_predictions.csv"
    all_df.to_csv(out_csv, index=False)
    return all_df

# === Plots für Confusion & Inference ===
def plot_confusion_and_inference(df):
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    for col in pred_cols:
        model = col.split("_")[1]
        prob_f = col.replace("_sex", "_proba_f")
        prob_m = col.replace("_sex", "_proba_m")
        for split in df["__split__"].unique():
            sub = df[df["__split__"] == split]
            if split != "inference":
                y_true = sub["sex"].map({"f": "F", "m": "M"})
                y_pred = sub[col].map({0: "F", 1: "M"})
                cm = confusion_matrix(y_true, y_pred, labels=["F", "M"])
                acc = accuracy_score(y_true, y_pred)
                plt.figure(figsize=(4,4))
                sns.heatmap(cm / cm.sum(axis=1, keepdims=True),
                            annot=True, fmt=".2f", cmap="Blues")
                plt.title(f"{model} — {split} (Acc {acc:.1%})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.show()
            else:
                pivot = (sub
                         .pivot_table(index="trail", columns=col, aggfunc="size", fill_value=0)
                         .rename(columns={0: "F", 1: "M"}))
                pivot.plot.bar(stacked=True, figsize=(6,3), color={"F": "#C08080", "M": "#8080C0"})
                plt.title(f"{model} — inference")
                plt.xlabel("Trail")
                plt.ylabel("Count")
                plt.legend(title="Predicted")
                plt.tight_layout()
                plt.show()

# === Plots für Qualitäts-Heatmaps ===
def plot_quality(df):
    # true_label
    df = df.copy()
    df["true_label"] = df["sex"].map({"f": "F", "m": "M"})
    # pred_label & Correct pro Modell
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    for col in pred_cols:
        key = col.split("_")[1]
        df[f"pred_label_{key}"] = df[col].map({0: "F", 1: "M"})
    # True wenn irgendein Modell richtig war
    df["Correct"] = np.any([
        df[f"pred_label_{k}"] == df["true_label"] for k in MODELS
    ], axis=0)

    # Klassifizierung
    def classify(group):
        acc = group["Correct"].mean()
        if acc >= 0.9:    return "High"
        if acc >= 0.7:    return "Moderate"
        if acc >= 0.5:    return "Low"
        return "Misclassified"

    # Erstelle Heatmaps a) und b)
    cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
    for tag, cols in [("a)", ["trail","true_label"]), ("b)", ["individual_id","true_label"])]:
        kvals = (df.groupby(cols, group_keys=False)
                  .apply(classify)
                  .reset_index(name="Class"))
        pivot = (kvals
                 .pivot_table(index="true_label", columns="Class", values=cols[0], aggfunc="count", fill_value=0)
                 .reindex(columns=["High","Moderate","Low","Misclassified"], fill_value=0))
        proportions = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
        counts = pivot.astype(int)
        total = counts.values.sum()

        # Annotation per cell
        annot = counts.copy().astype(str)
        for i in counts.index:
            for j in counts.columns:
                v = counts.at[i,j]
                annot.at[i,j] = f"{v}\n({v/total:.0%})" if v>0 else ""

        plt.figure(figsize=(4,4))
        sns.heatmap(proportions, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=1,
                    linewidths=0.5, linecolor="gray")
        plt.title(tag, loc="left", fontweight="bold")
        plt.xlabel("Quality")
        plt.ylabel("True Sex")
        # Kommentar rechts
        plt.gca().text(1.02, 0.5, "(trail)" if tag=="a)" else "(animal)",
                       transform=plt.gca().transAxes, va="center", color="gray")
        plt.tight_layout()
        plt.show()
