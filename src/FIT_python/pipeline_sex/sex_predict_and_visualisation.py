import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import warnings
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from FIT_python.config import DATA_DIR, RESULTS_DATA_DIR
from FIT_python.plot_style import SEX_COLORS
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
    models_dir: Path | None = None,
    csv_path: Path | None = None,
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
    splits_dir, models_def, csv_def = _base_paths(species, prefer_generic)
    models_dir = models_dir or models_def
    csv_path = csv_path or csv_def

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
            num_cols = df.select_dtypes(include=np.number).columns
            feature_cols = [
                c for c in num_cols if not c.startswith("pred_") and c != "Fold"
            ]
            X = df[feature_cols]
            df[f"pred_{key}_sex"] = clf.predict(X)
            proba = clf.predict_proba(X)
            df[f"pred_{key}_proba_f"] = proba[:, 0]
            df[f"pred_{key}_proba_m"] = proba[:, 1]

    # CSV speichern
    all_df = pd.concat(dfs.values(), ignore_index=True)
    all_df.to_csv(csv_path, index=False)
    return all_df


def plot_hyperparam_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot a heatmap visualising mean CV accuracy across preprocessing options."""
    from FIT_python.plot_style import apply_style

    if df.empty:
        raise ValueError("DataFrame for heatmap is empty")

    required = {"fs_method", "reduce_pre_method", "cv_balanced_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for heatmap: {sorted(missing)}")

    # Pivot dynamically so the heatmap adapts to available hyperparameter values
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
    from FIT_python.caption_utils import save_caption

    plt.tight_layout()
    plt.savefig(out_file)
    save_caption(out_file, "Mean CV accuracy by selector and reducer")
    plt.close()
    return out_file


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
def plot_confusion(
    df: pd.DataFrame,
    metric: str | None = None,
    out_file: str | Path | None = None,
    caption: str | None = None,
) -> None:
    """Plot CV vs. test confusion matrices for each model."""
    apply_style()
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")

    models = {
        c[len("pred_") : -len("_sex")] for c in pred_cols if not c.endswith("_cv_sex")
    }
    if metric is not None:
        models = [metric]
    for mk in sorted(models):
        test_col = f"pred_{mk}_sex"
        cv_col = f"pred_{mk}_cv_sex"
        if test_col not in df.columns:
            continue

        train = df[(df["__split__"] == "train") & df["sex"].isin(["f", "m"])]
        test = df[(df["__split__"] == "test") & df["sex"].isin(["f", "m"])]
        if train.empty or test.empty:
            continue

        y_true_train = train["sex"].map({"f": "F", "m": "M"})
        y_true_test = test["sex"].map({"f": "F", "m": "M"})

        y_pred_train = (
            train[cv_col].map({0: "F", 1: "M"}) if cv_col in df.columns else None
        )
        y_pred_test = test[test_col].map({0: "F", 1: "M"})

        cm_test = confusion_matrix(y_true_test, y_pred_test, labels=["F", "M"])
        if y_pred_train is not None:
            cm_train = confusion_matrix(y_true_train, y_pred_train, labels=["F", "M"])
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
            mats = [(cm_train, "CV (train)"), (cm_test, "Test")]
        else:
            fig, axes = plt.subplots(1, 1, figsize=(4, 4))
            axes = [axes]
            mats = [(cm_test, "Test")]

        for ax, (cm, title) in zip(axes, mats):
            sns.heatmap(
                cm / cm.sum(axis=1, keepdims=True),
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=["Female", "Male"],
                yticklabels=["Female", "Male"],
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel("Predicted")
            if ax is axes[0]:
                ax.set_ylabel("True")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

        from FIT_python.caption_utils import save_caption
        from sklearn.metrics import accuracy_score

        acc = accuracy_score(
            test["sex"].map({"f": 0, "m": 1}),
            y_pred_test,
        )
        plt.tight_layout()
        if out_file:
            out_file = Path(out_file)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_file)
            caption = caption or f"{mk} accuracy {acc:.1%}"
            save_caption(out_file, caption)
            plt.close(fig)
        else:
            plt.show()


def plot_inference(df: pd.DataFrame) -> None:
    """Plot predicted sex counts for the inference split."""
    apply_style()
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")
    if "inference" not in df["__split__"].unique():
        return

    sub = df[df["__split__"] == "inference"]
    for col in pred_cols:
        pivot = sub.pivot_table(
            index="trail", columns=col, aggfunc="size", fill_value=0
        ).rename(columns={0: "F", 1: "M"})
        colors = [SEX_COLORS.get(c, "#333333") for c in pivot.columns]
        ax = pivot.plot.bar(stacked=True, figsize=(6, 3), color=colors)
        plt.xlabel("Trail")
        plt.ylabel("Count")
        ax.legend(title="Predicted", labels=["Female", "Male"])
        plt.tight_layout()
        plt.show()


def plot_confusion_and_inference(df: pd.DataFrame) -> None:
    """Backward compatible wrapper calling :func:`plot_confusion` and :func:`plot_inference`."""
    plot_confusion(df)
    plot_inference(df)


# === Plots für Qualitäts-Heatmaps ===
def plot_quality(df):
    apply_style()
    # true_label
    df = df.copy()
    df = df[df["sex"].isin(["f", "m"])]
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

    # Erstelle Heatmaps a) und b) nebeneinander
    cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, (tag, cols) in zip(
        axes,
        [
            ("a)", ["trail", "true_label"]),
            ("b)", ["individual_id", "true_label"]),
        ],
    ):
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
        ax.set_title(tag, loc="left", fontweight="bold")
        ax.set_xlabel("Quality")
        if ax is axes[0]:
            ax.set_ylabel("True Sex")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    # previously annotated as "(trail)" or "(animal)" on the right side of
    # the plot. These labels caused visual artefacts in the heatmaps and have
    # been removed.
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
    grouped["sex_std"] = grouped["sex"].map(
        lambda s: (
            "Female"
            if str(s).lower().startswith("f")
            else "Male" if str(s).lower().startswith("m") else "Unknown"
        )
    )
    palette = {
        "Female": SEX_COLORS["F"],
        "Male": SEX_COLORS["M"],
        "Unknown": SEX_COLORS.get("Unknown", "#333333"),
    }
    for col in proba_cols:
        model = col.split("_")[1]
        plt.figure(figsize=(5, 3))
        sns.histplot(
            grouped,
            x=col,
            hue="sex_std",
            element="step",
            stat="density",
            common_norm=False,
            palette=palette,
        )
        plt.xlabel("Predicted probability male")
        plt.ylabel("Density")
        from FIT_python.caption_utils import save_caption

        plt.tight_layout()
        file = out_path / f"{model}_individual_probabilities.png"
        plt.savefig(file)
        save_caption(file, f"Predicted male probability for {model}")
        plt.close()


def _plot_quality_heatmaps_single(
    df_sub: pd.DataFrame,
    pred_col: str,
    proba_cols: list[str],
    title: str | None = None,
    out_file: str | Path | None = None,
    caption: str | None = None,
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
            lambda x: (
                f"{int(x)}\n({int(round(x / total * 100))}%)"
                if total > 0
                else "0\n(0%)"
            )
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
    axes[0].set(ylabel="True Label")
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
    axes[1].set()
    for ax in axes:
        ax.set_xlabel("Prediction Quality")
        ax.set_xticklabels(["High", "Moderate", "Low"], rotation=0)
        ax.set_yticklabels(["Female", "Male"], rotation=0)
    # no super title so subfigures can be labelled externally
    from FIT_python.caption_utils import save_caption

    plt.tight_layout()
    if out_file:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
        save_caption(out_file, caption or (title or "Quality heatmap"))
        plt.close(fig)
    else:
        plt.show()


def plot_quality_heatmaps(
    df: pd.DataFrame,
    pred_col: str | None = None,
    proba_cols: list[str] | None = None,
    title: str | None = None,
    out_file: str | Path | None = None,
) -> None:
    """Plot prediction-quality heatmaps.

    Parameters ``pred_col`` and ``proba_cols`` can be used to display the
    heatmap for a specific model and split.  When they are omitted, the
    function behaves like the other plotting helpers and iterates over all
    available models and the ``train``/``test`` splits automatically.
    """

    apply_style()

    if pred_col is not None and proba_cols is not None:
        df_sub = df.copy()
        df_sub["true_label"] = df_sub["sex"].map({"f": "F", "m": "M"})
        _plot_quality_heatmaps_single(
            df_sub,
            pred_col,
            proba_cols,
            title,
            out_file=out_file,
            caption=title,
        )
        return

    # Automatic generation for all models and splits
    df = df.copy()
    df["true_label"] = df["sex"].map({"f": "F", "m": "M"})

    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")

    split_order = [s for s in ["train", "test"] if s in df["__split__"].unique()]
    for split in split_order:
        for col in pred_cols:
            model = col[len("pred_") : -len("_sex")]
            probs = [f"pred_{model}_proba_f", f"pred_{model}_proba_m"]
            df_sub = df[df["__split__"] == split].copy()
            df_sub = df_sub[df_sub["sex"].isin(["f", "m"])]
            df_sub = df_sub[df_sub[col].isin([0, 1])]
            if df_sub.empty:
                continue
            _plot_quality_heatmaps_single(
                df_sub,
                col,
                probs,
                title=f"{model} — {split}",
            )


def plot_model_quality_heatmaps(df: pd.DataFrame) -> None:
    """Deprecated wrapper for :func:`plot_quality_heatmaps`.

    This function now simply calls :func:`plot_quality_heatmaps` without
    additional parameters so that all available models are plotted.  It will
    be removed in a future version.
    """

    warnings.warn(
        "plot_model_quality_heatmaps is deprecated; use plot_quality_heatmaps",
        DeprecationWarning,
        stacklevel=2,
    )

    plot_quality_heatmaps(df)
