import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import warnings
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from FIT_python.config import DATA_DIR, RESULTS_DATA_DIR, PATHS
from FIT_python.Visualisations.plot_style import SEX_COLORS
from FIT_python.Visualisations.plot_style import apply_style, map_sex


DEFAULT_SPECIES = "eurasian_otter"


def _base_paths(
    species: str = DEFAULT_SPECIES,
    prefer_generic: bool = False,
    models_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Return split dir, model dir and output CSV for ``species``.

    If ``models_dir`` is provided, it is used directly.  Otherwise the function
    falls back to the species-specific directory under
    ``results/data``. When ``prefer_generic`` is ``True`` or the species
    directory does not exist, ``random_search_standard_metrics`` is used
    instead.  This enables notebooks that work on a single species to load
    custom models while the all-species notebook can still rely on the generic
    directory.
    """

    splits = DATA_DIR / "splits" / species

    if models_dir is not None:
        models = Path(models_dir)
    else:
        specific = PATHS["random_search"].with_name(
            f"{species}_random_search_standard_metrics"
        )
        generic = PATHS["random_search"]
        models = generic if prefer_generic else specific
        if not models.exists():
            models = specific if prefer_generic else generic

    csv = Path(models) / f"{species}_all_predictions.csv"
    return splits, Path(models), csv


# === Modelle definieren ===
MODELS = {
    "balanced_acc": "best_balanced_test_acc",
    "maj_pct": "best_maj_test_pct",
    "neg_log_loss": "best_mean_test_neg_log_loss",
    "accuracy": "best_accuracy_test",
}

def predict_simple_baseline(
    species: str,
    exp_dir: Path,
    *,
    include_inference: bool = True,
    reuse_csv: bool = True,
) -> pd.DataFrame:
    """Return predictions of the simple baseline model for ``species``.

    Parameters
    ----------
    species:
        Species folder under ``data/splits``.
    exp_dir:
        Directory containing the trained models produced by
        :func:`run_simple_baseline_all_species`.
    include_inference:
        Include the ``inference`` split when it exists.
    reuse_csv:
        When ``True`` the function expects ``{species}_baseline_predictions.csv``
        to exist in ``exp_dir``. If the file is missing a ``FileNotFoundError``
        is raised. Set to ``False`` to recompute the predictions.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the original splits and three additional columns:
        ``pred_baseline_sex``, ``pred_baseline_proba_f`` and
        ``pred_baseline_proba_m``.
    """

    exp_dir = Path(exp_dir)
    splits_dir = DATA_DIR / "splits" / species
    csv_path = exp_dir / f"{species}_baseline_predictions.csv"

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

    model_path = exp_dir / "models" / f"{species}.joblib"
    clf = load(model_path)

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
        if X.empty:
            df["pred_baseline_sex"] = []
            df["pred_baseline_proba_f"] = []
            df["pred_baseline_proba_m"] = []
        else:
            df["pred_baseline_sex"] = clf.predict(X)
            proba = clf.predict_proba(X)
            df["pred_baseline_proba_f"] = proba[:, 0]
            df["pred_baseline_proba_m"] = proba[:, 1]
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    if not reuse_csv:
        all_df.to_csv(csv_path, index=False)

    return all_df

# === CSV erzeugen (einmal laufen lassen) ===
def predict_all(
    species: str = DEFAULT_SPECIES,
    prefer_generic: bool = False,
    models_dir: str | Path | None = None,
    include_inference: bool = True,
    reuse_csv: bool = True,
    use_cv_train_predictions: bool = False,
) -> pd.DataFrame:
    """Return dataframe with model predictions for ``species``.

    Parameters
    ----------
    species:
        The species folder under ``data/splits``.
    prefer_generic:
        If ``True``, models are loaded from the shared
        ``random_search_standard_metrics`` directory when present. Ignored when
        ``models_dir`` is given.
    models_dir:
        Custom directory containing the trained models. The predictions CSV is
        also written to this directory. When ``None`` (default) the directory is
        determined automatically based on ``prefer_generic`` and the existence of
        a species-specific directory.
    include_inference:
        Include the ``inference`` split if the corresponding parquet exists.
    reuse_csv:
        When ``True`` the existing prediction CSV must be present, otherwise a
        ``FileNotFoundError`` is raised. Set to ``False`` to recompute the
        predictions if the file is missing.
    use_cv_train_predictions:
        If ``True`` use out-of-fold predictions already stored in
        ``train.parquet`` instead of computing new predictions for the
        training split.
    """
    splits_dir, models_dir, csv_path = _base_paths(
        species, prefer_generic, models_dir
    )

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
        for name, df in dfs.items():
            if name == "train" and use_cv_train_predictions:
                continue
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
    """Plot a heatmap visualising mean CV accuracy across preprocessing options.

    Parameters
    ----------
    df:
        DataFrame with the preprocessing options and CV scores.  Columns using
        the names produced by :class:`skopt.BayesSearchCV` (e.g.
        ``select__method`` and ``mean_test_score``) are mapped automatically.
    out_dir:
        Directory where the plot will be saved.
    """
    from FIT_python.Visualisations.plot_style import apply_style

    if df.empty:
        raise ValueError("DataFrame for heatmap is empty")

    # Accept column names from BayesSearchCV output for convenience
    col_map = {
        "select__method": "fs_method",
        "reduce_pre__method": "reduce_pre_method",
        "mean_test_score": "cv_balanced_accuracy",
        "mean_test_balanced_accuracy": "cv_balanced_accuracy",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Missing pipeline step columns default to "None" so the heatmap can still
    # be generated even when a search varied only a subset of options.
    if "fs_method" not in df.columns:
        df["fs_method"] = "None"
    if "reduce_pre_method" not in df.columns:
        df["reduce_pre_method"] = "None"
    if "cv_balanced_accuracy" not in df.columns:
        raise KeyError("Missing column 'cv_balanced_accuracy' for heatmap")

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
    """Predict sex for all species and combine into a single CSV.

    Parameters
    ----------
    species_list:
        Optional list of species names. When ``None`` the function iterates over
        the sub-directories of ``data/splits``.
    """

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

    dfs = [predict_all(sp, prefer_generic=True) for sp in valid_species]
    if not dfs:
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)
    out_csv = RESULTS_DATA_DIR / "all_species_all_predictions.csv"
    all_df.to_csv(out_csv, index=False)
    return all_df


# === Plots für Confusion & Inference ===
def plot_confusion(df: pd.DataFrame) -> None:
    """Plot CV vs. test confusion matrices for each model."""
    apply_style()
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")

    models = {
        c[len("pred_") : -len("_sex")] for c in pred_cols if not c.endswith("_cv_sex")
    }
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
            height = plt.rcParams["figure.figsize"][1] * 0.6
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(8, height),
                sharey=True,
            )
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
            ax.set_xlabel("Predicted Sex")
            if ax is axes[0]:
                ax.set_ylabel("True Sex")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xticklabels(["Female", "Male"], rotation=0)
            ax.set_yticklabels(["Female", "Male"], rotation=0)

        plt.tight_layout()
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
    df["true_label"] = map_sex(df["sex"])
    # pred_label & Correct pro Modell
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    for col in pred_cols:
        key = col.split("_")[1]
        df[f"pred_label_{key}"] = df[col].map({0: "Female", 1: "Male"})

    # True wenn irgendein Modell richtig war

    pred_label_cols = [c for c in df if c.startswith("pred_label_")]
    if not pred_label_cols:
        raise KeyError("No prediction label columns found in dataframe")

    df["Correct"] = np.any([df[c] == df["true_label"] for c in pred_label_cols], axis=0)

    # Klassifizierung
    def label(acc: float) -> str:
        if acc >= 0.9:
            return "High"
        if acc >= 0.7:
            return "Moderate"
        if acc >= 0.5:
            return "Low"
        return "Misclassified"

    # Erstelle Heatmaps a) und b) nebeneinander
    cmap = LinearSegmentedColormap.from_list("green", ["white", "mediumseagreen"])
    height = plt.rcParams["figure.figsize"][1] * 0.75
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plt.rcParams["figure.figsize"][0], height),
        sharey=True,
    )
    for ax, (tag, cols) in zip(
        axes,
        [
            ("a)", ["trail", "true_label"]),
            ("b)", ["individual_id", "true_label"]),
        ],
    ):
        acc = df.groupby(cols)["Correct"].mean().reset_index(name="acc")
        acc["Class"] = acc["acc"].map(label)
        kvals = acc.drop(columns="acc")
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
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(["Female", "Male"], rotation=0)
        ax.set_title(tag, loc="left", fontweight="bold")
        ax.set_xlabel("Quality")
        if ax is axes[0]:
            ax.set_ylabel("Sex")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    # previously annotated as "(trail)" or "(animal)" on the right side of
    # the plot. These labels caused visual artefacts in the heatmaps and have
    # been removed.
    # Legends were intentionally removed to keep the focus on the heatmaps.
    plt.tight_layout()
    plt.show()


def plot_individual_probabilities(df: pd.DataFrame, out_dir: str | Path):
    """Plot distribution of predicted sex probabilities for each individual."""
    apply_style()
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
    grouped["sex_std"] = map_sex(grouped["sex"])
    palette = {
        "Female": SEX_COLORS["Female"],
        "Male": SEX_COLORS["Male"],
        "Unknown": SEX_COLORS.get("Unknown", "#333333"),
    }
    for col in proba_cols:
        model = col.split("_")[1]
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
    group_by: str | None = None,
) -> None:
    """Plot prediction quality heatmaps for a single model and split.

    When ``group_by`` is provided, predictions are aggregated by this column
    (e.g. ``"trail"`` or ``"individual_id"``) before computing the quality
    categories.  This ensures each group is counted only once using a majority
    vote across its predictions.
    """

    df = df_sub.copy()
    df["pred_label"] = df[pred_col].map({0: "Female", 1: "Male"})
    df["Correct"] = df["pred_label"] == df["true_label"]
    df["Max_Prob"] = df[proba_cols].max(axis=1)
    df["Quality"] = df["Max_Prob"].apply(
        lambda p: "High" if p > 0.9 else ("Moderate" if p > 0.7 else "Low")
    )

    if group_by is not None:
        grouped = df.groupby([group_by, "true_label"])
        df = grouped.agg(
            correct_rate=("Correct", "mean"),
        ).reset_index()
        df["Correct"] = df["correct_rate"] > 0.5
        df["Quality"] = df["correct_rate"].apply(
            lambda p: (
                "High"
                if p > 0.9
                else ("Moderate" if p > 0.7 else ("Low" if p > 0.5 else "Misclassified"))
            )
        )

    idx = pd.MultiIndex.from_product(
        [["Female", "Male"], [True, False]], names=["true_label", "Correct"]
    )
    columns = ["High", "Moderate", "Low"]
    if group_by is not None:
        columns.append("Misclassified")
    counts = (
        df.groupby(["true_label", "Correct", "Quality"]).size().unstack(fill_value=0)
    ).reindex(index=idx, columns=columns, fill_value=0)

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

    height = plt.rcParams["figure.figsize"][1] * 0.75
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plt.rcParams["figure.figsize"][0], height),
        sharey=True,
    )
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
        ax.set_xlabel("Prediction Quality")
        labels = ["High", "Moderate", "Low"]
        if group_by is not None:
            labels.append("Misclassified")
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(["Female", "Male"], rotation=0)
    # no super title so subfigures can be labelled externally
    plt.tight_layout()
    plt.show()


def plot_quality_heatmaps(
    df: pd.DataFrame,
    pred_col: str | None = None,
    proba_cols: list[str] | None = None,
    title: str | None = None,
    group_by: str | None = None,
) -> None:
    """Plot prediction-quality heatmaps.

    Parameters ``pred_col`` and ``proba_cols`` can be used to display the
    heatmap for a specific model and split.  When they are omitted, the
    function behaves like the other plotting helpers and iterates over all
    available models and the ``train``/``test`` splits automatically.
    If ``group_by`` is provided, predictions are aggregated per group before
    counting cells in the heatmap.
    """

    apply_style()

    if pred_col is not None and proba_cols is not None:
        df_sub = df.copy()
        df_sub["true_label"] = map_sex(df_sub["sex"])
        _plot_quality_heatmaps_single(df_sub, pred_col, proba_cols, title, group_by)
        return

    # Automatic generation for all models and splits
    df = df.copy()
    df["true_label"] = map_sex(df["sex"])

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
                group_by=group_by,
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
