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
from FIT_python.Visualisations.plot_style import SEX_COLORS, SEX_VALUE_MAP
from FIT_python.Visualisations.plot_style import apply_style, map_sex
from sklearn.base import clone


DEFAULT_SPECIES = "eurasian_otter"

# Map raw sex labels to integers for model training and back
SEX_TO_INT = {"f": 0, "m": 1}
INT_TO_SEX = {v: k for k, v in SEX_TO_INT.items()}


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
        # Heuristically resolve the directory containing the saved models
        if not any((models / d).exists() for d in MODELS.values()):
            # Common experiment layout puts the models under
            # ``results/data/random_search_standard_metrics`` relative to the
            # experiment root.  When ``models_dir`` points to the experiment
            # directory or ``models`` subfolder, adjust accordingly.
            candidate = models / "results" / "data" / "random_search_standard_metrics"
            if not candidate.exists() and models.name == "models":
                candidate = models.parent / "results" / "data" / "random_search_standard_metrics"
            if candidate.exists():
                models = candidate
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
    "mean_rank": "best_mean_rank",
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
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.base import clone
from joblib import load

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import joblib
import warnings

def predict_all(
    species: str,
    metric_key: str = "best_balanced_test_acc",  # z.B. "best_accuracy_test"
    prefer_generic: bool = False,
    models_dir: str | Path | None = None,
    include_inference: bool = True,
    reuse_csv: bool = True,
    use_cv_train_predictions: bool = True,
) -> pd.DataFrame:
    """Return dataframe with best-model predictions for Train/Test/Inference.

    Train: Out-of-fold predictions (OOF) using 'Fold' column.
    Test + Inference: Predictions from model retrained on full Train.
    """

    splits_dir, models_dir, csv_path = _base_paths(
        species, prefer_generic, models_dir
    )

    # When reuse_csv=True the predictions must already exist on disk
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

    # Splits laden
    split_names = ["train", "test"]
    if include_inference and (splits_dir / "inference.parquet").exists():
        split_names.append("inference")
    splits = {n: splits_dir / f"{n}.parquet" for n in split_names}
    dfs = {name: pd.read_parquet(p) for name, p in splits.items()}

    for name, df in dfs.items():
        df.columns = df.columns.str.replace(r"[.\-]", "_", regex=True).str.replace("T", "t")
        df["__split__"] = name

    # Bestes Modell laden
    best_model_path = models_dir / metric_key / f"{species}.joblib"
    if not best_model_path.exists():
        raise FileNotFoundError(f"No model found at {best_model_path}")
    best_model = joblib.load(best_model_path)

    # Features bestimmen
    num_cols = dfs["train"].select_dtypes(include=np.number).columns
    feature_cols = [c for c in num_cols if not c.startswith("pred_") and c != "Fold"]

    # OOF Predictions für Train
    if use_cv_train_predictions and "Fold" in dfs["train"].columns:
        n_samples = len(dfs["train"])
        oof_preds = np.empty(n_samples, dtype=object)  # Strings zulassen
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

    # Retrain auf ganzem Trainset für Test und Inference
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

    # Zusammenführen und Speichern
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


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion(df: pd.DataFrame) -> None:
    """Plot confusion matrices (train + test) mit a)/b) Titeln und Count + % in den Zellen."""

    apply_style()

    if "pred_sex" not in df.columns:
        raise KeyError("DataFrame contains no 'pred_sex' column")
    if "__split__" not in df.columns:
        raise KeyError("DataFrame is missing the '__split__' column")

    mapping = {
        "f": "F", "m": "M", "female": "F", "male": "M",
        "F": "F", "M": "M", 0: "F", 1: "M",
    }

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
        return

    y_true_train, y_pred_train = train["true_label"], train["pred_label"]
    y_true_test, y_pred_test   = test["true_label"],  test["pred_label"]

    cm_train = confusion_matrix(y_true_train, y_pred_train, labels=["F", "M"])
    cm_test  = confusion_matrix(y_true_test,  y_pred_test,  labels=["F", "M"])

    # Prozentwerte berechnen
    cm_train_percent = cm_train / cm_train.sum(axis=1, keepdims=True) * 100
    cm_test_percent  = cm_test  / cm_test.sum(axis=1, keepdims=True) * 100

    # Labels "Count (XX%)"
    annot_train = np.array([
        [f"{cm_train[i, j]} ({cm_train_percent[i, j]:.0f}%)" for j in range(2)] for i in range(2)
    ])
    annot_test = np.array([
        [f"{cm_test[i, j]} ({cm_test_percent[i, j]:.0f}%)" for j in range(2)] for i in range(2)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    mats = [
        (cm_train_percent, annot_train, "a)"),
        (cm_test_percent,  annot_test,  "b)"),
    ]

    for ax, (cm, annot, title) in zip(axes, mats):
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=["Female", "Male"],
            yticklabels=["Female", "Male"],
            ax=ax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
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
    """Plot predicted sex counts for inference split with Location + Trail number labels."""
    apply_style()
    pred_cols = [c for c in df if c.startswith("pred_") and c.endswith("_sex")]
    if not pred_cols:
        raise KeyError("DataFrame contains no prediction columns")
    if "inference" not in df["__split__"].unique():
        return

    sub = df[df["__split__"] == "inference"].copy()
    for col in pred_cols:
        sub[col] = sub[col].map(SEX_VALUE_MAP)

        # Nur letzte Zahl aus Trail extrahieren
        sub["trail_num"] = sub["trail"].str.extract(r"(\d+)$")
        sub["loc_trail"] = sub["location"] + " " + sub["trail_num"].fillna("")

        pivot = sub.pivot_table(
            index="loc_trail", columns=col, aggfunc="size", fill_value=0
        )

        colors = [SEX_COLORS.get(c, "#333333") for c in pivot.columns]
        ax = pivot.plot.bar(stacked=True, figsize=(10, 4), color=colors)

        plt.xlabel("Location & Trail")
        plt.ylabel("Count")
        ax.legend(title="Predicted", labels=pivot.columns)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()



def plot_confusion_and_inference(df: pd.DataFrame) -> None:
    """Backward compatible wrapper calling :func:`plot_confusion` and :func:`plot_inference`."""
    plot_confusion(df)
    plot_inference(df)


# === Plots für Qualitäts-Heatmaps ===
def plot_quality_grouped(df: pd.DataFrame) -> None:
    """Plot quality heatmaps based on majority decision per Trail or Individual."""

    apply_style()

    splits = ["train", "test"] if "__split__" in df.columns else [None]
    for split in splits:
        df_sub = df if split is None else df[df["__split__"] == split]
        if df_sub.empty:
            continue

        # nur f/m berücksichtigen
        df_plot = df_sub[df_sub["sex"].isin(["f", "m"])].copy()
        df_plot["true_label"] = map_sex(df_plot["sex"])

        # alle Prediction-Spalten holen
        pred_cols = [c for c in df_plot if c.startswith("pred_") and c.endswith("_sex")]
        if not pred_cols:
            raise KeyError("No prediction label columns found in dataframe")

        # Vorhersagen pro Print -> Label
        for col in pred_cols:
            key = col.split("_")[1]
            df_plot[f"pred_label_{key}"] = df_plot[col].map({0: "Female", 1: "Male"})

        # pro Gruppe Mehrheitsentscheidung
        def classify_majority(group):
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
        fig, axes = plt.subplots(1, 2,
                                 figsize=(plt.rcParams["figure.figsize"][0], height),
                                 sharey=True)

        for ax, (tag, cols) in zip(
            axes,
            [
                ("a)", ["trail", "true_label"]),
                ("b)", ["individual_id", "true_label"]),
            ],
        ):
            acc = df_plot.groupby(cols).apply(classify_majority).reset_index(name="Class")

            pivot = acc.pivot_table(
                index="true_label",
                columns="Class",
                values=cols[0],
                aggfunc="count",
                fill_value=0,
            ).reindex(columns=["High", "Moderate", "Low", "Misclassified"], fill_value=0)

            proportions = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
            counts = pivot.astype(int)
            total = counts.values.sum()

            # Annotation: counts und prozentualer Anteil
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
            ax.set_xlabel("Quality")
            if ax is axes[0]:
                ax.set_ylabel("Sex")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

        if split is not None:
            fig.suptitle(f"{split} set", y=1.02)

        plt.tight_layout()
        plt.show()





def plot_individual_probabilities(df: pd.DataFrame, out_dir: str | Path) -> None:
    """Plot distribution of predicted sex probabilities for train and test."""

    apply_style()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ``predict_all`` now produces a single probability column ``pred_proba_m``.
    # For backwards compatibility we still detect any columns ending with
    # ``_proba_m``.
    proba_cols = [c for c in df.columns if c.endswith("_proba_m")]
    if not proba_cols:
        raise ValueError(
            "DataFrame contains no probability columns ending with '_proba_m'.",
        )

    palette = {
        "Female": SEX_COLORS["Female"],
        "Male": SEX_COLORS["Male"],
        "Unknown": SEX_COLORS.get("Unknown", "#333333"),
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
            # Remove prefix/suffix to obtain a readable model identifier.  When
            # no identifier is present (e.g. ``pred_proba_m``) fall back to a
            # generic name to avoid creating file names with double underscores.
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
            from FIT_python.caption_utils import save_caption

            plt.tight_layout()
            suffix = f"_{split}" if split is not None else ""
            file = out_path / f"{model}_individual_probabilities{suffix}.png"
            plt.savefig(file)
            save_caption(
                file,
                f"Predicted male probability for {model}{suffix.replace('_', ' ')}".strip(),
            )
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
        splits = ["train", "test"] if "__split__" in df.columns else [None]
        for split in splits:
            df_sub = df if split is None else df[df["__split__"] == split]
            if df_sub.empty:
                continue
            df_sub = df_sub.copy()
            df_sub["true_label"] = map_sex(df_sub["sex"])
            sub_title = title if title is not None else (f"{split} set" if split else None)
            _plot_quality_heatmaps_single(
                df_sub, pred_col, proba_cols, sub_title, group_by
            )
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
            model_prefix = f"{model}_" if model else ""
            probs = [
                f"pred_{model_prefix}proba_f",
                f"pred_{model_prefix}proba_m",
            ]
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
