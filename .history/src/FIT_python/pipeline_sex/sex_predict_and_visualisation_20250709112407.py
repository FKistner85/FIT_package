import pandas as pd
from joblib import load
from pathlib import Path
from FIT_python.config import DATA_DIR, RESULTS_DATA_DIR  # aus Eurer config

# 1) Basis-Pfade aus der config
SPLITS_DIR       = DATA_DIR        / "splits" / "eurasian_otter"
RESULTS_MODELS   = RESULTS_DATA_DIR / "eurasian_otter_random_search_standard_metrics"
OUTPUT_CSV       = RESULTS_DATA_DIR.parent / "eurasian_otter_random_search_standard_metrics" / "eurasian_otter_all_predictions.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# 2) Modelle
models = {
    "balanced_acc": RESULTS_MODELS / "best_balanced_test_acc" / "eurasian_otter.joblib",
    "maj_pct":      RESULTS_MODELS / "best_maj_test_pct"      / "eurasian_otter.joblib",
    "best_mean_test_neg_log_loss": RESULTS_MODELS / "best_mean_test_neg_log_loss"      / "eurasian_otter.joblib",
    "best_accuracy_test": RESULTS_MODELS / "best_accuracy_test"      / "eurasian_otter.joblib",

}

# 3) Splits
splits = {
    "train":     SPLITS_DIR / "train.parquet",
    "test":      SPLITS_DIR / "test.parquet",
    "inference": SPLITS_DIR / "inference.parquet",
}

# 4) Daten laden und 'set'-Spalte anfügen
dfs = {}
for split_name, path in splits.items():
    df = pd.read_parquet(path)
    df.columns = [c.replace('.', '_').replace('-', '_').replace('T', 't') for c in df.columns]
    df["__split__"] = split_name
    dfs[split_name] = df

# 5) Modelle laden & Vorhersagen anfügen
for model_name, model_fp in models.items():
    if not model_fp.is_file():
        raise FileNotFoundError(f"Modell nicht gefunden: {model_fp}")
    clf = load(model_fp)
    for split_name, df in dfs.items():
        df[f"pred_{model_name}_sex"]     = clf.predict(df)
        proba = clf.predict_proba(df)
        df[f"pred_{model_name}_proba_f"] = proba[:, 0]
        df[f"pred_{model_name}_proba_m"] = proba[:, 1]

# 6) Alles zusammenführen & speichern
all_df = pd.concat(dfs.values(), ignore_index=True)
all_df.to_csv(OUTPUT_CSV, index=False)
print(f"Alle Vorhersagen gespeichert in: {OUTPUT_CSV}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from FIT_python.config import RESULTS_DATA_DIR
from pathlib import Path

# === 1) CSV laden ===
file_path = RESULTS_DATA_DIR / "eurasian_otter_random_search_standard_metrics" / "eurasian_otter_all_predictions.csv"
df = pd.read_csv(file_path)

# === 2) Spaltennamen ausgeben zum Debug (optional) ===
print("Spalten im DataFrame:", df.columns.tolist())

# === 3) Filter & Mapping ===
df = df[df.filter(like="pred_").any(axis=1)]  # Zeilen mit wenigstens einer Vorhersage
df["true_label"] = df["sex"].str.upper().map({"F":"F","M":"M"})

df = pd.read_csv(file_path)

# Labels standardisieren
df["true_label"] = df["sex"].str.upper().map({"F":"F","M":"M"})

# Confusion-Matrix-Funktion
def get_conf_matrix(y_true, y_pred, labels=["F","M"]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float)/row_sums
        cm_norm[np.isnan(cm_norm)] = 0
    annot = np.empty_like(cm).astype(object)
    for i in range(len(labels)):
        for j in range(len(labels)):
            cnt = cm[i,j]
            pct = int(round(cm_norm[i,j]*100))
            annot[i,j] = f"{cnt}\n({pct}%)"
    acc = accuracy_score(y_true, y_pred)
    return cm_norm, annot, acc

# Modelle dynamisch ermitteln
pred_sex_cols = [c for c in df.columns if c.startswith("pred_") and c.endswith("_sex")]
model_names = [c[len("pred_"):-len("_sex")] for c in pred_sex_cols]

# Splits in der Datei
splits = df["__split__"].unique()

for model_name in model_names:
    pred_col = f"pred_{model_name}_sex"
    for split_name in splits:
        df_sub = df[(df["__split__"]==split_name) & df[pred_col].isin([0,1])].copy()
        if df_sub.empty:
            continue

        df_sub["pred_label"] = df_sub[pred_col].map({0:"F",1:"M"})
        df_sub["true_label"] = df_sub["true_label"]

        if split_name != "inference":
            # Confusion Matrix für train/test
            cm, annot, acc = get_conf_matrix(df_sub["true_label"], df_sub["pred_label"])
            plt.figure(figsize=(5,5))
            im = plt.imshow(cm, vmin=0, vmax=1, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            ticks = ["F","M"]
            plt.xticks(range(2), ticks)
            plt.yticks(range(2), ticks)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, annot[i,j], ha="center", va="center")
            plt.tight_layout()
            plt.show()
            print(f"Dataset: {split_name}, Model: {model_name}  (Accuracy: {acc:.2%})")

        else:
            # Barplot für Inference
            pivot = df_sub.pivot_table(
                index="trail",
                columns="pred_label",
                values=pred_col,
                aggfunc="count",
                fill_value=0
            )
            ax = pivot.plot(
                kind="bar",
                stacked=True,
                figsize=(8,5),
                color={"F": "#CA201A", "M": "#2355E0"}
            )
            ax.set_xlabel("Trail")
            ax.set_ylabel("Numpber of Prints in Trail")
            plt.tight_layout()
            plt.show()
            print(f"Dataset: {split_name}, Model: {model_name}")


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from FIT_python.config import DATA_DIR, RESULTS_DATA_DIR

# === Basis-Pfade aus config ===
MODELS_DIR = RESULTS_DATA_DIR / "eurasian_otter_random_search_standard_metrics"
CSV_PATH   = MODELS_DIR / "eurasian_otter_all_predictions.csv"

# === CSV laden ===
df = pd.read_csv(CSV_PATH)
# Korrektes Mapping ohne NaNs
df["true_label"] = df["sex"].map({"f": "F", "m": "M"})
print(f"Loaded DataFrame with {len(df)} rows; true_label unique: {df['true_label'].unique()}")

# === Helper für Heatmaps ===
def plot_quality_heatmaps(df_sub, pred_col, proba_cols):
    df = df_sub.copy()
    df["pred_label"] = df[pred_col].map({0: "F", 1: "M"})
    df["Correct"]    = df["pred_label"] == df["true_label"]
    df["Max_Prob"]   = df[proba_cols].max(axis=1)
    df["Quality"]    = df["Max_Prob"].apply(lambda p: "High" if p>0.9 else ("Moderate" if p>0.7 else "Low"))

    # MultiIndex aller Kombis
    idx = pd.MultiIndex.from_product(
        [["F","M"], [True, False]],
        names=["true_label", "Correct"]
    )
    counts = (
        df
        .groupby(["true_label", "Correct", "Quality"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=idx, columns=["High","Moderate","Low"], fill_value=0)
    )
    if counts.values.sum() == 0:
        print("    → No data to plot")
        return

    normed = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    def make_annot(block):
        total = block.values.sum()
        return block.applymap(
            lambda x: f"{int(x)}\n({int(round(x/total*100))}%)" if total>0 else "0\n(0%)"
        )
    annot_corr   = make_annot(counts.xs(True, level="Correct"))
    annot_incorr = make_annot(counts.xs(False, level="Correct"))

    green_cmap = LinearSegmentedColormap.from_list("green", ["white","mediumseagreen"])
    red_cmap   = LinearSegmentedColormap.from_list("red",   ["white","crimson"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    sns.heatmap(
        normed.xs(True, level="Correct"),
        annot=annot_corr, fmt="",
        cmap=green_cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="gray",
        ax=axes[0], cbar=True
    )
    axes[0].set(title="Correct Predictions", ylabel="True Label")
    sns.heatmap(
        normed.xs(False, level="Correct"),
        annot=annot_incorr, fmt="",
        cmap=red_cmap, vmin=0, vmax=1,
        linewidths=0.5, linecolor="gray",
        ax=axes[1], cbar=True
    )
    axes[1].set(title="Incorrect Predictions")
    for ax in axes:
        ax.set_xlabel("Prediction Quality")
        ax.set_xticklabels(["High","Moderate","Low"], rotation=0)
        ax.set_yticklabels(["F","M"], rotation=0)
    plt.tight_layout()
    plt.show()

# === Für jeden Split und jedes Modell plotten ===
pred_sex_cols = [c for c in df.columns if c.startswith("pred_") and c.endswith("_sex")]
for split in ["train", "test"]:
    for pred_col in pred_sex_cols:
        model = pred_col[len("pred_"):-len("_sex")]
        proba_cols = [f"pred_{model}_proba_f", f"pred_{model}_proba_m"]
        df_sub = df[df["__split__"] == split].copy()
        df_sub = df_sub[df_sub[pred_col].isin([0, 1])]
        print(f"Dataset: {split}, Model: {model}, Rows: {len(df_sub)}")
        if df_sub.empty:
            continue
        plot_quality_heatmaps(df_sub, pred_col, proba_cols)

# Aktiviere Inline-Plots im Notebook

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from FIT_python.config import RESULTS_DATA_DIR

# === Basis-Pfade ===
MODELS_DIR = RESULTS_DATA_DIR / "eurasian_otter_random_search_standard_metrics"
CSV_PATH   = MODELS_DIR / "eurasian_otter_all_predictions.csv"

# === CSV laden ===
df = pd.read_csv(CSV_PATH)
df["true_label"] = df["sex"].map({"f":"F","m":"M"})

# Klassifizierungsqualität
def classify_majority_accuracy(group):
    acc = group["Correct"].mean()
    if   acc >= 0.9: return "High"
    elif acc >= 0.7: return "Moderate"
    elif acc >= 0.5: return "Low"
    else:            return "Misclassified"

# Modelle ermitteln
pred_sex_cols = [c for c in df.columns if c.startswith("pred_") and c.endswith("_sex")]
model_names   = [c[len("pred_"):-len("_sex")] for c in pred_sex_cols]

for model in model_names:
    print(f"\n==== Model: {model} ====")
    pred_col   = f"pred_{model}_sex"
    proba_cols = [f"pred_{model}_proba_f", f"pred_{model}_proba_m"]

    # pred_label & Correct
    df_mod = df.copy()
    df_mod["pred_label"] = df_mod[pred_col].map({0:"F",1:"M"})
    df_mod["Correct"]    = df_mod["pred_label"] == df_mod["true_label"]

    # Erstelle Pivot für Trail- und Animal-Level
    pivots = []
    for level, group_cols in zip(
        ["trail", "individual_id"],
        [["trail","true_label"], ["individual_id","true_label"]]
    ):
        quality = (
            df_mod
            .groupby(group_cols, group_keys=False)
            .apply(classify_majority_accuracy)
            .reset_index(name="Class")
        )
        piv = (
            quality
            .pivot_table(index="true_label", columns="Class", values=level,
                         aggfunc="count", fill_value=0)
            .reindex(columns=["High","Moderate","Low","Misclassified"], fill_value=0)
        )
        pivots.append(piv)

    # Plot
    cmap = LinearSegmentedColormap.from_list("green", ["white","mediumseagreen"])
    fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)
    class_labels = ["High","Moderate","Low","Misclassified"]
    y_labels     = ["F","M"]

    for ax, piv, tag, label_info in zip(
        axes, pivots, ["a)", "b)"],
        ["(trail-level)", "(animal-level)"]
    ):
        # Prozentwerte
        totals      = piv.sum(axis=1)
        props       = piv.div(totals, axis=0).fillna(0)
        counts      = piv.astype(int)
        total_sum   = counts.values.sum()

        # Annotation per Zelle
        annot = pd.DataFrame("", index=counts.index, columns=counts.columns)
        for i in counts.index:
            for j in counts.columns:
                v = counts.at[i, j]
                if v:
                    annot.at[i, j] = f"{v}\n({v/total_sum:.0%})"

        sns.heatmap(
            props, annot=annot, fmt="",
            cmap=cmap, vmin=0, vmax=1,
            linewidths=0.5, linecolor="gray",
            ax=ax, cbar=True
        )
        # Nur "a)" bzw "b)" als Titel
        ax.set_title(tag, loc="left", fontsize=14, fontweight="bold")
        # Info als Kommentar an die Achse
        ax.text(1.02, 0.5, label_info, transform=ax.transAxes,
                rotation=90, va="center", fontsize=10, color="gray")
        ax.set_ylabel("True Sex")
        ax.set_yticks(np.arange(len(y_labels)) + 0.5)
        ax.set_yticklabels(y_labels, rotation=0)
        ax.set_xticks(np.arange(len(class_labels)) + 0.5)
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_xlabel("Classification Quality")

        # Drucke die reinen Counts drunter
        print(f"{tag} counts {label_info}:\n{counts}\n")

    plt.tight_layout()
    plt.show()