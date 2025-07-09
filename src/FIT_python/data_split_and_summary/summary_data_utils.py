# src/FIT_python/step_02_b_summary_datasets/utils.py
"""Utility functions for dataset split summaries."""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def discover_splits(splits_dir: Path) -> Dict[str, Dict[str, Path]]:
    results: Dict[str, Dict[str, Path]] = {}
    for path in sorted(splits_dir.rglob("*.parquet")):
        if path.parent == splits_dir:
            if "_" in path.stem:
                dataset, origin = path.stem.rsplit("_", 1)
            else:
                # called on a specific dataset directory containing
                # files like ``train.parquet`` or ``test.parquet``
                dataset = splits_dir.name
                origin = path.stem
        else:
            dataset = path.parent.name
            origin = path.stem
        dataset = dataset.strip().replace(" ", "_").lower()
        origin = origin.strip().lower()
        results.setdefault(dataset, {})[origin] = path
    return results

def load_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Split file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise RuntimeError(f"Unsupported file type: {path.suffix}")
    if "sex" not in df.columns:
        raise RuntimeError(f"Missing 'sex' column in {path}")
    return df

def compute_summary(
    df: pd.DataFrame,
    dataset: str,
    origin: str,
) -> pd.DataFrame:
    """Return summary per Split (dataset+origin) and Sex."""
    print(f"\n[DEBUG] compute_summary called for {dataset}/{origin}, df shape = {df.shape}")

    if df.empty:
        print("[DEBUG] df is empty, returning empty summary")
        return pd.DataFrame(columns=[
            "Dataset", "Sex", "NumberOfFootprints",
            "UniqueIndividuals", "UniqueTrails",
            "MeanFootprintsPerIndividual", "SDFootprintsPerIndividual",
            "MeanTrailsPerIndividual", "SDTrailsPerIndividual",
            "Species",
        ])

    # 1) unified label
    ds_label = f"{dataset} {origin.capitalize()}"
    print(f"[DEBUG] ds_label = {ds_label}")

    # 1b) extract species code
    col1 = df.get("species")
    col2 = df.get("Species")
    print(f"[DEBUG] species column raw: species exists? {col1 is not None}, Species exists? {col2 is not None}")

    if col1 is not None:
        species_col = col1
        print("[DEBUG] using 'species' column")
    elif col2 is not None:
        species_col = col2
        print("[DEBUG] using 'Species' column")
    else:
        species_col = None
        print("[DEBUG] no species column found")

    if species_col is not None:
        species_vals = (
            species_col.astype(str)
                         .str.strip()
                         .str.lower()
                         .unique()
        )
        print(f"[DEBUG] species_vals = {species_vals}")
        raw_code = species_vals[0] if len(species_vals) else "unknown"
        species_code = raw_code.replace("_", " ")
    else:
        species_code = "unknown"
    print(f"[DEBUG] species_code = {species_code}")

    # 2) clean sex categories
    df = df.copy()
    df["sex"] = (
        df["sex"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda s: "F" if s.startswith("f")
                     else "M" if s.startswith("m")
                     else "Unknown")
    )
    print(f"[DEBUG] sex value counts:\n{df['sex'].value_counts(dropna=False)}")

    rows = []
    for sex in ["F", "M", "Unknown"]:
        sub = df[df["sex"] == sex]
        print(f"[DEBUG] sex = {sex}, subset shape = {sub.shape}")

        # Footprints count
        n_fp = len(sub)
        # Unique counts
        n_ind = sub["individual_id"].nunique() if "individual_id" in sub.columns else 0
        n_tr  = sub["trail"].nunique()          if "trail" in sub.columns         else 0
        print(f"[DEBUG]  n_fp={n_fp}, n_ind={n_ind}, n_tr={n_tr}")

        # Per-individual stats
        if n_ind > 0:
            fp_per_ind = sub.groupby("individual_id").size()
            mean_fp = fp_per_ind.mean()
            sd_fp   = fp_per_ind.std(ddof=0)

            tr_per_ind = sub.groupby("individual_id")["trail"].nunique()
            mean_tr = tr_per_ind.mean()
            sd_tr   = tr_per_ind.std(ddof=0)
            print(f"[DEBUG]   fp_per_ind summary: mean={mean_fp}, sd={sd_fp}")
            print(f"[DEBUG]   tr_per_ind summary: mean={mean_tr}, sd={sd_tr}")
        else:
            mean_fp = sd_fp = mean_tr = sd_tr = 0.0
            print(f"[DEBUG]   no individuals, setting means and sds to 0")

        rows.append({
            "Dataset": ds_label,
            "Sex": sex,
            "NumberOfFootprints": n_fp,
            "UniqueIndividuals": n_ind,
            "UniqueTrails": n_tr,
            "MeanFootprintsPerIndividual": round(mean_fp, 2),
            "SDFootprintsPerIndividual": round(sd_fp, 2),
            "MeanTrailsPerIndividual": round(mean_tr, 2),
            "SDTrailsPerIndividual": round(sd_tr, 2),
            "Species": species_code,
        })

    result = pd.DataFrame(rows)
    print(f"[DEBUG] Returning summary DataFrame with shape {result.shape}")
    return result



def _lighten(color: str, amount: float) -> str:
    base = mcolors.to_rgb(color)
    r, g, b = [1 - (1 - c) * amount for c in base]
    return mcolors.to_hex((r, g, b))

def plot_summary_table(df_summary: pd.DataFrame, fig_dir: Path) -> None:
    """
    1) "Summary All" plot: one small stacked train/test bar chart per species.
    2) Individual plots: female and male bars stacked (train below, test above).
    Titles use italic scientific names.
    """
    from FIT_python.plot_style import apply_style
    import matplotlib.pyplot as plt
    import math

    apply_style()

    fig_dir.mkdir(parents=True, exist_ok=True)

    sexes = ['F', 'M']
    splits = ['Train', 'Test']
    colors = {
        'Train': {'F': '#800000', 'M': '#000080'},
        'Test':  {'F': '#cc6666', 'M': '#6666cc'},
    }

    SPECIES_REMAP = {
        "a_j_soemmeringii": "acinonyx_jubatus_soemmeringii",
        "a_j_jubatus": "acinonyx_jubatus_jubatus",
    }

    def make_italic(name_code: str) -> str:
        # map non‐standard und ensure spaces
        name_code = SPECIES_REMAP.get(name_code.lower(), name_code)
        name_code = name_code.replace("_", " ")
        parts = name_code.split()
        # capitalise genus only, rest lower‐case
        parts = [parts[0].capitalize()] + [p.lower() for p in parts[1:]]
        sci   = " ".join(parts)
        return rf"$\mathit{{{sci}}}$"

    # --- 1) Summary-All Plot ---
    species_codes = sorted(df_summary["Species"].dropna().unique())
    n = len(species_codes)
    cols = 3
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), sharey=True)
    axes = axes.flatten()

    for ax, code in zip(axes, species_codes):
        sub = df_summary[df_summary["Species"] == code]
        bottoms = {'F':0,'M':0}
        for split in splits:
            for sex in sexes:
                row = sub[(sub['Sex']==sex)&sub['Dataset'].str.endswith(split)]
                cnt = int(row['NumberOfFootprints'].iloc[0]) if not row.empty else 0
                ind_n = int(row['UniqueIndividuals'].iloc[0]) if not row.empty else 0
                trl_n = int(row['UniqueTrails'].iloc[0])      if not row.empty else 0
                ax.bar(sex, cnt, bottom=bottoms[sex],
                       color=colors[split][sex], edgecolor='black', width=0.6)
                if cnt:
                    ax.text(sex, bottoms[sex]+cnt/2,
                            f'n=({ind_n},{trl_n})',
                            ha='center', va='center', fontsize=8)
                bottoms[sex] += cnt
        ax.set_title(make_italic(code), pad=6)
        ax.set_xticks(sexes)
        ax.set_xticklabels(['Female','Male'])
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(fig_dir/f"summary_all.{ext}")
    #plt.show()
   # plt.close(fig)

    # --- 2) per-species plots ---
    for code in species_codes:
        sub = df_summary[df_summary["Species"] == code]
        # total individuals from the training split (not displayed but may be useful)
        total_ind = {sex: int(
            sub[(sub['Sex']==sex)&sub['Dataset'].str.endswith('Train')]['UniqueIndividuals'].iloc[0]
        ) if not sub.empty else 0 for sex in sexes}

        fig, ax = plt.subplots(figsize=(6,5))
        bottoms = {'F':0,'M':0}

        for split in splits:
            for sex in sexes:
                row = sub[(sub['Sex']==sex)&sub['Dataset'].str.endswith(split)]
                cnt = int(row['NumberOfFootprints'].iloc[0]) if not row.empty else 0
                ind_n = int(row['UniqueIndividuals'].iloc[0]) if not row.empty else 0
                trl_n = int(row['UniqueTrails'].iloc[0])      if not row.empty else 0

                ax.bar(sex, cnt, bottom=bottoms[sex],
                       color=colors[split][sex], edgecolor='black', width=0.6)
                if cnt:
                    ax.text(sex, bottoms[sex]+cnt/2,
                            f'n=({ind_n},{trl_n})',
                            ha='center', va='center', fontsize=10)
                bottoms[sex] += cnt

        ax.set_xticks(sexes)
        ax.set_xticklabels(['Female', 'Male'])
        ax.set_ylabel('Number of Footprints')
        # title with scientific name in italics
        ax.set_title(make_italic(code))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        for ext in ('png','svg'):
            fig.savefig(fig_dir/f"{code}_summary.{ext}")
       # plt.show()
       # plt.close(fig)







def plot_split_proportions(df_summary: pd.DataFrame, fig_dir: Path) -> None:
    """Plot fraction of footprints per split for each species."""
    from FIT_python.plot_style import apply_style
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = df_summary.copy()
    df['Split'] = df['Dataset'].str.split().str[-1].str.capitalize()
    if 'Species' not in df.columns:
        df['Species'] = df['Dataset'].str.split().str[0]
    counts = (
        df.groupby(['Species', 'Split'])['NumberOfFootprints']
          .sum()
          .unstack(fill_value=0)
    )
    split_order = ['Train', 'Test', 'Inference']
    for col in split_order:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[split_order]
    proportions = counts.div(counts.sum(axis=1), axis=0)

    species_codes = counts.index.tolist()
    x = np.arange(len(species_codes))
    width = 0.25
    colors = {'Train': '#4e79a7', 'Test': '#f28e2b', 'Inference': '#999999'}

    fig, ax = plt.subplots(figsize=(max(4, len(species_codes)*1.5), 4))
    for i, split in enumerate(split_order):
        ax.bar(x + (i-1)*width, proportions[split], width=width,
               color=colors[split], edgecolor='black', label=split)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in species_codes], rotation=45, ha='right')
    ax.set_ylabel('Fraction of Footprints')
    ax.set_ylim(0, 1)
    ax.legend(title='Split')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(fig_dir / f'split_proportions.{ext}')
