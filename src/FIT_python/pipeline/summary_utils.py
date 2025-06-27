from __future__ import annotations

"""Utilities for generating dataset split summaries."""

from pathlib import Path
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def discover_splits(splits_dir: Path) -> Dict[str, Dict[str, Path]]:
    results: Dict[str, Dict[str, Path]] = {}
    for path in sorted(splits_dir.rglob("*.parquet")):
        if path.parent == splits_dir:
            if "_" not in path.stem:
                continue
            dataset, origin = path.stem.rsplit("_", 1)
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


def compute_summary(df: pd.DataFrame, dataset: str, origin: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Dataset",
                "Sex",
                "NumberOfFootprints",
                "UniqueIndividuals",
                "UniqueTrails",
                "MeanFootprintsPerIndividual",
                "SDFootprintsPerIndividual",
                "MeanTrailsPerIndividual",
                "SDTrailsPerIndividual",
            ]
        )

    ds_label = dataset
    df = df.copy()
    df["sex"] = (
        df["sex"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda s: "F" if s.startswith("f") else "M" if s.startswith("m") else "Unknown")
    )

    rows = []
    for sex in ["F", "M", "Unknown"]:
        sub = df[df["sex"] == sex]
        n_fp = len(sub)
        n_ind = sub["individual_id"].nunique() if "individual_id" in sub.columns else 0
        n_tr = sub["trail"].nunique() if "trail" in sub.columns else 0

        if n_ind > 0:
            fp_per_ind = sub.groupby("individual_id").size()
            mean_fp = fp_per_ind.mean()
            sd_fp = fp_per_ind.std(ddof=0)

            tr_per_ind = sub.groupby("individual_id")["trail"].nunique()
            mean_tr = tr_per_ind.mean()
            sd_tr = tr_per_ind.std(ddof=0)
        else:
            mean_fp = sd_fp = mean_tr = sd_tr = 0.0

        rows.append(
            {
                "Dataset": ds_label,
                "Origin": origin.capitalize(),
                "Sex": sex,
                "NumberOfFootprints": n_fp,
                "UniqueIndividuals": n_ind,
                "UniqueTrails": n_tr,
                "MeanFootprintsPerIndividual": round(mean_fp, 2),
                "SDFootprintsPerIndividual": round(sd_fp, 2),
                "MeanTrailsPerIndividual": round(mean_tr, 2),
                "SDTrailsPerIndividual": round(sd_tr, 2),
            }
        )

    return pd.DataFrame(rows)


def _lighten(color: str, amount: float) -> str:
    base = mcolors.to_rgb(color)
    r, g, b = [1 - (1 - c) * amount for c in base]
    return mcolors.to_hex((r, g, b))


def plot_summary_table(df_summary: pd.DataFrame, fig_dir: Path) -> None:
    import math

    fig_dir.mkdir(parents=True, exist_ok=True)

    sexes = ["F", "M"]
    splits = ["Train", "Test"]
    colors = {
        "Train": {"F": "#800000", "M": "#000080"},
        "Test": {"F": "#cc6666", "M": "#6666cc"},
    }

    def make_italic(name_code: str) -> str:
        parts = name_code.split("_")
        parts = [parts[0].capitalize()] + [p.lower() for p in parts[1:]]
        sci = " ".join(parts)
        return rf"$\mathit{{{sci}}}$"

    species_codes = sorted({label.rsplit(" ", 1)[0] for label in df_summary["Dataset"].unique()})
    n = len(species_codes)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), sharey=True)
    axes = axes.flatten()

    for ax, code in zip(axes, species_codes):
        sub = df_summary[df_summary["Dataset"].str.startswith(code + " ")]
        bottoms = {"F": 0, "M": 0}
        for split in splits:
            for sex in sexes:
                row = sub[(sub["Sex"] == sex) & sub["Dataset"].str.endswith(split)]
                cnt = int(row["NumberOfFootprints"].iloc[0]) if not row.empty else 0
                ax.bar(sex, cnt, bottom=bottoms[sex], color=colors[split][sex], edgecolor="black", width=0.6)
                bottoms[sex] += cnt
        ax.set_title(make_italic(code), pad=6)
        ax.set_xticks(sexes)
        ax.set_xticklabels(["Female", "Male"])
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(fig_dir / f"summary_all.{ext}")
    plt.show()
    plt.close(fig)

    for code in species_codes:
        sub = df_summary[df_summary["Dataset"].str.startswith(code + " ")]
        total_ind = {sex: int(sub[(sub["Sex"] == sex) & sub["Dataset"].str.endswith("Train")]["UniqueIndividuals"].iloc[0]) if not sub.empty else 0 for sex in sexes}

        fig, ax = plt.subplots(figsize=(6, 5))
        bottoms = {"F": 0, "M": 0}

        for split in splits:
            for sex in sexes:
                row = sub[(sub["Sex"] == sex) & sub["Dataset"].str.endswith(split)]
                cnt = int(row["NumberOfFootprints"].iloc[0]) if not row.empty else 0
                ind_n = int(row["UniqueIndividuals"].iloc[0]) if not row.empty else 0
                trl_n = int(row["UniqueTrails"].iloc[0]) if not row.empty else 0

                ax.bar(sex, cnt, bottom=bottoms[sex], color=colors[split][sex], edgecolor="black", width=0.6)
                if cnt:
                    ax.text(sex, bottoms[sex] + cnt / 2, f"n=({ind_n},{trl_n})", ha="center", va="center", fontsize=10)
                bottoms[sex] += cnt

        ax.set_xticks(sexes)
        ax.set_xticklabels([f'Female\n(n={total_ind["F"]})', f'Male\n(n={total_ind["M"]})'])
        ax.set_ylabel('Number of Footprints')
        ax.set_title(make_italic(code))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(fig_dir / f"{code}_summary.{ext}")
        plt.show()
        plt.close(fig)


def run_summary(
    splits_dir: Path,
    output_table: Path,
    fig_dir: Path,
    *,
    force: bool = False,
    plot: bool = True,
) -> None:
    splits = discover_splits(splits_dir)
    if not splits:
        raise RuntimeError(f"No split files found in {splits_dir}")

    rows: list[pd.DataFrame] = []
    for dataset, parts in splits.items():
        for origin, path in parts.items():
            df = load_split_data(path)
            summary_df = compute_summary(df, dataset, origin)
            if not summary_df.empty:
                rows.append(summary_df)

    df_summary = pd.concat(rows, ignore_index=True)
    if output_table.exists() and not force:
        print(f"Skipping: {output_table} exists. Use --force to overwrite.")
    else:
        output_table.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(output_table, index=False)
        print(f"[SUCCESS] summary written to {output_table}")

    if plot:
        plot_summary_table(df_summary, fig_dir)
