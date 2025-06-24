from pathlib import Path
import pandas as pd
from typing import List, Dict

from FIT_python.config import RAW_DIR, RESULTS_DATA_DIR
from FIT_python.splits import all_splits


META_COLS = ["species", "individual_id", "trail", "sex"]


def summary_stats(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "UniqueIndividuals": df["individual_id"].nunique(),
        "Trails": df["trail"].nunique(),
        "Rows": len(df),
        "Features": df.shape[1] - 4,
    }


def summarize_dataset(name: str, split_name: str, df: pd.DataFrame, *, add_origin: bool) -> List[Dict[str, object]]:
    """Return summary rows for one dataset split."""
    rows: List[Dict[str, object]] = []
    species = df["species"].iloc[0]
    base = {"Species": species, "Split": split_name}

    # Overall totals
    row = {**base, "Dataorigin": "All", "Sex": "Total", **summary_stats(df)}
    rows.append(row)
    for sex in sorted(df["sex"].dropna().unique()):
        sub = df[df["sex"] == sex]
        rows.append({**base, "Dataorigin": "All", "Sex": sex, **summary_stats(sub)})

    if add_origin and "dataorigin" in df.columns:
        for origin, df_origin in df.groupby("dataorigin"):
            base_o = {**base, "Dataorigin": origin}
            rows.append({**base_o, "Sex": "Total", **summary_stats(df_origin)})
            for sex in sorted(df_origin["sex"].dropna().unique()):
                sub = df_origin[df_origin["sex"] == sex]
                rows.append({**base_o, "Sex": sex, **summary_stats(sub)})
    return rows


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path)
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")


def main() -> None:
    splits = all_splits(RAW_DIR)
    # add datasets skipped during splitting as unsplit 'full'
    loaded_raw = {k: v for k, v in splits.items()}
    for file in RAW_DIR.glob("*.csv"):
        name = file.stem.replace(" ", "_")
        if name not in loaded_raw:
            df = pd.read_csv(file)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            if "animal" in df.columns:
                df.rename(columns={"animal": "individual_id"}, inplace=True)
            splits[name] = {"full": df}

    rows = []
    for name, parts in splits.items():
        for split_name, df in parts.items():
            species_name = str(df["species"].iloc[0]).lower()
            add_origin = "lutra" in species_name and "lutra" in species_name.split()
            rows.extend(summarize_dataset(name, split_name, df, add_origin=add_origin))

    summary = pd.DataFrame(rows)

    # Global totals per split
    totals = []
    for split in summary["Split"].unique():
        for sex in ["F", "M", "Total"]:
            df_sex = summary[(summary["Split"] == split) & (summary["Sex"] == sex) & (summary["Dataorigin"] == "All")]
            if df_sex.empty:
                continue
            totals.append({
                "Species": "Total",
                "Split": split,
                "Dataorigin": "All",
                "Sex": sex,
                "UniqueIndividuals": df_sex["UniqueIndividuals"].sum(),
                "Trails": df_sex["Trails"].sum(),
                "Rows": df_sex["Rows"].sum(),
                "Features": df_sex["Features"].iloc[0],
            })
    summary = pd.concat([summary, pd.DataFrame(totals)], ignore_index=True)

    gesamt_dir = RESULTS_DATA_DIR / "gesamt"
    save_df(summary, gesamt_dir / "summary_datasets.csv")

    for species in summary["Species"].unique():
        if species == "Total":
            continue
        sp_df = summary[summary["Species"] == species]
        sp_dir = RESULTS_DATA_DIR / species.strip().replace(" ", "_")
        save_df(sp_df, sp_dir / "summary.csv")


if __name__ == "__main__":
    main()
