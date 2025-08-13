from __future__ import annotations

from pathlib import Path


import pandas as pd


def aggregate_all_folds(
    base_dir: Path | str,
    out_path: Path | None = None,
    origin: str = "cv",
) -> pd.DataFrame:
    """Combine pairs from ``master_pairs.parquet`` files.

    Parameters
    ----------
    base_dir:
        Directory containing one sub-folder per species with a
        ``master_pairs.parquet`` file.
    out_path:
        Optional location where the combined table should be written.
        When not provided, the file is saved as ``all_species_folds.parquet``
        inside ``base_dir``.
    origin:
        Value of the ``origin`` column to filter by. Typical values are ``"cv"``,
        ``"test"`` and ``"inference"``. Rows with other origins are dropped.

    Returns
    -------
    pandas.DataFrame
        Concatenated data frame of all cross-validation pairs with an added
        ``species`` column.
    """
    base_dir = Path(base_dir)
    tables: list[pd.DataFrame] = []
    for species_dir in sorted(base_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        master_fp = species_dir / "master_pairs.parquet"
        if not master_fp.exists():
            continue
        df = pd.read_parquet(master_fp)
        if "origin" in df.columns:
            df = df[df["origin"] == origin]
        df["species"] = species_dir.name
        tables.append(df)

    if not tables:
        raise FileNotFoundError(f"No 'master_pairs.parquet' found under {base_dir}")

    result = pd.concat(tables, ignore_index=True)

    if out_path is None:
        out_path = base_dir / "all_species_folds.parquet"
    result.to_parquet(out_path, index=False)

    return result
