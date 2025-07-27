from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def aggregate_all_folds(base_dir: Path | str, out_path: Path | None = None) -> pd.DataFrame:
    """Combine per-species ``all_folds.csv`` tables.

    Parameters
    ----------
    base_dir:
        Directory containing one sub-folder per species with an
        ``all_folds.csv`` file.
    out_path:
        Optional location where the combined CSV should be written.
        When not provided, the file is saved as ``all_species_folds.csv``
        inside ``base_dir``.

    Returns
    -------
    pandas.DataFrame
        Concatenated data frame of all folds with an added ``species`` column.
    """
    base_dir = Path(base_dir)
    tables: list[pd.DataFrame] = []
    for species_dir in sorted(base_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        csv_fp = species_dir / "all_folds.csv"
        if not csv_fp.exists():
            continue
        df = pd.read_csv(csv_fp)
        df["species"] = species_dir.name
        tables.append(df)

    if not tables:
        raise FileNotFoundError(f"No 'all_folds.csv' found under {base_dir}")

    result = pd.concat(tables, ignore_index=True)

    if out_path is None:
        out_path = base_dir / "all_species_folds.csv"
    result.to_csv(out_path, index=False)

    return result
