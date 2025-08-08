from __future__ import annotations

from pathlib import Path

from FIT_python.config import RESULTS_DIR, DATA_DIR, PATHS


def get_species_paths(species: str, category: str = "sex_modelling") -> dict[str, Path]:
    """Return result directories for a species within ``category``.

    Parameters
    ----------
    species:
        Species key used for the directory name.
    category:
        One of ``dataprocessing``, ``sex_modelling`` or ``individual_id``.
        Defaults to ``sex_modelling`` for backwards compatibility.
    """

    base_root = PATHS.get(category, RESULTS_DIR / category)
    base_dir = base_root / species
    base_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name in ["models", "predictions", "heatmaps", "logs", "search", "figures", "tables"]:
        path = base_dir / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    paths["splits"] = DATA_DIR / "splits" / species
    return paths
