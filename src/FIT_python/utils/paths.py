from __future__ import annotations

from pathlib import Path

from FIT_python.config import RESULTS_DATA_DIR


def get_species_paths(species: str) -> dict[str, Path]:
    """Return result directories for a species.

    For the given ``species`` name, this helper creates and returns a
    dictionary containing paths for the sub-directories ``models``,
    ``predictions``, ``heatmaps``, ``logs``, ``search`` and ``splits`` under
    ``RESULTS_DATA_DIR / <species>``.
    """
    base_dir = RESULTS_DATA_DIR / species
    base_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name in ["models", "predictions", "heatmaps", "logs", "search", "splits"]:
        path = base_dir / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths
