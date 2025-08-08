from __future__ import annotations

from pathlib import Path

from FIT_python.config import RESULTS_DIR, PATHS


def get_species_paths(species: str, section: str = "sex_modelling") -> dict[str, Path]:
    """Return result directories for a species within ``section``.

    Parameters
    ----------
    species:
        Species key used for the directory name.
    section:
        One of ``dataprocessing``, ``sex_modelling`` or ``individual_id``.
        Defaults to ``sex_modelling`` for backwards compatibility.

    Returns
    -------
    dict[str, Path]
        Mapping of directory names to :class:`pathlib.Path` objects. Always
        contains ``figures`` and ``tables``. For modelling sections
        (``sex_modelling`` and ``individual_id``) an additional ``models``
        directory is included.
    """

    base_root = PATHS.get(section, RESULTS_DIR / section)
    base_dir = base_root / species
    base_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    figures_dir = base_dir / "figures"
    tables_dir = base_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths["figures"] = figures_dir
    paths["tables"] = tables_dir

    if section in {"sex_modelling", "individual_id"}:
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        paths["models"] = models_dir

    return paths
