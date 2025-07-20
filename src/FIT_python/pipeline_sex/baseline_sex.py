"""Baseline sex-classification helper."""

from __future__ import annotations

from pathlib import Path

from FIT_python.config import SPLITS_DIR
from .pipeline_wrapper_sex import PipelineWrapper


def run_baseline_all_species(exp_dir: Path) -> None:
    """Train baseline LDA classifiers for each species.

    Parameters
    ----------
    exp_dir:
        Directory where ``raw_results.csv`` and best models will be stored.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_dir / "models"
    model_dir.mkdir(exist_ok=True)
    best_dir = exp_dir / "best_models"
    best_dir.mkdir(exist_ok=True)

    species_dirs = [d for d in sorted(SPLITS_DIR.iterdir()) if d.is_dir()]
    if not species_dirs:
        raise FileNotFoundError(f"No split directories found in {SPLITS_DIR}")
    names = ", ".join(d.name for d in species_dirs)
    print(f"[INFO] Found species: {names}")

    wrapper = PipelineWrapper(model_keys=["lda"], fs_method="forward")
    wrapper._model_dir = model_dir
    wrapper._best_dir = best_dir

    wrapper.prepare()

    # PipelineWrapper internally iterates over ``data/splits``
    # and trains a model for each species directory found there.
    df = wrapper.train()

    df.to_csv(exp_dir / "raw_results.csv", index=False)

    return None
