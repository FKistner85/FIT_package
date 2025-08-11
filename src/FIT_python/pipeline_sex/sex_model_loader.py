from pathlib import Path
from FIT_python import config as cfg

def resolve_common_species(name: str) -> str:
    """Map wissenschaftlicher Name -> Common-Name laut cfg.SPECIES_MODEL_MAP."""
    key = (name or "").strip().lower()
    return cfg.SPECIES_MODEL_MAP.get(key, key)

def get_sex_model_path(*, species: str, metric: str | None = None) -> Path:
    """
    Pfad zu Sex-Modell basierend auf Species + Metric ermitteln.
    Species kann wissenschaftlich oder Common-Name sein.
    """
    common = resolve_common_species(species)
    mkey = (metric or cfg.SEX_PREDICT_METRIC).strip()
    base = cfg.PATHS["sex_modelling"] / common / "models" / mkey
    fp = base / f"{common}.joblib"
    if not fp.exists():
        raise FileNotFoundError(f"Sex model file not found: {fp}")
    return fp
