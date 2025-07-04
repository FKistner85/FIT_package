import os
import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


def write_config(pkg_dir: Path) -> None:
    cfg = pkg_dir / "config.py"
    data_dir = pkg_dir.parent / "data"
    text = """from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
CLEANED_DIR = DATA_DIR / 'cleaned'
SPLITS_DIR = DATA_DIR / 'splits'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_SPLITS_DIR = PROCESSED_DIR / 'splits'
SCALED_DIR = PROCESSED_DIR / 'scaled'
FEATURE_SELECTED_DIR = PROCESSED_DIR / 'feature_selected'
DIM_REDUCED_DIR = PROCESSED_DIR / 'dim_reduced'
NUMERIC_DIR = PROCESSED_DIR / 'numeric'
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DATA_DIR = RESULTS_DIR / 'data'
FIGURES_DIR = RESULTS_DIR / 'figures'
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'
OTTER_META_COLS = ['id', 'dataorigin']
DEFAULT_TARGETS = ['species', 'individual_id', 'trail', 'sex']
GROUP_COL = 'individual_id'
STRATIFY_COL = 'sex'
GLOBAL_RANDOM_SEED = 42
TEST_SIZE = 0.2
NUM_FOLDS = 3
DEFAULT_SCALER = 'standard'
SCALER_PARAMS = {'standard': {}, 'robust': {}}
LOG_LEVEL = 'INFO'
DEBUG_MODE = True

def normalize_dataset_name(name: str) -> str:
    return name.strip().replace(' ', '_').lower()

FS_DEFAULT_METHODS = [
    'forward_count', 'forward_p', 'random_forest',
    'anova_kbest', 'mutual_info', 'chi2_kbest', 'l1_logistic'
]
FS_TARGET_FEATURE_COUNTS = [10, 20, 30]
FS_P_THRESH = 0.05
FS_MIN_NUM_FEATURES = 3
FS_N_JOBS = 1
VALIDATION_MODE = 'external_folds'
NUM_KFOLDS = 5
"""
    cfg.write_text(text)


def prepare_package(tmp_path: Path, src_root: Path) -> Path:
    pkg_src = src_root / "FIT_python"
    pkg_dst = tmp_path / "FIT_python"
    shutil.copytree(pkg_src, pkg_dst)
    write_config(pkg_dst)
    return pkg_dst


def create_dummy_raw(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "species": ["otter"] * 8,
            "individual_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "trail": [1, 2, 1, 2, 1, 2, 1, 2],
            "sex": ["F", "F", "M", "M", "F", "M", "F", "M"],
            "dataorigin": ["Own Data Collection"] * 6 + ["Fieldprints Portugal"] * 2,
            "feat1": range(8),
            "feat2": range(10, 18),
            "feat3": np.linspace(0, 1, 8),
        }
    )
    df.to_csv(raw / "demo_otter.csv", index=False)


def run_script(script: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_end_to_end_pipeline(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    pkg_dir = prepare_package(tmp_path, repo_root / "src")
    create_dummy_raw(tmp_path)

    env = {**os.environ, "PYTHONPATH": f'{tmp_path}:{repo_root / "src"}'}

    results = {}
    # Step 1: Cleaning
    run_script(scripts_dir / "clean_data.py", env)
    cleaned_dir = pkg_dir.parent / "data" / "cleaned"
    assert cleaned_dir.exists()
    first_file = next(cleaned_dir.glob("*.parquet"))
    df_clean = pd.read_parquet(first_file)
    assert "individual_id" in df_clean.columns
    results["clean"] = "ok"

    # Step 2: Splitting
    run_script(scripts_dir / "create_splits.py", env)
    splits_dir = pkg_dir.parent / "data" / "processed" / "splits"
    train_file = splits_dir / "demo_otter_train.parquet"
    assert train_file.exists()
    df_split = pd.read_parquet(train_file)
    assert {"sex", "individual_id"} <= set(df_split.columns)
    results["split"] = "ok"

    # Step 3: Summary
    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "summary.py"),
            "--splits-dir",
            str(splits_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    summary_csv = pkg_dir.parent / "results" / "data" / "summary.csv"
    assert summary_csv.exists()
    df_sum = pd.read_csv(summary_csv)
    expected = {
        ("demo_otter", o.capitalize(), s)
        for o in ["train", "test"]
        for s in ["F", "M", "Unknown"]
    }
    combos = set(zip(df_sum["Dataset"], df_sum["Origin"], df_sum["Sex"]))
    assert expected.issubset(combos)
    results["summary"] = "ok"

    # Step 4: Scaling
    run_script(scripts_dir / "scale_splits.py", env)
    scaled_train = (
        pkg_dir.parent / "data" / "processed" / "scaled" / "demo_otter_train.parquet"
    )
    assert scaled_train.exists()
    df_scaled = pd.read_parquet(scaled_train)
    feat_cols = ["feat1", "feat2", "feat3"]
    assert np.allclose(df_scaled[feat_cols].mean(), 0, atol=1e-6)
    assert np.allclose(df_scaled[feat_cols].std(ddof=0), 1, atol=1e-6)
    results["scale"] = "ok"

    # Step 5: Feature selection
    run_script(scripts_dir / "create_feature_selection.py", env)
    fs_train = (
        pkg_dir.parent
        / "data"
        / "processed"
        / "feature_selected"
        / "demo_otter_train.parquet"
    )
    assert fs_train.exists()
    df_fs = pd.read_parquet(fs_train)
    meta_cols = ["id", "date", "location", "dataorigin", "substrate"]
    selected_features = [
        c
        for c in df_fs.columns
        if c not in meta_cols + ["species", "individual_id", "trail", "sex"]
    ]
    assert len(selected_features) == 2
    results["feature"] = "ok"

    # Step 6: Dimensionality reduction
    run_script(scripts_dir / "create_dim_reduction.py", env)
    dim_file = (
        pkg_dir.parent
        / "data"
        / "processed"
        / "dim_reduced"
        / "demo_otter_train.parquet"
    )
    assert dim_file.exists()
    df_dim = pd.read_parquet(dim_file)
    assert {"PC1", "PC2"} <= set(df_dim.columns)
    results["dim"] = "ok"

    # Step 7: Model comparison
    run_script(scripts_dir / "compare_models.py", env)
    mc_csv = pkg_dir.parent / "results" / "data" / "model_comparison.csv"
    assert mc_csv.exists()
    df_mc = pd.read_csv(mc_csv)
    assert len(df_mc["Model"].unique()) >= 2
    results["compare"] = "ok"

    print("\nIntegration summary:")
    for step in ["clean", "split", "summary", "scale", "feature", "dim", "compare"]:
        print(f" {step}: {results.get(step, 'failed')}")
