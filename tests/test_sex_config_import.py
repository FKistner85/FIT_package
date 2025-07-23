"""Basic import test for sex_config."""

import importlib


def test_sex_config_import(monkeypatch):
    """Import sex_config to ensure no syntax errors."""
    from FIT_python.soft_config import SOFT_CONFIG
    if "run_otter_search" not in SOFT_CONFIG.get("pipeline_sex", {}):
        SOFT_CONFIG["pipeline_sex"]["run_otter_search"] = SOFT_CONFIG["pipeline_sex"].get("run_otter_search_sex", {})
    module = importlib.import_module("FIT_python.pipeline_sex.sex_config")
    importlib.reload(module)

