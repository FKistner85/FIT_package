"""Basic import test for the sex pipeline search module."""

import importlib


def test_search_import(monkeypatch):
    """Import search module to ensure no syntax errors."""
    from FIT_python.config import CONFIG
    if "run_otter_search" not in CONFIG.get("pipeline_sex", {}):
        CONFIG["pipeline_sex"]["run_otter_search"] = CONFIG["pipeline_sex"].get("run_otter_search_sex", {})
    module = importlib.import_module("FIT_python.pipeline_sex.search")
    importlib.reload(module)
