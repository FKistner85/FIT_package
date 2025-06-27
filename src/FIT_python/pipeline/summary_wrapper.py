from __future__ import annotations

"""Simple wrapper around :func:`run_summary`."""

from pathlib import Path
import sys

from FIT_python import config as cfg
from .summary_utils import run_summary as _run_summary

__all__ = ["run_summary", "SummaryWrapper"]


def run_summary(*args, **kwargs):
    """Call :func:`~summary_utils.run_summary` directly."""
    return _run_summary(*args, **kwargs)


class SummaryWrapper:
    def summarize_all(
        self,
        splits_dir: Path | None = None,
        output_table: Path | None = None,
        fig_dir: Path | None = None,
    ) -> int:
        try:
            run_summary(
                splits_dir or cfg.SPLITS_DIR,
                output_table or cfg.RESULTS_DATA_DIR / "summary.csv",
                fig_dir or cfg.FIGURES_DIR / "summary",
                force=True,
                plot=True,
            )
            return 0
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            if getattr(cfg, "DEBUG_MODE", False):
                raise
            return 1
