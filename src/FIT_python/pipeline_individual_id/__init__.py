"""High level API for individual identification baselines."""

# Importing ``simple_baseline`` directly would pull in heavy dependencies during
# test collection.  The helpers below therefore load the actual implementation
# lazily when called while still exposing accurate docstrings and signatures.

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Callable


_SIMPLE_BASELINE = Path(__file__).with_name("simple_baseline.py")


def _load_metadata(func_name: str) -> tuple[str, inspect.Signature]:
    """Return the docstring and :class:`inspect.Signature` of ``func_name``.

    The information is extracted statically from ``simple_baseline.py`` so no
    heavy dependencies need to be imported when this module is loaded. The
    wrapper signatures and docstrings therefore automatically reflect the
    current implementation of the underlying helpers.
    """

    source = _SIMPLE_BASELINE.read_text(encoding="utf-8")
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            doc = ast.get_docstring(node) or ""
            # create a minimal stub to extract the signature without executing
            new_node = ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=[ast.Pass()],
                decorator_list=[],
                returns=node.returns,
                type_comment=node.type_comment,
            )
            stub_mod = ast.Module(body=[new_node], type_ignores=[])
            ast.fix_missing_locations(stub_mod)
            compiled = compile(stub_mod, _SIMPLE_BASELINE.as_posix(), "exec")
            ns: dict[str, object] = {}
            exec(compiled, ns)
            sig = inspect.signature(ns[func_name])
            return doc, sig

    return "", inspect.Signature()  # pragma: no cover - should never happen


def _make_lazy_wrapper(func_name: str) -> Callable:
    """Create a lazy wrapper for ``func_name`` from ``simple_baseline``."""

    doc, sig = _load_metadata(func_name)

    def wrapper(*args, **kwargs):
        from . import simple_baseline as _mod

        impl = getattr(_mod, func_name)
        return impl(*args, **kwargs)

    wrapper.__name__ = func_name
    wrapper.__doc__ = doc
    wrapper.__signature__ = sig
    return wrapper


run_simple_baseline_otter = _make_lazy_wrapper("run_simple_baseline_otter")


run_baseline_all_species = _make_lazy_wrapper("run_baseline_all_species")


run_simple_baseline_all_species = _make_lazy_wrapper(
    "run_simple_baseline_all_species"
)


run_sex_prediction_experiment = _make_lazy_wrapper("run_sex_prediction_experiment")


collect_id_metrics = _make_lazy_wrapper("collect_id_metrics")


plot_bcr_comparison = _make_lazy_wrapper("plot_bcr_comparison")


def run_id_search(*args, **kwargs):
    """Lazy wrapper around :func:`id_config.run_species_search`."""

    from .id_config import run_species_search as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "run_simple_baseline_otter",
    "run_baseline_all_species",
    "run_simple_baseline_all_species",
    "run_sex_prediction_experiment",
    "collect_id_metrics",
    "plot_bcr_comparison",
    "run_id_search",
]

