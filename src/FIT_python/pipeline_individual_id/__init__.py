"""High level API for individual identification baselines."""

# Importing ``simple_baseline`` at module level causes a ``SyntaxError`` due to
# leftover merge markers in that file. To keep imports lazy and avoid the error
# during test collection we provide small wrappers that load the module only
# when the corresponding function is called.


def run_simple_baseline_otter(*args, **kwargs):
    """Lazy wrapper around :func:`simple_baseline.run_simple_baseline_otter`."""

    from .simple_baseline import run_simple_baseline_otter as _impl

    return _impl(*args, **kwargs)


def run_baseline_all_species(*args, **kwargs):
    """Lazy wrapper around :func:`simple_baseline.run_baseline_all_species`."""

    from .simple_baseline import run_baseline_all_species as _impl

    return _impl(*args, **kwargs)


def run_simple_baseline_all_species(*args, **kwargs):
    """Lazy wrapper around :func:`simple_baseline.run_simple_baseline_all_species`."""

    from .simple_baseline import run_simple_baseline_all_species as _impl

    return _impl(*args, **kwargs)


def run_sex_prediction_experiment(*args, **kwargs):
    """Lazy wrapper around :func:`simple_baseline.run_sex_prediction_experiment`."""

    from .simple_baseline import run_sex_prediction_experiment as _impl

    return _impl(*args, **kwargs)


def collect_id_metrics(*args, **kwargs):
    """Lazy wrapper around :func:`simple_baseline.collect_id_metrics`."""

    from .simple_baseline import collect_id_metrics as _impl

    return _impl(*args, **kwargs)


def plot_bcr_comparison(*args, **kwargs):
    """Lazy wrapper around :func:`simple_baseline.plot_bcr_comparison`."""

    from .simple_baseline import plot_bcr_comparison as _impl

    return _impl(*args, **kwargs)


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

