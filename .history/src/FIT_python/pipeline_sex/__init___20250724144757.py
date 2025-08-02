"""High-level API for the sex classification pipeline."""

from .pipeline_wrapper_sex import PipelineWrapper, plot_pipeline_timings
from .baseline_sex import (
    run_baseline_all_species,
    collect_best_metrics,
    plot_accuracy_comparison,
    plot_majority_comparison,
    plot_accuracy_by_sex,
)
from .sex_config import (
    run_otter_search_sex,
    run_other_species_search,
    run_species_search,
)
from .simple_baseline import run_simple_baseline_all_species
from .sex_predict_and_visualisation import (
    predict_all,
    predict_simple_baseline,
    plot_confusion,
    plot_inference,
    plot_confusion_and_inference,
    plot_quality,
    plot_individual_probabilities,
    plot_quality_heatmaps,
    plot_model_quality_heatmaps,
    plot_hyperparam_heatmap,
)

__all__ = [
    "PipelineWrapper",
    "run_pipeline",
    "plot_pipeline_timings",
    "predict_all",
    "plot_confusion",
    "plot_inference",
    "plot_confusion_and_inference",
    "plot_quality",
    "plot_individual_probabilities",
    "plot_quality_heatmaps",
    "plot_model_quality_heatmaps",
    "plot_hyperparam_heatmap",
    "run_baseline_all_species",
    "run_simple_baseline_all_species",
    "predict_simple_baseline",
    "collect_best_metrics",
    "plot_accuracy_comparison",
    "plot_majority_comparison",
    "plot_accuracy_by_sex",
    "run_species_search",
    "run_otter_search_sex",
    "run_other_species_search",
]


def run_pipeline(**kwargs):
    """Prepare data and train the sex-classification models.

    This is a convenience wrapper around :class:`PipelineWrapper`.  All
    keyword arguments are forwarded to :class:`~pipeline_sex.pipeline_wrapper_sex.PipelineWrapper`.

    Parameters
    ----------
    model_keys : list[str], optional
        Identifiers of models to train.  Available keys are defined in
        ``pipeline_wrapper_sex.MODELS``.
    fs_method : str or None, optional
        Feature-selection algorithm. Options: ``forward``, ``random_forest``,
        ``variance``, ``univariate``, ``lasso`` or ``None``.
    fs_k : int, optional
        Number of features selected when ``fs_method`` is not ``None``.
    impute_method : str or None, optional
        Imputation strategy. Options: ``miss_forest`` or ``None``.
    outlier_method : str or None, optional
        Outlier cleaning method. Options: ``clip``, ``zscore`` or ``None``.
    scaler_method : str or None, optional
        Feature scaling approach. Options: ``standard``, ``robust`` or ``None``.
    reduce_pre_method, reduce_post_method : str or None, optional
        Dimensionality reduction before/after feature selection. Options:
        ``pca``, ``umap``, ``tsne`` or ``None``.
    n_jobs : int, optional
        Number of parallel jobs used for cross-validation. ``-1`` uses all
        available CPU cores.
    debug : bool, optional
        If ``True`` additional debug information is printed during training.

    Returns
    -------
    pandas.DataFrame
        The result table produced by :meth:`pipeline_sex.pipeline_wrapper_sex.PipelineWrapper.train`.
    """
    wrapper = PipelineWrapper(**kwargs)
    wrapper.prepare()
    return wrapper.train()
