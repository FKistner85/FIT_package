"""High-level API for the sex classification pipeline."""

from .pipeline_wrapper_sex import PipelineWrapper, plot_pipeline_timings
from .baseline_sex import (
    run_baseline_all_species,
    collect_best_metrics,
    plot_accuracy_comparison,
    plot_majority_comparison,
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
]


def run_pipeline(**kwargs):
    """Prepare data and train the sex-classification models.

    Parameters are forwarded to :class:`PipelineWrapper`.  Returns the
    DataFrame produced by ``PipelineWrapper.train``.
    """
    wrapper = PipelineWrapper(**kwargs)
    wrapper.prepare()
    return wrapper.train()
