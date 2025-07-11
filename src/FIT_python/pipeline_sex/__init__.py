"""High-level API for the sex classification pipeline."""

from .pipeline_wrapper_sex import PipelineWrapper, plot_pipeline_timings
from .sex_predict_and_visualisation import (
    predict_all,
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
]


def run_pipeline(**kwargs):
    """Prepare data and train the sex-classification models.

    Parameters are forwarded to :class:`PipelineWrapper`.  Returns the
    DataFrame produced by ``PipelineWrapper.train``.
    """
    wrapper = PipelineWrapper(**kwargs)
    wrapper.prepare()
    return wrapper.train()
