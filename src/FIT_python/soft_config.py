"""Default configuration parameters for pipelines."""

SOFT_CONFIG = {
    "data_split_and_summary": {
        "sex_categories": ["Female", "Male"],
        "split_labels": ["Train", "Test"],
        "species_remap": {
            "a_j_soemmeringii": "acinonyx_jubatus_soemmeringii",
            "a_j_jubatus": "acinonyx_jubatus_jubatus",
        },
    },
    "general_pipeline_steps": {
        "outlier_defaults": {
            "method": "clip",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "z_thresh": 3.0,
        },
        "scaler_default": {"method": "standard"},
        "imputation_defaults": {
            "n_estimators": 10,
            "max_iter": 10,
            "random_state": 42,
        },
        "dim_reducer_defaults": {
            "method": "pca",
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "whiten": False,
        },
    },
    "pipeline_sex": {
        "model_keys": [
            "logreg_l2",
            "logreg_l1",
            "rf_small",
            "rf_med",
            "rf_large",
            "knn_3",
            "knn_5",
            "knn_7",
            "svm_linear",
            "svm_rbf",
            "xgb_std",
            "xgb_hist",
            "lgbm_std",
            "lgbm_md10",
            "lda",
        ],
        "search_spaces": {
            "select__method": [None, "forward", "lasso", "variance", "random_forest"],
            "select__k": [1, 2, 3, 4, 5, 6, 10, 20, 50, 100],
            "clf": None,  # placeholder, to be filled with MODELS
        },
        "metrics": [
            "maj_test_pct",
            "balanced_test_acc",
            "accuracy_test",
            "mean_test_neg_log_loss",
        ],
        "scoring": {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "neg_log_loss": "neg_log_loss",
        },
        "pipeline_order": [
            "outlier",
            "scale",
            "select__method",
            "select__k",
            "reduce_pre",
            "reduce_post",
            "clf",
        ],
        "run_otter_search_sex": {
            "n_iter": 2,
            "cv": 2,
            "random_state": 42,
        },
    },
    "pipeline_individual_id": {
        "pairwise_defaults": {
            "k_features": 15,
            "reducers": ["lda"],
            "selection_method": "forward",
            "n_components": 2,
        },
        "trail_generation_defaults": {
            "sample_size": 9,
            "subsample_sizes": [3, 5, 7],
            "n_candidates": 20,
        },
        "sequential_holdout_val_sizes": [2, 4, 6, 8],
    },
}
