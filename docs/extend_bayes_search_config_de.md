# Erweiterung der BayesSearchCV-Konfiguration

Dieses kurze How‑To beschreibt, wie sich weitere Vorverarbeitungsschritte und Klassifikatoren in die Sex‑Klassifikation einbinden lassen. Alle Einstellungen werden in `SOFT_CONFIG` gepflegt und in `sex_config.py` zu `SEARCH_SPACES` verarbeitet.

## 1. `SOFT_CONFIG` anpassen

Unter `pipeline_sex.search_spaces` können neue Optionen hinterlegt werden. Beispiel:

```python
SOFT_CONFIG["pipeline_sex"]["search_spaces"].update(
    {
        "outlier": [None, "clip", "zscore"],
        "scale": [None, "standard", "robust"],
        "reduce_pre__method": [None, "pca", "umap_unsupervised", "umap_supervised", "lda"],
        "reduce_post__method": [None, "pca", "umap_unsupervised", "umap_supervised", "lda"],
    }
)
```

Die Listen lassen sich bei Bedarf um weitere Transformatoren ergänzen.

## 2. `SEARCH_SPACES` erweitern

`sex_config.py` wandelt diese Listen in `skopt.space.Categorical` Objekte um. Jeder Parameter erhält einen eigenen Eintrag; mehrere Klassifikatoren werden per `EstimatorWrapper` bereitgestellt:

```python
SEARCH_SPACES = {
    "outlier": Categorical(
        [
            OutlierCleanerTransformer(method="clip"),
            OutlierCleanerTransformer(method="zscore"),
        ],
        transform="identity",
    ),
    "scale": Categorical(
        [
            FeatureScalerTransformer(method="standard"),
            FeatureScalerTransformer(method="robust"),
        ],
        transform="identity",
    ),
    "reduce_pre__method": Categorical(SEARCH_SPACE_CFG["reduce_pre__method"]),
    "reduce_post__method": Categorical(SEARCH_SPACE_CFG["reduce_post__method"]),
    "select__method": Categorical(SEARCH_SPACE_CFG["select__method"]),
    "select__k": Categorical(SEARCH_SPACE_CFG["select__k"]),
    "clf": Categorical(
        [EstimatorWrapper(MODELS[k]) for k in ("rf_small", "rf_med", "xgb_std", "lda")],
        transform="identity",
    ),
}
```

Damit untersucht `BayesSearchCV` unterschiedliche Kombinationen aus Ausreißerbehandlung, Skalierung, Dimensionsreduktion und mehreren Klassifikatoren. `get_pipeline_steps` akzeptiert bereits die notwendigen Argumente – weitere Änderungen sind nicht erforderlich.
