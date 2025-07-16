# Konfigurationsreferenz

Dieses Dokument listet alle Parameter auf, die im Repository eingestellt werden können. **Alle automatisch erstellten Codes müssen ihre Werte aus `FIT_python.soft_config.SOFT_CONFIG` lesen und dürfen keine festen Literale verwenden.**

## data_split_and_summary
- `sex_categories`: Kategorien zur Zusammenfassung der Datensätze. Der
  Standardwert ist `['Female', 'Male']`, entsprechend dem Ergebnis der
  Funktion `map_sex()`.
- `split_labels`: Bezeichnungen für Train/Test-Splits.
- `species_remap`: Zuordnung der Kurzarten zu den wissenschaftlichen Namen.

## general_pipeline_steps
- `outlier_defaults`: Voreinstellungen für `OutlierCleanerTransformer`.
- `scaler_default`: Standard-Skalierungsmethode.
- `imputation_defaults`: Einstellungen für den iterativen Imputer.
- `dim_reducer_defaults`: Parameter für die Dimensionsreduktion.

## pipeline_sex
- `model_keys`: Reihenfolge der Klassifikatorkürzel aus `MODELS`.
- `param_distributions`: Suchraum für die Hyperparameter.
- `metrics`: erfasste Metriken während der Suche.
- `scoring`: Mapping der Scoring-Bezeichner von sklearn.
- `pipeline_order`: Reihenfolge der Kennzeichen in Pipeline-IDs.
- `run_otter_search`: Standardwerte für die Hilfsfunktion `run_otter_search`.

## pipeline_individual_id
- `pairwise_defaults`: Voreinstellungen für Paar-Embedding-Pipelines.
- `sequential_holdout_val_sizes`: Validierungsgrößen für die sequentiellen Holdouts.

---
Künftige Erweiterungen sollten auf diese Variablen Bezug nehmen, anstatt Werte direkt im Code zu hinterlegen.
