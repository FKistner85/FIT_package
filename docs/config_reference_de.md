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
- `search_spaces`: Suchräume für BayesSearchCV mit `Categorical`-Objekten.
- `metrics`: erfasste Metriken während der Suche.
- `scoring`: Mapping der Scoring-Bezeichner von sklearn.
- `pipeline_order`: Reihenfolge der Kennzeichen in Pipeline-IDs.
- `run_otter_search_sex`: Standardwerte für die Hilfsfunktion `run_otter_search_sex`.
- `run_species_search`: Standardwerte für die vereinheitlichte Suchfunktion.
  Beide Funktionen besitzen das Flag `reuse_results`, um vorhandene
  CSV-Dateien und Modelle aus `RESULTS_DATA_DIR` zu laden und die Suche zu
  überspringen.
- Wird `"cv": "fold"` gesetzt, nutzt die Suche die `Fold`-Spalte über `PredefinedSplit`.

## pipeline_individual_id
- `pairwise_defaults`: Voreinstellungen für Paar-Embedding-Pipelines.
  - `k_features`: Anzahl der auszuwählenden Merkmale.
  - `reducers`: Liste der Methoden zur Dimensionsreduktion.
  - `selection_method`: Verfahren zur Merkmalsauswahl.
  - `n_components`: Ziel-Dimensionalität der Reduktion.
  - `outlier_methods`: Liste der Ausreißer-Methoden oder `None`.
  - `scaler_methods`: Liste der Skalierungsverfahren oder `None`.
  - `use_sexmodel_prediction`: Ob Sex-Modell-Vorhersagen angehängt werden.
- `sequential_holdout_val_sizes`: Validierungsgrößen für die sequentiellen Holdouts.

## gui_annotator
Die verwendeten Pfade werden aus :mod:`FIT_python.config` abgeleitet:
- Rohbilder: ``config.RAW_DIR / "images"``
- Bearbeitete Bilder: ``config.PROCESSED_DIR / "images"``
- Annotationen: ``config.PROCESSED_DIR / "annotations"``
- Referenzvorlagen: ``config.RAW_DIR / "reference_templates"``
- `default_scale`: Standard-Skalierungsfaktor bei der Verarbeitung.
- `display_size`: Auflösung der Zeichenfläche.

---
Künftige Erweiterungen sollten auf diese Variablen Bezug nehmen, anstatt Werte direkt im Code zu hinterlegen.
