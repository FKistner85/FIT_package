# pipeline_wrapper_sex.py

## Überblick
Orchestriert Import, Splitting, Zusammenfassung und Modelltraining für die Geschlechtsklassifikation.

## Wichtige Bestandteile
- PipelineWrapper
- get_pipeline_steps

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## Annahmen und Einschränkungen
Caches pipelines on disk; saves best model per species.
