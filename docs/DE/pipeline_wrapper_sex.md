# pipeline_wrapper_sex.py

## Überblick
Orchestriert Import, Splitting, Zusammenfassung und Modelltraining für die Geschlechtsklassifikation.
Der Wrapper bildet den gesamten Workflow von der Datenaufbereitung bis zum gespeicherten Modell ab. Er ermöglicht reproduzierbare Experimente und verpackt gängige Schritte in einer Funktion. Nachteil ist, dass ungewöhnliche Pipelines nur schwer abzubilden sind.

## Wichtige Bestandteile
- PipelineWrapper
- get_pipeline_steps

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## Annahmen und Einschränkungen
Legt Pipelines auf der Festplatte ab und speichert das beste Modell pro Art.
