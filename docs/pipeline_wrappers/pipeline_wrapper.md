# pipeline_wrapper.py

Dieses Modul enthält `get_pipeline_steps` und die Klasse `PipelineWrapper`.
Ersteres baut abhängig von den gewählten Optionen eine Liste von Vorverarbeitungsschritten auf, letzterer organisiert Training und Evaluation.

Eine ausführlichere Beschreibung der einzelnen Transformatoren befindet sich in den Dateien dieses Ordners sowie im Dokument `pipeline_sex_methodology.md`.

`get_pipeline_steps` erstellt abhängig von den gewählten Hyperparametern eine Liste von Vorverarbeitungsschritten. Dazu zählen numerische Konvertierung, optionale Imputation, Ausreißerbereinigung, Skalierung, Dimensionsreduktion und Feature-Selektion. Die Funktion prüft dabei, ob die angegebenen Methoden erlaubt sind und gibt eine konsistente Reihenfolge für den Aufbau eines `Pipeline`-Objekts zurück.

Die Klasse `PipelineWrapper` führt diesen Ablauf end-to-end aus. Mit `prepare()` werden Import, Splits und Zusammenfassungen nur einmal erzeugt. `train()` trainiert für jede Spezies verschiedene Modellvarianten, misst die Balanced Accuracy und speichert sowohl einzelne Zwischenschritte als auch das jeweils beste Modell ab. Damit eignet sich der Wrapper für systematische Vergleiche, birgt jedoch den Nachteil längerer Laufzeiten, wenn viele Kombinationen ausprobiert werden.

### Referenzen
* Die [scikit-learn Dokumentation zu Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) beschreibt das zugrundeliegende Konzept.
