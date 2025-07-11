# pipeline_wrapper_sex.py

Dieser Wrapper orchestriert das gesamte Training der Sex-Klassifikationspipeline. Er ruft die einzelnen Transformatoren aus diesem Ordner auf und nutzt die Modellliste aus `models.py`.
Der Ablauf besteht aus den Schritten Import, Split-Erzeugung, Zusammenfassung und eigentlichem Modelltraining. Dabei können verschiedene Hyperparameter für Feature-Selektion, Imputation oder Dimensionsreduktion ausprobiert werden.

Nach Abschluss des Trainings werden alle Pipelines auf Platte gespeichert und pro Tierart wird das jeweils beste Modell separat abgelegt. Das ist praktisch für spätere Vorhersagen, führt aber zu einem größeren Speicherbedarf.

Details zur Methodik stehen in `pipeline_sex_methodology.md`.

## Modell-Caching

Seit Version 1.2 prüft `PipelineWrapper.train()` nach dem Zusammenbau des Dateinamens,
ob das jeweilige Modellartefakt bereits unter `results/data/sex_models` existiert.
Ist dies der Fall, wird das Modell mit `joblib.load()` geladen und für die
Vorhersagen verwendet statt neu trainiert zu werden. Über die Konsole wird dies
mit entsprechenden Print-Ausgaben dokumentiert. Nur wenn kein Artefakt gefunden
wird, erfolgt ein erneutes Fitten und Speichern.

## Cross-Validation Predictions

Ab Version 1.1 werden während `PipelineWrapper.train()` zusätzlich
Out-of-Fold-Vorhersagen pro Modell berechnet. Diese werden als Spalten
`pred_<modell>_cv_sex` in den Trainingsdaten gespeichert und landen
somit auch in den von `predict_all()` erzeugten CSV-Dateien. Die
Funktion `plot_confusion()` nutzt diese Spalten, um eine zweigeteilte
Matrix auszugeben: links die zusammengefassten CV-Ergebnisse der
Trainingsdaten, rechts die Vorhersagen auf dem Testsatz.

### Referenzen
* Siehe auch die Dokumentation zu `PipelineWrapper` für generelle Hinweise zum Aufbau.
