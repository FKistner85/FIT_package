# pipeline_wrapper_sex.py

Dieser Wrapper orchestriert das gesamte Training der Sex-Klassifikationspipeline. Er ruft die einzelnen Transformatoren aus diesem Ordner auf und nutzt die Modellliste aus `models.py`.
Der Ablauf besteht aus den Schritten Import, Split-Erzeugung, Zusammenfassung und eigentlichem Modelltraining. Dabei können verschiedene Hyperparameter für Feature-Selektion, Imputation oder Dimensionsreduktion ausprobiert werden.

Nach Abschluss des Trainings werden alle Pipelines auf Platte gespeichert und pro Tierart wird das jeweils beste Modell separat abgelegt. Das ist praktisch für spätere Vorhersagen, führt aber zu einem größeren Speicherbedarf.

Details zur Methodik stehen in `pipeline_sex_methodology.md`.

### Referenzen
* Siehe auch die Dokumentation zu `PipelineWrapper` für generelle Hinweise zum Aufbau.
