# pipeline_wrapper.py

Das Modul vereint zwei zentrale Bestandteile.
Die Funktion get_pipeline_steps stellt abhängig von den gewählten Optionen eine Liste von Vorverarbeitungsschritten zusammen.
Darauf baut die Klasse PipelineWrapper auf, die Trainingsläufe über verschiedene Daten und Modelle automatisiert.
Ihre run-Methode sorgt für gültige Splits, führt Cross-Validation mit den Standard fünf Folds von scikit-learn aus und speichert sowohl Metriken als auch die besten Modelle.
