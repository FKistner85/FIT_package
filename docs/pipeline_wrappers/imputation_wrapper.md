# imputation_wrapper.py

Die Klasse ImputationWrapper setzt auf einen IterativeImputer mit Random‑Forest‑Regressor, um fehlende Werte zu schätzen.
Beim Erzeugen des Objekts lassen sich Anzahl der Bäume, Iterationen und ein Zufallszustand angeben.
Die fit-Methode trainiert den Imputer auf allen numerischen Spalten.
transform füllt anschließend die entsprechenden Lücken in DataFrames oder Arrays auf.
