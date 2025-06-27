# outlier_wrapper.py

Der OutlierCleanerTransformer mindert Ausreißer in numerischen Merkmalen.
Beim Anlegen entscheidet man sich für Clipping nach Perzentilen oder für eine Z‑Score‑Begrenzung.
Die fit-Methode berechnet je nach Variante die benötigten Quantile oder Statistikwerte.
In transform werden die Werte daran angepasst, wobei DataFrames ihre Spaltenbezeichnungen behalten.
