# feature_scaler_wrapper.py

Der FeatureScalerTransformer skaliert numerische Eingaben.
Je nach Einstellung nutzt er eine Standardisierung oder eine robuste Skalierung.
In fit wird der entsprechende Scikit‑Learn‑Scaler auf die Daten angepasst.
transform skaliert danach neue Beobachtungen und gibt bei Bedarf einen DataFrame mit den ursprünglichen Spalten zurück.
