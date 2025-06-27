# transform_wrapper.py

Der NumericTransformer wandelt DataFrames in numerische Matrizen um.
Beim Fit merkt er sich alle Spalten, die als Features dienen sollen und ignoriert Meta- und Zielspalten.
Während transform ersetzt er Kommas durch Punkte, konvertiert die Werte zu Fließzahlen und liefert ein NumPy‑Array der ausgewählten Spalten.
