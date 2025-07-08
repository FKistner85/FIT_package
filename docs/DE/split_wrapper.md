# split_wrapper.py

## Überblick
Hochrangige Schnittstelle zum Laden bereinigter Daten und Schreiben von Train-/Test-Folds.
Der Wrapper nutzt die Funktionen aus split_utils und speichert die resultierenden Folds als Dateien. So können verschiedene Modelle auf exakt denselben Splits getestet werden. Bei Änderungen der Parameter müssen die Dateien jedoch neu erzeugt werden.

## Wichtige Bestandteile
- SplitWrapper.split_all

## Referenzen
- https://scikit-learn.org/stable/modules/cross_validation.html

## Annahmen und Einschränkungen
Erstellt pro Art Parquet- und CSV-Dateien.
