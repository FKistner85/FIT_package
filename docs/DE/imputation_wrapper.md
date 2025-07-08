# imputation_wrapper.py

## Überblick
Wrapper um den IterativeImputer mit RandomForestRegressor.
Der Wrapper nutzt einen Random-Forest-Regressor innerhalb des IterativeImputers. Fehlende Werte werden dadurch anhand der verbleibenden Merkmale geschätzt. Dies führt zu plausibleren Ergebnissen, kann bei großen Datenmengen jedoch sehr rechenintensiv sein.

## Wichtige Bestandteile
- ImputationWrapper

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

## Annahmen und Einschränkungen
Imputiert nur numerische Spalten und wird auf dem numerischen Teil des DataFrames angepasst.
