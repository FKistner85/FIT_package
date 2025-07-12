# models.py

## Überblick
Dictionary vordefinierter scikit-learn- und Gradient-Boosting-Klassifikatoren.
Die Sammlung enthält gängige Klassifikatoren wie Random Forest, SVM und Gradient Boosting mit sinnvollen Voreinstellungen. Sie ermöglicht einen schnellen Einstieg in Experimente. Nachteil ist die geringere Flexibilität bei speziellen Parametern.

## Wichtige Bestandteile
- MODELS

### MODELS
Dictionary, das kurze String-Schlüssel auf vorkonfigurierte Klassifikatoren abbildet, etwa logistische Regression, Random Forests und Gradient-Boosting-Varianten. Die benötigten Bibliotheken wie XGBoost, LightGBM und CatBoost müssen installiert sein, damit die jeweiligen Einträge funktionieren. Die Voreinstellungen decken einen Bereich von leichten bis zu komplexen Modellen ab; die Parameter können angepasst werden, falls Trainingsdauer oder Genauigkeit nicht passen.

## Referenzen
- https://scikit-learn.org/stable/

## Annahmen und Einschränkungen
Hyperparameter sind für verschiedene Modellkomplexitäten voreingestellt.
