# feature_scaler_wrapper.py

## Überblick
Skalierungs-Transformer mit StandardScaler oder RobustScaler.
Je nach gewählter Methode wird eine Standard- oder robuste Skalierung ausgeführt. Der Transformer ist sinnvoll, wenn Modelle empfindlich auf unterschiedliche Wertebereiche reagieren. Die Austauschbarkeit erleichtert Experimenten, kann aber bei falscher Wahl zu schlechteren Ergebnissen führen.

## Wichtige Bestandteile
- FeatureScalerTransformer

## Referenzen
- https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features

## Annahmen und Einschränkungen
Gibt ein Array oder DataFrame im selben Typ wie die Eingabe zurück.
Nicht-numerische Spalten werden unverändert durchgereicht.
