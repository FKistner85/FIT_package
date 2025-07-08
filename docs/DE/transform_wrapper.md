# transform_wrapper.py

## Überblick
NumericTransformer wandelt DataFrame-Features in eine NumPy-Matrix um.
Der Transformer erzeugt aus Pandas-Daten einen reinen numerischen Matrix-Input. Dabei bleiben die Ziel- und Metadaten unangetastet. Er ist grundlegend für Modelle, die NumPy-Arrays erwarten, macht aber keine Aussagen über Feature-Skalierung.

## Wichtige Bestandteile
- NumericTransformer

## Referenzen
- https://pandas.pydata.org/docs/

## Annahmen und Einschränkungen
Ignoriert Metadaten- und Zielspalten, die in der Konfiguration definiert sind.
