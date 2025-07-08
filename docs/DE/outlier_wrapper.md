# outlier_wrapper.py

## Überblick
Bereinigt Ausreißer per Quantil-Clipping oder Z-Score-Grenzen.
Der Transformer bereitet numerische Daten für nachgelagerte Analysen auf. Er eignet sich vor allem für Sensordaten, bei denen Ausreißer häufig auftreten. Vorteilhaft ist die einfache Parametrierung; allerdings können bei aggressivem Clipping Informationen verloren gehen.

## Wichtige Bestandteile
- OutlierCleanerTransformer

## Referenzen
- https://scikit-learn.org/stable/modules/preprocessing.html#robust-scaler

## Annahmen und Einschränkungen
Voraussetzung sind numerische Eingaben; der Rückgabewert hat den gleichen Typ wie die Eingabe.
