# feature_selection_wrapper.py

## Überblick
Merkmalsselektion per Forward, Random Forest, Varianzfilter, univariat oder LASSO.
Die Klasse bietet unterschiedliche Strategien zur Auswahl relevanter Merkmale. Dadurch lassen sich lineare und nichtlineare Zusammenhänge untersuchen. Vorteilhaft ist die automatische Bewertung, Nachteil sind längere Laufzeiten bei komplexen Auswahlverfahren.

## Wichtige Bestandteile
- FeatureSelectionTransformer

## Referenzen
- https://scikit-learn.org/stable/modules/feature_selection.html

## Annahmen und Einschränkungen
Die Länge des Rankings wird durch k gesteuert; None wählt alle Features aus.
