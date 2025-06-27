# feature_selection_wrapper.py

Dieses Modul bietet die Funktion _forward_ranking und die Klasse FeatureSelectionTransformer.
Ersteres wählt Merkmale iterativ anhand von Varianzanalysen aus und erstellt ein Ranking.
Der Transformer kann entweder auf dieses Verfahren zurückgreifen oder die Wichtigkeiten eines Random‑Forest‑Klassifikators verwenden.
Nach dem Fit liegt eine geordnete Liste der Merkmale vor, aus der transform die gewünschte Teilmenge extrahiert.
