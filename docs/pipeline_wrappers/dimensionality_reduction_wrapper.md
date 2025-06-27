# dimensionality_reduction_wrapper.py

Dieses Modul enthält die Klasse DimensionalityReducerTransformer. Sie kapselt eine Reduktion der Dimensionalität mittels Principal Component Analysis.
Beim Initialisieren wählt man die gewünschte Zahl an Hauptkomponenten.
Die fit-Methode passt diese Zahl an die maximal sinnvolle Größe an und trainiert anschließend die PCA.
Mit transform wird die Projektion auf neue Daten angewandt.
Über get_feature_names_out lassen sich die Namen der erzeugten Komponenten abrufen.
