# split_utils.py

## Überblick
Hilfsfunktionen für stratifizierte und gruppenbasierte Train/Test-Aufteilungen.
Diese Funktionen erleichtern reproduzierbare Daten-Splits, bei denen Individuen und Klassen ausgewogen vertreten sind. Sie sind hilfreich, wenn mehrere Messungen pro Individuum vorhanden sind. Der Aufwand steigt jedoch mit der Komplexität der Gruppenstruktur.

## Wichtige Bestandteile
- stratified_individual_split
- train_test_group_split
- group_stratified_kfold
- ensure_valid_splits

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

## Annahmen und Einschränkungen
Erfordert Gruppen- und Stratifizierungsspalten und sorgt für ausgewogene Folds.

### stratified_individual_split
Teilt ein DataFrame nach Individuum auf und erhält dabei das Verhältnis der
Geschlechter in ``stratify_col`` zwischen Train und Test. Bei gesetztem
``add_folds=True`` enthält der Train-Split zusätzlich eine ``Fold``-Spalte aus
``StratifiedGroupKFold``. Dadurch landen alle Spuren eines Tieres im selben Fold
und das Geschlechterverhältnis bleibt stabil.
