# split_utils.py

## Überblick
Hilfsfunktionen für stratifizierte und gruppenbasierte Train/Test-Aufteilungen.

## Wichtige Bestandteile
- stratified_individual_split
- train_test_group_split
- group_stratified_kfold
- ensure_valid_splits

## Referenzen
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

## Annahmen und Einschränkungen
Requires group and stratify columns; ensures balanced folds.
