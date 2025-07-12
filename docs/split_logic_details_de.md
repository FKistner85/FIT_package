# Logik des Train/Test-Splits

Dieses Dokument beschreibt das Aufteilen der Daten, wie es in `split_wrapper.py` und den Hilfsfunktionen in `split_utils.py` umgesetzt ist.

## 1. Art-spezifische Splits
`SplitWrapper.split_all()` lädt bereinigte Datensätze und legt unter `data/splits` pro Art einen Ordner mit den Dateien `train.parquet`, `test.parquet` und optional `inference.parquet` an. Alle Spuren eines Individuums erscheinen **nur in einem** dieser Splits. Dafür sorgt `stratified_individual_split`:

1. Prüft, ob `individual_id` und `sex` vorhanden sind.
2. Erstellt eine Meta-Tabelle mit dem Geschlecht jedes Individuums.
3. Wendet `train_test_split` auf die Individuen an und stratifiziert nach Geschlecht.
4. Weist die Zeilen der entsprechenden Train- und Test-Individuen zu. Fehlende Sex-Angaben landen im `inference`-Set.

So bleiben die Fußspuren eines Tiers zusammen und die Geschlechterverteilung in Train und Test ist ausgeglichen.

## 2. Cross-Validation (optional)
Wenn Folds benötigt werden, weist `group_stratified_kfold` eine `Fold`-Spalte zu. Hierbei sorgt `StratifiedGroupKFold` dafür, dass

- ein Individuum nie in mehreren Folds vorkommt und
- jedes Fold eine ähnliche Geschlechterverteilung aufweist.

Der Schritt wird über `_make_folds` in `SplitWrapper.split_all()` ausgeführt.

## 3. Zusammenfassung
Für jeden erzeugten Split wird eine kurze Übersicht ausgegeben: Anzahl der Zeilen, der Individuen und der Verteilung nach Geschlecht. So lässt sich leicht prüfen, ob die Stratifikation funktioniert hat.
