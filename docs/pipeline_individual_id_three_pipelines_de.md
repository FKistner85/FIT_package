# Individuen-ID: Überblick über drei Pipelines

Dieses Dokument fasst die Notizen in `pipeline_individual_id_3_pipelines.txt` zusammen. Es beschreibt den geplanten Ablauf zum Trainieren und Validieren dreier Ansätze zur Identifikation einzelner Tiere.

## 1. Datenvorbereitung
1. **Daten laden**
   - Nutze den Train/Test-Split der Geschlechtspipeline über den vorhandenen `SplitWrapper`.
   - Der Testsatz dient nur zur abschließenden Bewertung.
   - Bei der Art `Lutra lutra` wird `create_train_test_split_otter` aufgerufen, andernfalls der generische Splitter.
2. **Splits zusammenfassen**
   - Mit `compute_summary` die Größen von Train-, Test- und Inferenzsatz prüfen.
3. **Morphometrische Merkmale wählen**
   - `morph_feature_cols` enthält alle Spalten, die mit `dist`, `ang`, `t` oder `v` beginnen und numerisch sind.
   - Die Namen der ausgewählten Spalten werden ausgegeben.
4. **Merkmale skalieren**
   - Der Skalierer wird auf dem Trainingssatz fit und auf alle Sätze angewendet.
   - Optional entsteht eine Korrelationsgrafik der skalierten Merkmale.
5. **Vorhersagen des Sex-Modells**
   - `predict_all('<species>', reuse_csv=False)` erzeugt einmalig
     `{species}_all_predictions.csv` im Verzeichnis
     `results/data/random_search_standard_metrics`. Spätere Aufrufe
     können `reuse_csv=True` verwenden, um die Datei erneut zu laden.
   - Für den Trainingssatz werden die bereits gespeicherten
     Kreuzvalidierungs‑Vorhersagen verwendet, um nur Out‑of‑Fold‑Werte zu
     nutzen.
   - Diese werden als zusätzliche Merkmale `sex_features` gespeichert.
   - Die Individual-ID-Pipelines laden diese Dateien automatisch, wenn
     `use_sexmodel_prediction=True` gesetzt ist.

## 2. Trainingseinrichtung
1. **Individuen auswählen**
   - Liste aller `individual_id` im Trainingssatz erstellen.
   - Optional eine Teilmenge ziehen (Standard: 10 Individuen).
2. **Sequentielle Holdouts**
   - Anzahl der Wiederholungen festlegen (Standard: 1, später etwa 10).
   - Für jede Wiederholung Validierungssätze mit wachsender Individuenzahl erstellen, z. B. `[2,4,6,8]`.
   - Ein Individuum befindet sich jeweils komplett im Train- oder Validierungssatz.
3. **Pro-Wiederholung-Dictionary**
   - Zuordnung der Train-/Val-IDs für jede Iteration speichern.

## 3. Pipelines
Für jeden Train-/Validation-Split werden folgende Pipelines durchlaufen.

### Pipeline 1 – Distanz-Baseline
1. `individual_id` und `trail` im Trainingssatz durch `rcv` ersetzen, um eine Referenzmenge zu bilden.
2. Es werden ausschließlich Distanzen berechnet; es wird kein Modell auf den Testdaten trainiert.
3. Standardmäßig werden die 16 besten Merkmale aus `morph_feature_cols` verwendet.

### Pipeline 2 – Supervised UMAP
1. `y_train = train_set["individual_id"]`.
2. Falls aktiviert, `OutlierCleanerTransformer` anwenden.
3. `FeatureSelectionTransformer` auf die Merkmale ansetzen.
4. Optional `sex_features` anhängen und ebenfalls selektieren.
5. Dimension mit einem überwachtem Verfahren (Standard: UMAP) reduzieren; Embeddings und euklidische Distanzen zurückgeben.
6. Aussagekräftige Distanzmerkmale auswählen und einen Klassifikator trainieren.
   Fehlende Distanzwerte werden vor dem Training verworfen, damit der
   Logit-Klassifikator keine ungültigen Eingaben erhält.

### Pipeline 3 – Siamese Network
1. `y_train = train_set["individual_id"]`.
2. Optional Outlier-Bereinigung und Merkmalsauswahl wie oben.
3. `sex_features` anhängen, sofern nicht deaktiviert.
4. Ein Siamese-Netzwerk mit Triplet-Loss trainieren, das innerhalb einer Identität kleine und zwischen Identitäten große Distanzen erzeugt.
5. Optional Feature-Selektion auf dem erlernten Embedding.
6. Einen Klassifikator auf den abgeleiteten Distanzen trainieren.

## 4. Validierungsablauf
1. **Trails erzeugen**
   - Jeden Validierungssatz in Trails der Länge `trail_size` (Standard 10) aufteilen.
   - Individuen mit zu wenigen Spuren überspringen.
2. **Subsampling**
   - Für jeden Trail zufällige Subtrails der Größen `[7,5,3,2]` bilden.
   - Alle Trails und Subtrails paaren, Paare desselben Trails werden verworfen.
3. **Metadaten je Paar**
   - `ind_a`, `ind_b`, `trail_a_id`, `trail_b_id`, `same_individual`, `same_sex` (oder `unknown`), Listen der Indexe, Trailgrößen und die absolute Differenz speichern.

## 5. Paarverarbeitung
Der Ablauf für ein einzelnes Paar sieht verkürzt so aus:

```python
df_base = df.set_index('id')
for comparison in comparisons:
    df_a = df_base.loc[comparison['samples_a']]
    df_b = df_base.loc[comparison['samples_b']]
    df_r = df_train_rcv
    y_ab = [0] * len(df_a) + [1] * len(df_b)
    for out_method in outs:
        steps = []
        if out_method:
            steps.append(('outlier', OutlierCleanerTransformer(method=out_method)))
        selector = FeatureSelectionTransformer(method=selection_method, k=k_max)
        steps.append(('select', selector))
        pipe = Pipeline(steps)
        pipe.fit(pd.concat([df_a, df_b]), y_ab)
        df_a_fs, df_b_fs, df_r_fs = pipe.transform(df_a), pipe.transform(df_b), pipe.transform(df_r)
        # ggf. sex features ergänzen und reduzierer ausprobieren
```

Nach der Distanzberechnung kann `compute_overlap_jsl_style` prüfen, ob sich die Ellipsen überlappen. Dabei steuert das Argument ``p`` (Standard ``0.5``) den Chi-Quadrat-Radius.

## 6. Auswertung
1. Für jede Pipeline werden Konfusionsmatrizen erstellt.
2. Die Anzahl übersprungener Validierungsschritte (z. B. fehlende Trails) wird vermerkt.
3. Abschließend werden Durchschnittswerte und Summen der verarbeiteten Trails und Paare gemeldet.

## Ausführung des Trainingsskripts
`train_individual_id_pipelines.py` führt diesen Ablauf von der Kommandozeile aus. Gereinigte Daten und die Splits der Geschlechtspipeline müssen gemäß `src/FIT_python/config.py` vorliegen. Für jedes Holdout werden alle drei Pipelines trainiert und die Ergebnisse gespeichert.

### Benötigte Eingaben
1. Bereinigte Spurentabellen in `data/cleaned/`.
2. Train- und Testpartitionen in `data/splits/`.

### Kommandozeilenargumente
```bash
python train_individual_id_pipelines.py [options]
```
- `--iterations N` – Anzahl der Holdout-Wiederholungen (Standard 1).
- `--include-sex`/`--no-include-sex` – Vorhersagen des Sex-Modells anhängen (Standard aktiviert).
- `--debug` – Ausführliche Meldungen und Abbruch bei fehlenden Dateien.

### Ausgabe
Alle Artefakte liegen unter `results/data/individual_id`. Jede Iteration erhält einen eigenen Unterordner mit Modellen und CSV-Übersichten. Grafiken wie Konfusionsmatrizen landen in `results/figures/individual_id`.
