# Pairwise Individual Identification Pipeline

This document summarises the code found in `src/FIT_python/pipeline_individual_id` which implements the pairwise trail comparison workflow.  The goal is to evaluate whether two trails originate from the same individual.

## 1. Generating Trail Pairs

The function `generate_pairwise_comparisons_from_df` in `generate_trails_and_trailpairs.py` creates all necessary pairings.  Its docstring outlines the steps:

```
1) Erzeuge `group_id` aus `id_col` bzw. `fallback_col`.
2) Erzeuge `trails_per_animal` entweder aus vorgegebenen Pools oder durch
   diverse Teilmengen-Sampling mittels Jaccard-Distanz.
3) Baue alle Cross- und Within-Individual-Paare.
4) `same_individual` und `same_sex` markieren Gleichheit oder "unknown".
5) Weist jede Person per `StratifiedKFold` (stratifiziert nach Geschlecht)
   einem Fold zu und behält nur Paare aus demselben Fold.
6) Summary-Tabelle mit pro-Länge und Total-Zeile inkl. avg/sd Pair counts.
```
【F:src/FIT_python/pipeline_individual_id/generate_trails_and_trailpairs.py†L209-L218】

Each generated pair dictionary includes the indexes of both trails, unique trail IDs and metadata such as `same_individual`, `same_sex`, trail sizes and the assigned fold.  A summary table records the number of animals and trails per length. Individuals are stratified by sex into folds and only comparisons from the same fold are retained.【F:src/FIT_python/pipeline_individual_id/generate_trails_and_trailpairs.py†L153-L161】

## 2. Distance Metrics

`distance_metrics.py` defines `compute_distances`, which returns several distances between two vectors. Supported metrics include Euclidean, Manhattan, cosine, Chebyshev, Canberra and Bray‑Curtis.

```python
    distances = {
        "euclidean": euclidean(a, b),
        "manhattan": cityblock(a, b),
        "cosine": cosine(a, b),
        "chebyshev": chebyshev(a, b),
        "canberra": canberra(a, b),
        "braycurtis": braycurtis(a, b),
    }
```
【F:src/FIT_python/pipeline_individual_id/distance_metrics.py†L12-L19】

## 3. Pairwise Projection Workflow

The main computation takes place in `run_all_pairwise_projections_parallel` within `pairwise_individual_id_pipeline.py` (starting at line 40).  The docstring summarises the processing sequence:

```
0) Falls use_sexmodel_prediction aktiv ist, das Sex-Modell laden und predict_proba vorberechnen.
1) Basis-DF bereinigen
2) Pipeline-Schritte: Outlier-Cleaning & Feature-Scaling
3) Feature-Selection (einmal mit k_max)
4) RCV-Set als Komplement
5) Sex-probas extrahieren & mitteln
6) Für jede Kombi (outlier, scaler, reducer, n_components, k):
   - apply LDA/PCA/UMAP
   - Abstände berechnen
   - Result-Dict inkl. avg_proba_A/B/R 0/1
```
【F:src/FIT_python/pipeline_individual_id/pairwise_individual_id_pipeline.py†L55-L69】

After optionally loading a pre-trained sex classifier, the function converts the selected feature columns to numeric values (lines 85‑88) and precomputes `predict_proba` values if requested (lines 90‑94).  For each pair the pipeline iterates over outlier cleaning and scaling choices, applies forward feature selection, and then reduces the dimension with either LDA, PCA or UMAP.  The results include the projected coordinates of both trails and of an RCV reference set, as well as multiple distance metrics computed between the trail centroids.

Distances between all individual points are also summarised to provide mean and median values both between and within the two trails.  This happens at lines 324‑357 of the same file.

## 4. Parallel Execution

The projection of each pair is handled by `process_pair`, and the top-level function parallelises these calls using `joblib.Parallel` with a `tqdm` progress bar.  After processing all batches the nested lists are flattened into a single result list.【F:src/FIT_python/pipeline_individual_id/pairwise_individual_id_pipeline.py†L363-L370】

## 5. Output Structure

Each result dictionary returned by `run_all_pairwise_projections_parallel` contains the trail IDs, lists of `id` values for both trails, individual identifiers, fold number and whether the trails belong to the same individual.  It also includes the chosen preprocessing options (`pipeline`), the selected features with their scores, the computed distances and—if a sex classifier is used—the average predicted probabilities for each group.  Coordinates for each trail and the RCV set are stored under `coords_*` fields, along with their centroids.

The collected results can later be written to a CSV file for statistical analysis or model training.

For a description of the helper transformers (outlier cleaning, scaling, feature selection and dimensionality reduction) see the markdown files in `docs/pipeline_individual_id/`.
