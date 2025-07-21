# Individual ID: Three-Pipeline Overview

This document translates the rough notes in `pipeline_individual_id_3_pipelines.txt` into a clearer pseudocode description. The goal is to outline the planned workflow for training and validating three different approaches to individual identification.

## 1. Data Preparation
1. **Load Data**
   - Reuse the train/test split from the sex classification pipeline via the existing `SplitWrapper`.
   - The test set is held back for final evaluation only.
   - If the species column equals `"Lutra lutra"`, call `create_train_test_split_otter`; otherwise use the generic splitter.
2. **Summarise Splits**
   - Call `compute_summary` to confirm the sizes of train, test and inference sets.
3. **Select Morphometric Features**
   - `morph_feature_cols = [c for c in df.columns if c.startswith(("dist","ang","t","v")) and c != "trail" and df_train[c].dtype.kind in "if"]`
   - Print the chosen column names.
4. **Scale Features**
   - For every holdout split fit a `StandardScaler` on the training data and transform both the training and validation sets.
   - Optionally create a correlation heatmap from the scaled training values.
5. **Sex Model Predictions**
   - Run `predict_all('<species>', reuse_csv=False)` once after training to
     generate `{species}_all_predictions.csv` under `results/data/`. Future
     executions can pass `reuse_csv=True` to reuse this CSV.
   - For the training split the pre-computed cross-validation predictions are
     used so that the model only sees out-of-fold values.
   - Store these predictions as additional features `sex_features`.

## 2. Training Setup
1. **Select Individuals**
   - Build a list of unique `individual_id` values present in the training data.
   - Optionally sample a subset (default: 10 unique individuals) for quicker experiments.
2. **Sequential Holdouts**
   - Define the number of iterations (default: 1, later ~10).
   - For each iteration, create validation sets with increasing numbers of unique individuals e.g. `[2,4,6,8]`.
   - An individual must appear entirely in either the train or validation split of a given iteration.
   - The helper `sequential_holdout_ids` implements this logic and returns a list
     of dictionaries with `train_ids` and `val_ids` for each step.
3. **Per‑Iteration Dictionaries**
   - Store the mapping of train/val individual IDs for every iteration to reproduce splits.

## 3. Pipelines
For each generated train/validation split run the following pipelines. All components should be modular and allow hyperparameter tuning.

### Pipeline 1 – Distance Baseline
1. Replace `individual_id` and `trail` in the training set with the identifier `"rcv"` to form a reference set (RCV).
2. The pipeline uses only distance computations; no model is fitted on the test data.
3. Default: select the top 16 features from `morph_feature_cols`.

### Pipeline 2 – Supervised UMAP
1. `y_train = train_set["individual_id"]`.
2. Apply `OutlierCleanerTransformer` if enabled.
3. Run `FeatureSelectionTransformer` on morphometric features.
4. Optionally append `sex_features` and perform feature selection on them.
5. Reduce dimensionality using a supervised method (default: UMAP); return embeddings and Euclidean distances.
6. Select informative distance features and train a classifier to decide whether a pair stems from the same individual.

### Pipeline 3 – Siamese Network
1. `y_train = train_set["individual_id"]`.
2. Optionally clean outliers and select morphometric features as above.
3. Append `sex_features` unless disabled.
4. Train a siamese network with a triplet loss to learn an embedding that minimises within‑individual distances while maximising between‑individual distances.
5. Optionally perform feature selection on the learned embedding.
6. Train a classifier on pairwise distances derived from the embedding.

## 4. Validation Workflow
1. **Generate Trails**
   - Split each validation set into trails of length `trail_size` (default: 10).
   - Skip individuals with fewer samples than `trail_size` and warn if this occurs.
2. **Subsampling**
   - For each trail create random sub‑trails of sizes `[7,5,3,2]`.
   - Pair all trails and sub‑trails, dropping pairs originating from the same trail to avoid leakage.
3. **Metadata for Each Pair**
   - Store `ind_a`, `ind_b`, `trail_a_id`, `trail_b_id`, boolean `same_individual`, boolean or `"unknown"` `same_sex`, lists of sample indices, trail sizes and the absolute trail size difference.

## 5. Pair Processing
The following pseudocode outlines the common steps used to process a single pair during validation.

```python
df_base = df.set_index('id')
for each comparison in comparisons:
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
        if use_sexmodel_prediction:
            append sex features before dimensionality reduction
        for reducer in reducers:
            supervised = reducer in ('lda',)
            for nc in n_components:
                for k in k_features:
                    sel_feats = selector.feature_ranking_[:k]
                    # dimensionality reduction and distance computation
                    # store results including centroid coords and distances
```
After distances are computed you may run `compute_overlap_jsl_style` on each
result row to check whether the two ellipses overlap.  The function accepts a
probability ``p`` (default ``0.5``) used for the chi-square radius estimate.

## 6. Evaluation
1. For each pipeline compute confusion matrices on the validation pairs.
2. Record the number of skipped validation steps (e.g. not enough trails or missing classes).
3. After processing all iterations, report the average number of trails per individual and the total number of trail pairs evaluated.

This structured pseudocode should serve as a starting point for implementing the actual code in a modular fashion.

## Running the training script

The helper script `train_individual_id_pipelines.py` executes this workflow from
the command line. It assumes that cleaned data and the train/test splits created
for the sex models are present in the locations configured in
`src/FIT_python/config.py`. For each sequential holdout definition the script
trains all three pipelines and stores the evaluation results.

### Required inputs

1. Cleaned footprint tables in `data/cleaned/`.
2. Train and test partitions under `data/splits/`.

### Command-line arguments

```
python train_individual_id_pipelines.py [options]
```

- `--iterations N` &ndash; number of sequential holdout repetitions (default: `1`).
- `--include-sex` / `--no-include-sex` &ndash; whether to append predicted sex
  probabilities as extra features (default: enabled).
- `--debug` &ndash; print verbose progress information and stop on missing files.

### Output

All artefacts are placed in `results/data/individual_id`. Each iteration gets its
own subdirectory with the trained models and CSV summaries. Figures such as
confusion matrices are written to `results/figures/individual_id`.
