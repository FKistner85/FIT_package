# FIT-Py: reproducible publication experiments (2026)

This is a **new experiment pathway** in FIT-Py. It does **not** overwrite the legacy
sequential-holdout notebooks or retrospectively change dissertation results.
It targets a methodological article (e.g., *Ecological Informatics*) on the
new Python FIT workflow. All reported numerical results must come from actual
runs on the intended reference material; running the unit tests does not validate
published performance estimates.

## Most important methodological correction

In the legacy `pipeline_individual_id/sequential_holdout.py`, when `cutoff=None`,
`optimal_cutoff(dist_mat, true_n)` chooses the threshold using the **known number
of validation animals**. This is an *oracle-assisted* evaluation and should not be
reported as independent population-size accuracy. The original code has been
left unchanged for backward compatibility. **Do not use its default cutoff in a
publication benchmark.**

The new `paper_validation.calibrate_cutoff_from_train_only()` accepts only
inner calibration cases from an outer *training* partition. It freezes one
threshold before `evaluate_outer_population()` inspects the outer truth.
Train-only selection applies to both the distance threshold for same/different
classification and the population-clustering cutoff.

## Implementation

- `pipeline_individual_id/paper_validation.py`: group-disjoint manifests; train-only
  calibration; explicit missing-pair policy; DER, V-measure, false-merge and
  false-split metrics; pseudo-trails with no reused print IDs; OOF sex calibration;
  train-only imputation; missing-feature stress test.
- `pipeline_individual_id/paper_experiments.py`: reuses the existing repo's
  `generate_pairwise_comparisons_from_df` and `run_all_pairwise_projections_parallel`
  with `fit_df=training_reference`. Performs nested calibration, classical-FIT-like
  ellipse overlap baseline, and ablations of feature selection/distance choice.
- `scripts/run_paper_experiments.py`: reproducible CLI runner, grouped split
  manifest, config/input hash, summary files, diagnostic figures, failure log.
- `tests/test_paper_validation.py`: unit/regression tests, including invariance
  of cutoff to altered outer ground truth and correct parsing of "False" labels.

## Install / dry-run

From the repository root:

```bash
pip install -e .
pytest -q tests/test_paper_validation.py
python scripts/run_paper_experiments.py \
  --input data/cleaned/eurasian_otter.parquet \
  --species eurasian_otter \
  --features dist1,dist2,dist3,dist4 \
  --dry-run
```

The `--features` example is **illustrative only**. Use the actual numerical
morphometric feature column names of your data. With `--features` omitted, the
runner calls the repository's `get_feature_cols`. The benchmark requires **complete
known-individual input prints** at baseline, with columns `id`,
`individual_id`, `sex` and optional `trail` and `species`.

Check your real data path using `find data -iname '*otter*'`, as the repository
contains legacy backups, not necessarily the same processed inputs as the
final dissertation. Validate the sample set, identifiers and commit before running.

## Minimal full benchmark (computationally expensive)

```bash
python scripts/run_paper_experiments.py \
  --input path/to/complete_known_otter_reference.parquet \
  --output results/paper_2026/otter \
  --configs classic_forward_ellipse,forward_distance,lasso_distance,rf_distance,lasso_manhattan \
  --trail-sizes 2,4,6 \
  --missingness 0,0.1,0.25 \
  --folds 5 --repeats 3 --inner-repeats 3 --jobs 4 --seed 42
```

**Start smaller** with `--configs lasso_distance --trail-sizes 4 --missingness 0
--folds 3 --repeats 1 --inner-repeats 2 --jobs 1` to check for issues.
Then run the full protocol with consistent predetermined settings. Additional
species can be processed by the same runner when `species` is present in the
input; each is evaluated independently.

### What is evaluated?

1. Split by known **individual**, stratifying approximately by sex.
2. Within each outer training set, create repeated inner group-disjoint
   calibration splits. Each inner query becomes a pseudo-population of two
   *disjoint* trails per known animal, sampled without replacement.
3. For each method, extract pairwise distances using the *existing* FIT pipeline;
   reference-centroid training (RCV), scaling and other fitted preprocessing
   operate on inner reference animals. Pair-specific A/B feature selection and
   LDA projections are intentionally **transductive** and remain so in the outer
   test; this is not leakage of biological identity labels.
4. Choose binary classification threshold and clustering cutoff **only from
   the inner cases**, using balanced accuracy and 0.5*(1-DER)+0.5*V-measure
   respectively.
5. Freeze them; compute a pseudo-population from outer held-out animals.
   Report pair BA/AUC, false merge/split, DER, V, homogeneity, completeness,
   excluded animals, pair coverage and runtime, for each method.
6. Repeat with variable trail lengths and feature masking. For a missing-feature
   stress test, impute using **outer-train-fitted** median or forest imputer.
   Missingness here is at the **derived feature** level, not necessarily the
   anatomical landmark level. Use a separate annotated-landmark benchmark to
   study true landmark occlusion and geometric rederivation.

### Substrate transfer / domain shift

A separate, grouped transfer experiment can be run if enough animals have
prints of each substrate type:

```bash
python scripts/run_paper_experiments.py \
  --input path/to/known_otter_reference.parquet \
  --reference-substrate sand --test-substrate mud \
  --configs lasso_distance --trail-sizes 2,4
```

Grouping by individual occurs **before** substrate filtering; no animal is
present in both training and validation. If too few animals/prints are
available, the failure/eligibility logs are the result, not evidence of a
successful transfer. Within-model results should be compared to the same
eligible animal cohort whenever possible.

### Cross-species & sex calibration

For cross-species testing, use the same configuration and separate per-species
known-individual partitions. Do not pool all species into one accuracy figure.

For calibration of the sex classifier, supply genuine **out-of-fold**
probabilities with columns `individual_id`, `sex`, `p_male`,
`is_out_of_fold=True`:

```bash
python scripts/run_paper_experiments.py \
  --sex-oof path/to/sex_predictions_oof.csv \
  --output results/paper_2026/sex_calibration
```

This produces a Brier score and an individual-weighted reliability table. It
*rejects* rows flagged as in-sample. It does not infer that a saved full-data
model generated OOF predictions: the data provenance must be verified.

## Output files

- `run_manifest.json`: numeric settings, random seed, input CSV SHA-256 hash
- `split_manifest.csv`: identity-level outer group assignment
- `outer_results.csv`: **new** independent evaluation metrics, not dissertation values
- `summary.csv`: per configuration / trail size / missingness mean, SD and count
- `eligibility.csv`: animals retained/excluded at each synthetic trail length
- `failures.csv`: data/algorithm failures (never silently hide them)
- `paired_ablation_deltas.csv`: outer-fold paired differences against ellipse baseline
- `figures/`: accuracy vs. trail length, false merges and DER vs. missingness

## Remaining caveats (important for the paper)

1. A FIT-like `95%` covariance-ellipse check in Python is not established as
   numerically identical to the original JMP algorithm; call it **FIT-like**.
2. For unequal or non-Euclidean pairwise distances, **Ward** may be unsuitable.
   The default in this extension is average linkage. Report linkage sensitivity.
3. Some generated trails exclude animals with too few prints; always report
   `eligibility.csv` and use comparable eligible cohorts when comparing lengths.
4. Reference pseudo-trails do not reproduce field substrate artefacts, true
   landmark missingness, or observers' annotation error. The published 2024
   imputation study should remain cited as prior evidence rather than duplicated.
5. Split-level summaries from overlapping repeated folds are descriptive;
   do not bootstrap individual pair rows as independent observations.
6. Missing-pair policy defaults to *maximum observed distance* in the runner
   and must be reported and stress-tested; no method can infer reliable
   missing pairwise distances from nothing.
7. The old dissertation and supplementary experimental outputs **must be audited
   against the historical commit and exact run settings** before reconciling
   differences. A change in DER does not automatically invalidate the whole thesis.
8. Review metadata, licensing and sensitive geographic locations before public
   data deposition; the runner only needs morphometric tables.

## Implementation status

This extension provides code and locally tested core functions. An end-to-end
reference-data experiment is only complete once the specific dissertation
reference dataset has been supplied, all experiments have run, failure logs
have been checked, and numerical claims have been independently examined.
