"""Automated publication experiments for FIT-Py.

Requires original FIT-python package for the full reference-data runner.
All hyperparameters and population cutoffs are selected from individuals
outside the outer evaluation fold. Outputs are descriptive and not published
results until run on the intended reference datasets and audited.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .paper_validation import (
    PopulationCase, apply_missingness, calibrate_cutoff_from_train_only,
    choose_binary_distance_cutoff, cluster_with_cutoff, evaluate_outer_population,
    evaluate_pair_classifier, evaluate_sex_oof_calibration, impute_from_train_only,
    make_grouped_split_manifest, make_pseudo_trails, pairwise_scores_to_distances,
    parse_same_individual,
)

# Methods are accepted by the existing FeatureSelectionTransformer.
ABLATION_CONFIGS = (
    {"config": "classic_forward_ellipse", "selection": "forward", "metric": "dist_euclidean", "decision": "ellipse"},
    {"config": "forward_distance", "selection": "forward", "metric": "dist_euclidean", "decision": "distance"},
    {"config": "lasso_distance", "selection": "lasso", "metric": "dist_euclidean", "decision": "distance"},
    {"config": "rf_distance", "selection": "random_forest", "metric": "dist_euclidean", "decision": "distance"},
    {"config": "lasso_manhattan", "selection": "lasso", "metric": "dist_manhattan", "decision": "distance"},
)


def _make_pairs_and_scores(
    pseudo_query: pd.DataFrame, reference: pd.DataFrame,
    feature_cols: Sequence[str], *, selection: str,
    k_features: int, n_jobs: int, debug: bool = False,
) -> pd.DataFrame:
    """Re-use the repository's original pairwise FIT projection and RCV code."""
    from .generate_trails_and_trailpairs import generate_pairwise_comparisons_from_df
    from .pairwise_individual_id_pipeline import run_all_pairwise_projections_parallel
    comps, _ = generate_pairwise_comparisons_from_df(
        pseudo_query, strategy="select", trail_col="trail", evaluation=True
    )
    if not comps:
        raise ValueError("No pairs generated from pseudo-population")
    # All printed IDs must be distinct between training and pseudo-query sets.
    if set(reference["id"]) & set(pseudo_query["id"]):
        raise AssertionError("Reference and held-out query footprints overlap")
    concat = pd.concat([reference, pseudo_query], ignore_index=True)
    results = run_all_pairwise_projections_parallel(
        comps, concat, fit_df=reference, feature_cols=list(feature_cols),
        selection_method=selection, k_features=k_features,
        reducers=["lda"], n_components=2, n_jobs=n_jobs,
        outlier_methods=[None], scaler_methods=["standard"],
        use_sexmodel_prediction=False, debug=debug,
    )
    scored = pd.DataFrame(results)
    if scored.empty:
        raise ValueError("No computable pairwise projections")
    # If the package returns multiple variants, fail rather than silently mix them.
    if scored[["trail_a_id", "trail_b_id"]].duplicated().any():
        raise ValueError("Multiple score rows per trail-pair; specify one configuration")
    return scored


def _truth_for_trails(query: pd.DataFrame) -> pd.Series:
    if query.groupby("trail")["individual_id"].nunique().gt(1).any():
        raise ValueError("Mixed known identities within one trail")
    return query.groupby("trail")["individual_id"].first().astype(str)


def _score_case(scored: pd.DataFrame, query: pd.DataFrame, metric: str, *,
                case_id: str) -> PopulationCase:
    truth = _truth_for_trails(query)
    matrix = pairwise_scores_to_distances(scored, value=metric,
                                          trails=list(truth.index))
    return PopulationCase(matrix, truth, case_id)


def _inner_calibration(
    outer_train: pd.DataFrame, feature_cols: Sequence[str], *,
    selection: str, metric: str, trail_size: int, k_features: int,
    repeats: int, seed: int, n_jobs: int,
) -> tuple[list[PopulationCase], pd.DataFrame, pd.DataFrame]:
    cases: list[PopulationCase] = []
    pair_rows = []
    eligibility_rows = []
    splitter = GroupShuffleSplit(n_splits=repeats, test_size=0.3, random_state=seed)
    groups = outer_train["individual_id"]
    for inner_no, (ref_idx, query_idx) in enumerate(splitter.split(outer_train, groups=groups)):
        ref_raw, query_raw = outer_train.iloc[ref_idx], outer_train.iloc[query_idx]
        try:
            pseudo, elig = make_pseudo_trails(query_raw, prints_per_trail=trail_size,
                                              random_state=seed + inner_no)
        except ValueError:
            continue
        if pseudo["individual_id"].nunique() < 2:
            continue
        elig["inner_fold"] = inner_no
        eligibility_rows.append(elig)
        try:
            scored = _make_pairs_and_scores(pseudo, ref_raw, feature_cols,
                                             selection=selection, k_features=k_features,
                                             n_jobs=n_jobs)
            cases.append(_score_case(scored, pseudo, metric,
                                     case_id=f"inner_{inner_no}"))
            scored["inner_fold"] = inner_no
            pair_rows.append(scored)
        except ValueError:
            continue
    if not cases:
        raise ValueError("No usable inner population calibration folds; lower trail_size")
    return (cases, pd.concat(pair_rows, ignore_index=True),
            pd.concat(eligibility_rows, ignore_index=True))


def _binary_performance(scored: pd.DataFrame, threshold: float, config: dict) -> dict:
    if config["decision"] == "distance":
        return evaluate_pair_classifier(scored, threshold,
                                        distance_col=config["metric"])
    from .evaluation import compute_overlap_jsl_style_vec
    from sklearn.metrics import balanced_accuracy_score
    valid = scored.dropna(subset=["same_individual"])
    pred = compute_overlap_jsl_style_vec(valid, p=0.95)
    y = parse_same_individual(valid["same_individual"])
    return {"balanced_accuracy": float(balanced_accuracy_score(y,pred)),
            "false_merge_rate":float(np.mean(pred[~y])) if (~y).any() else float("nan"),
            "false_split_rate":float(np.mean(~pred[y])) if y.any() else float("nan"),
            "auc":float("nan"), "n_pairs":len(valid),
            "pair_coverage":len(valid)/len(scored)}


def run_reference_benchmark(
    footprints: pd.DataFrame,
    feature_cols: Sequence[str],
    out_dir: Path,
    *,
    configs: Sequence[dict] = ABLATION_CONFIGS,
    n_splits: int = 5,
    outer_repeats: int = 1,
    inner_repeats: int = 3,
    trail_sizes: Sequence[int] = (4,),
    missingness_rates: Sequence[float] = (0.0,),
    seed: int = 42,
    k_features: int = 12,
    n_jobs: int = 1,
    linkage_method: str = "average",
    missing_policy: str = "max_observed",
    imputer: str = "median",
    reference_substrate: str | None = None,
    test_substrate: str | None = None,
) -> pd.DataFrame:
    """Run a grouped, nested, reproducible methods/robustness study.

    Inputs must contain complete numeric FIT features for reference individuals
    in the zero-missingness condition. All comparisons are drawn from held-out
    animals, while RCV uses reference animals only.

    For missingness > 0, we deliberately simulate missing *features*, not
    anatomically missing landmarks. This is a partial robustness experiment;
    report it as such unless explicit landmark-feature group mappings are used.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (reference_substrate is None) != (test_substrate is None):
        raise ValueError("Provide both reference_substrate and test_substrate")
    if reference_substrate is not None and "substrate" not in footprints:
        raise KeyError("Substrate-shift evaluation requires a substrate column")
    needed = {"id", "individual_id", "sex"} | set(feature_cols)
    if not needed.issubset(footprints.columns):
        raise KeyError(f"Missing input columns: {sorted(needed.difference(footprints.columns))}")
    if footprints["id"].duplicated().any():
        raise ValueError("Duplicate footprint identifiers")
    if footprints[feature_cols].isna().any().any():
        raise ValueError("The initial benchmark requires complete features; choose a complete reference subset")

    manifest = make_grouped_split_manifest(footprints, n_splits=n_splits,
                                           repeats=outer_repeats, seed=seed)
    manifest.to_csv(out_dir/"split_manifest.csv", index=False)
    config_snapshot = {"configs":list(configs), "trail_sizes":list(trail_sizes),
                       "missingness_rates":list(missingness_rates), "seed":seed,
                       "n_splits":n_splits, "outer_repeats":outer_repeats,
                       "inner_repeats":inner_repeats, "k_features":k_features,
                       "linkage_method":linkage_method, "missing_policy":missing_policy,
                       "imputer":imputer, "n_jobs":n_jobs,
                       "reference_substrate":reference_substrate,
                       "test_substrate":test_substrate,
                       "input_sha256":sha256(footprints.to_csv(index=False).encode()).hexdigest()}
    (out_dir/"run_manifest.json").write_text(json.dumps(config_snapshot,indent=2),encoding="utf8")
    all_results = []
    eligibilities = []
    failures = []
    for (rep,fold), selected in manifest.groupby(["repeat","fold"],sort=True):
        test_ids = set(selected.loc[selected["partition"]=="test","individual_id"].astype(str))
        train_raw = footprints.loc[~footprints["individual_id"].astype(str).isin(test_ids)]
        test_raw = footprints.loc[footprints["individual_id"].astype(str).isin(test_ids)]
        if reference_substrate is not None:
            train_raw = train_raw.loc[train_raw["substrate"].astype(str)==reference_substrate]
            test_raw = test_raw.loc[test_raw["substrate"].astype(str)==test_substrate]
        for trail_size in trail_sizes:
            try:
                test_pseudo, elig = make_pseudo_trails(test_raw,
                    prints_per_trail=trail_size, random_state=seed+rep*1000+fold*100+trail_size)
                if test_pseudo["individual_id"].nunique()<2:
                    raise ValueError("Fewer than two eligible outer test individuals")
                elig["repeat"],elig["fold"],elig["trail_size"] = rep,fold,trail_size
                eligibilities.append(elig)
            except ValueError as exc:
                failures.append({"repeat":rep,"fold":fold,"trail_size":trail_size,"reason":str(exc)})
                continue
            for config in configs:
                started = perf_counter()
                try:
                    cases, inner_pairs, _ = _inner_calibration(train_raw,feature_cols,
                        selection=config["selection"],metric=config["metric"],
                        trail_size=trail_size,k_features=k_features,
                        repeats=inner_repeats,seed=seed+1000*rep+fold*100,
                        n_jobs=n_jobs)
                    # No test labels have been examined to choose either threshold.
                    calibration = calibrate_cutoff_from_train_only(cases,
                        method=linkage_method, missing_policy=missing_policy)
                    training_cut = choose_binary_distance_cutoff(inner_pairs,
                        distance_col=config["metric"])
                    for rate in missingness_rates:
                        if rate == 0:
                            ref,test = train_raw,test_pseudo
                        else:
                            masked = apply_missingness(test_pseudo,features=feature_cols,
                                fraction=rate,strategy="mcar",
                                random_state=seed+rep*10000+fold*1000+int(rate*1000))
                            ref,test = impute_from_train_only(train_raw,masked,feature_cols,
                                                               strategy=imputer,seed=seed)
                        scored = _make_pairs_and_scores(test,ref,feature_cols,
                                   selection=config["selection"],k_features=k_features,
                                   n_jobs=n_jobs)
                        p = _binary_performance(scored,training_cut,config)
                        case = _score_case(scored,test,config["metric"],case_id="outer")
                        clust = evaluate_outer_population(case.distances,case.identities,
                                                          calibration)
                        all_results.append({"repeat":rep,"fold":fold,"trail_size":trail_size,
                            "missingness_fraction":rate,"configuration":config["config"],
                            "n_test_individuals":test["individual_id"].nunique(),
                            "n_inner_cases":len(cases),"pair_cutoff":training_cut,
                            "cluster_cutoff":calibration.cutoff,
                            "runtime_seconds":perf_counter()-started,
                            **{f"pair_{k}":v for k,v in p.items()},
                            **{f"pop_{k}":v for k,v in clust.items()}})
                except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                    failures.append({"repeat":rep,"fold":fold,"trail_size":trail_size,
                                     "configuration":config["config"],"reason":str(exc)})
    results = pd.DataFrame(all_results)
    results.to_csv(out_dir/"outer_results.csv",index=False)
    pd.DataFrame(failures).to_csv(out_dir/"failures.csv",index=False)
    if eligibilities:
        pd.concat(eligibilities,ignore_index=True).to_csv(out_dir/"eligibility.csv",index=False)
    if not results.empty:
        group = results.groupby(["configuration","trail_size","missingness_fraction"])
        keys = ["pair_balanced_accuracy","pair_auc","pair_false_merge_rate",
                "pop_der","pop_v_measure","pop_false_merge_rate","pair_pair_coverage"]
        summary = group[keys].agg(["mean","std","count"])
        summary.to_csv(out_dir/"summary.csv")
    return results


def run_sex_reliability_file(prediction_path: Path, out_dir: Path) -> dict:
    """Analyse OOF sex probabilities and save a reliability CSV."""
    source = pd.read_parquet(prediction_path) if prediction_path.suffix==".parquet" else pd.read_csv(prediction_path)
    metrics, tab = evaluate_sex_oof_calibration(source)
    out_dir.mkdir(parents=True,exist_ok=True)
    tab.to_csv(out_dir/"sex_reliability_bins.csv",index=False)
    (out_dir/"sex_reliability_metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics
