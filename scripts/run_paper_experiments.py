#!/usr/bin/env python3
"""Reproduce newly proposed FIT-Py experiments (not the old dissertation figures).

Run from the repository root after pip install -e .; see docs/paper_validation.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from FIT_python.pipeline_individual_id.paper_experiments import (
    ABLATION_CONFIGS, run_reference_benchmark, run_sex_reliability_file,
)
from FIT_python.pipeline_individual_id.paper_validation import (
    make_grouped_split_manifest,
)


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Expected .csv or .parquet source data")


def _get_features(df: pd.DataFrame, features: str | None) -> list[str]:
    if features:
        requested = [s.strip() for s in features.split(",") if s.strip()]
        missing = sorted(set(requested).difference(df.columns))
        if missing:
            raise KeyError(f"Unknown feature columns: {missing}")
        return requested
    from FIT_python.data_split_and_summary.data_import_utils import get_feature_cols
    return list(get_feature_cols(df))


def _make_figures(results: pd.DataFrame, target: Path) -> None:
    if results.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target.mkdir(parents=True, exist_ok=True)
    for xcol, ycol, filename, ylabel in [
        ("trail_size", "pair_balanced_accuracy", "accuracy_by_trail_size.png", "Balanced accuracy"),
        ("missingness_fraction", "pair_false_merge_rate", "false_merges_by_missingness.png", "False-merge rate"),
        ("missingness_fraction", "pop_der", "population_der_by_missingness.png", "Relative cluster-count error"),
    ]:
        if xcol not in results or ycol not in results:
            continue
        fig, ax = plt.subplots(figsize=(7,4))
        for name, subset in results.groupby("configuration"):
            dat = subset.groupby(xcol)[ycol].agg(["mean","std","count"]).reset_index().sort_values(xcol)
            if dat.empty:
                continue
            se = dat["std"].fillna(0) / dat["count"].pow(.5)
            ax.errorbar(dat[xcol],dat["mean"],yerr=se,marker="o",capsize=3,label=name)
        ax.set(xlabel=xcol.replace("_"," "),ylabel=ylabel)
        ax.legend(fontsize=7,loc="best")
        fig.tight_layout()
        fig.savefig(target/filename,dpi=190)
        plt.close(fig)


def _paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    """Paired differences at the *outer-fold* level (not pair-row bootstrap)."""
    index=["repeat","fold","trail_size","missingness_fraction"]
    metrics=["pair_balanced_accuracy","pop_der","pop_v_measure"]
    if results.empty or not set(index + metrics).issubset(results.columns):
        return pd.DataFrame()
    wide=results.pivot_table(index=index,columns="configuration",values=metrics)
    reference="classic_forward_ellipse"
    if reference not in wide.columns.get_level_values(1):
        return pd.DataFrame()
    rows=[]
    for config in sorted(results["configuration"].unique()):
        if config==reference:
            continue
        for metric in metrics:
            reference_series=wide[(metric,reference)]
            comparison=wide[(metric,config)]
            deltas=(comparison-reference_series).dropna()
            if len(deltas):
                rows.append({"configuration":config,"reference":reference,"metric":metric,
                             "mean_delta":float(deltas.mean()),"sd_delta":float(deltas.std()),
                             "outer_folds_paired":len(deltas)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,help="Known-identity reference CSV or Parquet")
    p.add_argument("--output",type=Path,default=Path("results/paper_2026"))
    p.add_argument("--features",help="Explicit comma-separated morphometric columns (recommended)")
    p.add_argument("--species",help="Filter dataset by species label; omit for all species")
    p.add_argument("--sex-oof",type=Path,help="Optional independent out-of-fold sex-predictions file")
    p.add_argument("--configs",default="lasso_distance",help="Comma-separated configuration names, or 'all'")
    p.add_argument("--trail-sizes",default="4",help="Comma-separated synthetic trail lengths")
    p.add_argument("--missingness",default="0",help="Comma-separated feature-missingness fractions")
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--inner-repeats",type=int,default=3)
    p.add_argument("--jobs",type=int,default=1)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--k-features",type=int,default=12)
    p.add_argument("--reference-substrate",help="Optional substrate used for all training/reference prints")
    p.add_argument("--test-substrate",help="Optional substrate of the held-out pseudo-trails")
    p.add_argument("--imputer",choices=["median","forest"],default="median")
    p.add_argument("--linkage",choices=["average","complete","single","ward"],default="average")
    p.add_argument("--missing-pairs",choices=["error","max_observed"],default="max_observed")
    p.add_argument("--dry-run",action="store_true",help="Validate data and save grouped manifest; no expensive fits")
    p.add_argument("--summarize",type=Path,help="Regenerate comparisons and figures from a results CSV")
    args=p.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)

    if args.summarize:
        results=_load_table(args.summarize)
        _paired_deltas(results).to_csv(args.output/"paired_ablation_deltas.csv",index=False)
        _make_figures(results,args.output/"figures")
        print(f"Saved summary figures to {args.output}")
        return 0

    if args.sex_oof:
        print("OOF calibration:",json.dumps(run_sex_reliability_file(args.sex_oof,args.output)))
    if args.input is None:
        if args.sex_oof:
            return 0
        p.error("--input is required unless --sex-oof or --summarize is provided")
    data=_load_table(args.input)
    if "Trail" in data and "trail" not in data:
        data=data.rename(columns={"Trail":"trail"})
    if args.species:
        if "species" not in data:
            p.error("--species requested but no species column")
        data=data.loc[data["species"].astype(str).str.lower()==args.species.lower()].copy()
    if data.empty:
        p.error("No matching data")
    cols=_get_features(data,args.features)
    if args.dry_run:
        manifest=make_grouped_split_manifest(data,n_splits=args.folds,
                                              repeats=args.repeats,seed=args.seed)
        manifest.to_csv(args.output/"split_manifest.csv",index=False)
        print(f"{data['individual_id'].nunique()} individuals; {len(data)} prints; {len(cols)} features; wrote manifest")
        return 0
    configs=ABLATION_CONFIGS if args.configs=="all" else [
        next(c for c in ABLATION_CONFIGS if c["config"]==item.strip())
        for item in args.configs.split(",")]
    trail_sizes=[int(v) for v in args.trail_sizes.split(",")]
    rates=[float(v) for v in args.missingness.split(",")]
    if "species" in data and not args.species and data["species"].nunique()>1:
        for species,subset in data.groupby("species"):
            out=args.output / str(species).replace("/","_")
            run_reference_benchmark(subset,cols,out,configs=configs,
                n_splits=args.folds,outer_repeats=args.repeats,
                inner_repeats=args.inner_repeats,trail_sizes=trail_sizes,
                missingness_rates=rates,seed=args.seed,k_features=args.k_features,
                n_jobs=args.jobs,linkage_method=args.linkage,
                missing_policy=args.missing_pairs,imputer=args.imputer,
                reference_substrate=args.reference_substrate,
                test_substrate=args.test_substrate)
            results=_load_table(out/"outer_results.csv")
            _paired_deltas(results).to_csv(out/"paired_ablation_deltas.csv",index=False)
            _make_figures(results,out/"figures")
    else:
        results=run_reference_benchmark(data,cols,args.output,configs=configs,
            n_splits=args.folds,outer_repeats=args.repeats,
            inner_repeats=args.inner_repeats,trail_sizes=trail_sizes,
            missingness_rates=rates,seed=args.seed,k_features=args.k_features,
            n_jobs=args.jobs,linkage_method=args.linkage,
            missing_policy=args.missing_pairs,imputer=args.imputer,
            reference_substrate=args.reference_substrate,
            test_substrate=args.test_substrate)
        _paired_deltas(results).to_csv(args.output/"paired_ablation_deltas.csv",index=False)
        _make_figures(results,args.output/"figures")
    print(f"Completed benchmark. Results and failures are in {args.output}")
    return 0


if __name__=="__main__":
    sys.exit(main())
