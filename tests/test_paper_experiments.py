"""Integration smoke tests that mock only the computationally costly FIT projection.

They validate the orchestration and holdout bookkeeping but cannot stand in for
an end-to-end execution on original known-individual reference footprints.
"""
from itertools import combinations
import json

import numpy as np
import pandas as pd

from FIT_python.pipeline_individual_id import paper_experiments as ep


def _fake_projection(query, reference, feature_cols, *, selection, k_features, n_jobs, debug=False):
    # Intentional fake score: use known identity solely to construct test fixtures.
    # The real pipeline receives no known identity during distance calculation.
    groups=query.groupby("trail")["individual_id"].first()
    out=[]
    for (name_a,id_a),(name_b,id_b) in combinations(groups.items(),2):
        same=id_a==id_b
        d=.2 if same else 1.4
        a=np.array([[-0.04,0],[0.04,.02],[-.03,.02],[0,-.02]])
        b=np.array([[.01,0],[.02,-.02],[.02,.03],[-.02,.01]])+(0 if same else .8)
        out.append({"trail_a_id":name_a,"trail_b_id":name_b,
                    "dist_euclidean":d,"dist_manhattan":d*1.1,
                    "same_individual":str(same),
                    "coords_a_x":a[:,0].tolist(),"coords_a_y":a[:,1].tolist(),
                    "coords_b_x":b[:,0].tolist(),"coords_b_y":b[:,1].tolist()})
    return pd.DataFrame(out)


def test_full_runner_writes_metadata_robustness_and_no_oracle(tmp_path,monkeypatch):
    monkeypatch.setattr(ep,"_make_pairs_and_scores",_fake_projection)
    rng=np.random.default_rng(2)
    df=pd.DataFrame({"id":[f"s{i}" for i in range(240)],
                     "individual_id":np.repeat([f"id{i:02}" for i in range(20)],12),
                     "sex":np.repeat(["M","F"]*10,12),
                     "f1":rng.normal(size=240),
                     "f2":rng.normal(size=240),
                     "f3":rng.normal(size=240),
                     "f4":rng.normal(size=240)})
    results=ep.run_reference_benchmark(
        df,["f1","f2","f3","f4"],tmp_path,
        configs=[ep.ABLATION_CONFIGS[1],ep.ABLATION_CONFIGS[2]],
        n_splits=3,outer_repeats=1,inner_repeats=2,
        trail_sizes=(2,4),missingness_rates=(0,.2),
        k_features=3,n_jobs=1)
    assert len(results)==3*2*2*2
    assert results["pop_der"].ge(0).all()
    assert results["pop_cutoff"].eq(results["cluster_cutoff"]).all()
    assert (tmp_path/"split_manifest.csv").exists()
    assert (tmp_path/"summary.csv").exists()
    assert (tmp_path/"eligibility.csv").exists()
    assert (tmp_path/"run_manifest.json").exists()
    assert (tmp_path/"outer_results.csv").exists()
    provenance=json.loads((tmp_path/"run_manifest.json").read_text())
    assert provenance["n_splits"]==3
    assert len(provenance["input_sha256"])==64
    assert results["pair_balanced_accuracy"].eq(1).all()
