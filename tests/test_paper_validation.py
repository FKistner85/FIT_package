"""Unit/regression tests for leak-free publication validation."""
import numpy as np
import pandas as pd
import pytest

from FIT_python.pipeline_individual_id.paper_validation import (
    PopulationCase, apply_missingness, calibrate_cutoff_from_train_only,
    choose_binary_distance_cutoff, cluster_with_cutoff, evaluate_outer_population,
    evaluate_pair_classifier, evaluate_sex_oof_calibration, impute_from_train_only,
    make_grouped_split_manifest, make_pseudo_trails, pairwise_scores_to_distances,
    partition_metrics, prepare_distance_matrix, parse_same_individual,
)


def _d(ids, within=0.1, between=2.0):
    labels = np.asarray(ids)
    d = np.where(labels[:,None]==labels[None,:], within, between).astype(float)
    np.fill_diagonal(d,0)
    names = [f"t{i}" for i in range(len(ids))]
    return pd.DataFrame(d,index=names,columns=names),pd.Series(ids,index=names)


def test_calibration_does_not_use_outer_test_labels():
    inner_d, inner_labels = _d(["a","a","b","b"], within=0.13, between=1.5)
    calibration = calibrate_cutoff_from_train_only([PopulationCase(inner_d,inner_labels)])
    outer_d, outer_labels = _d(["c","c","d","d"], within=.14, between=1.7)
    first = evaluate_outer_population(outer_d,outer_labels,calibration)
    # Change *only* the outer identities. Evaluation metrics can change,
    # but cut-off and predictions may not be refitted.
    altered = outer_labels.copy()
    altered.iloc[1] = "e"
    second = evaluate_outer_population(outer_d,altered,calibration)
    assert first["cutoff"] == second["cutoff"] == calibration.cutoff
    assert first["der"] != second["der"]
    cluster_a,_ = cluster_with_cutoff(outer_d,calibration.cutoff)
    cluster_b,_ = cluster_with_cutoff(outer_d,calibration.cutoff)
    pd.testing.assert_series_equal(cluster_a,cluster_b)


def test_multi_case_calibration_and_metrics():
    a,ids_a=_d([1,1,2,2],within=.2,between=1)
    b,ids_b=_d([3,3,4,4],within=.1,between=2)
    cal=calibrate_cutoff_from_train_only([PopulationCase(a,ids_a),PopulationCase(b,ids_b)])
    assert 0 < cal.cutoff < 3
    assert cal.n_cases == 2
    metrics=evaluate_outer_population(a,ids_a,cal)
    assert metrics["true_n"] == 2
    assert 0 <= metrics["v_measure"] <=1
    assert "false_merge_rate" in metrics


def test_missing_distance_handling_explicit():
    d,_=_d([1,1,2,2])
    d.loc["t0","t3"]=np.nan
    d.loc["t3","t0"]=np.nan
    with pytest.raises(ValueError,match="Missing pairwise"):
        prepare_distance_matrix(d)
    filled,missing=prepare_distance_matrix(d,missing_policy="max_observed")
    assert missing == pytest.approx(1/6)
    assert filled.loc["t0","t3"] == 2
    assert filled.loc["t3","t0"] == 2


def test_paired_distances_duplicate_conflict():
    x = pd.DataFrame({"trail_a_id":["a","b"],"trail_b_id":["b","a"],
                      "dist_euclidean":[.4,.5]})
    with pytest.raises(ValueError,match="Contradictory"):
        pairwise_scores_to_distances(x)
    x.loc[1,"dist_euclidean"] = .4
    d=pairwise_scores_to_distances(x)
    assert d.loc["a","b"] == .4
    assert d.loc["a","a"] == 0


def test_grouped_manifest_disjoint_and_repeated():
    df = pd.DataFrame({"individual_id":np.repeat([f"id{i}" for i in range(20)],3),
                       "sex":np.repeat(["F","M"]*10,3)})
    m=make_grouped_split_manifest(df,n_splits=5,repeats=2)
    assert len(m)==20*5*2
    for (repeat,fold), grp in m.groupby(["repeat","fold"]):
        assert grp["individual_id"].nunique()==20
        assert grp.loc[grp.partition=="train","individual_id"].nunique()==16
        assert grp.loc[grp.partition=="test","individual_id"].nunique()==4
        assert not set(grp.loc[grp.partition=="train","individual_id"]) & set(grp.loc[grp.partition=="test","individual_id"])


def test_short_trails_disjoint_and_eligibility():
    df=pd.DataFrame({"id":[f"s{i}" for i in range(17)],
                     "individual_id":["a"]*10+["b"]*6+["c"]})
    pseudo,report=make_pseudo_trails(df,prints_per_trail=3,random_state=4)
    assert len(pseudo)==12
    assert pseudo["id"].nunique()==len(pseudo)
    assert pseudo.groupby("trail").size().eq(3).all()
    assert report.set_index("individual_id").loc["c","eligible"] == False


def test_train_only_imputer_not_fit_to_test():
    train=pd.DataFrame({"x":[0.,0.,0.,10.],"y":[1.,2.,3.,4.]})
    test=pd.DataFrame({"x":[np.nan,1e8],"y":[5.,6.]})
    _, a=impute_from_train_only(train,test,["x","y"],strategy="median")
    test2=test.copy(); test2.loc[1,"x"]=-1e8
    _, b=impute_from_train_only(train,test2,["x","y"],strategy="median")
    assert a.loc[0,"x"]==b.loc[0,"x"]==0


def test_missingness_mcar_repeatable_and_block():
    df=pd.DataFrame(np.ones((10,4)),columns=list("abcd"))
    a=apply_missingness(df,features=list("abcd"),fraction=.2,random_state=1)
    b=apply_missingness(df,features=list("abcd"),fraction=.2,random_state=1)
    pd.testing.assert_frame_equal(a,b)
    blocks=apply_missingness(df,features=list("abcd"),fraction=1,
                             strategy="block",feature_groups={"toe":["a","b"]},random_state=1)
    assert blocks[["a","b"]].isna().all().all()
    assert blocks[["c","d"]].notna().all().all()


def test_calibration_rejects_invalid_cases():
    d,ids=_d(["a","a","b","b"])
    with pytest.raises(ValueError):
        calibrate_cutoff_from_train_only([])
    with pytest.raises(ValueError):
        calibrate_cutoff_from_train_only([PopulationCase(d,ids.iloc[:2])])


def test_pair_classifier_requires_train_tuning():
    train=pd.DataFrame({"dist_euclidean":[.1,.2,1.5,2.],
                        "same_individual":[True,True,False,False]})
    cut=choose_binary_distance_cutoff(train)
    score=evaluate_pair_classifier(train,cut)
    assert score["balanced_accuracy"]==1
    assert score["false_merge_rate"]==0


def test_oof_sex_calibration_rejects_in_sample_and_averages_by_individual():
    x=pd.DataFrame({"individual_id":["a","a","b","b"],
                    "sex":["M","M","F","F"],
                    "p_male":[.9,.8,.1,.2],
                    "is_out_of_fold":[True]*4})
    m,tab=evaluate_sex_oof_calibration(x)
    assert m["n_individuals"]==2
    assert m["brier"]<.1
    x.loc[0,"is_out_of_fold"]=False
    with pytest.raises(ValueError,match="out-of-fold"):
        evaluate_sex_oof_calibration(x)


def test_partition_metrics_reflect_false_merge():
    d,t=_d(["a","a","b","b"])
    m=partition_metrics(pd.Series([1,1,1,1],index=t.index),t)
    assert m["false_merge_rate"]==1
    assert m["false_split_rate"]==0
    assert m["der"]==.5


def test_string_false_is_not_treated_as_true():
    a = pd.Series(["True", "False", "false", "1", "0"])
    assert parse_same_individual(a).tolist() == [True, False, False, True, False]
    data=pd.DataFrame({"dist_euclidean":[.1,1.5],"same_individual":["True","False"]})
    cutoff=choose_binary_distance_cutoff(data)
    assert evaluate_pair_classifier(data,cutoff)["balanced_accuracy"]==1
