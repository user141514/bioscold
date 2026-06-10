from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from content_calibration import (
    CFC3_VARIANT_CONFIGS,
    add_cf_c3_context_columns,
    ca_bucket_edges,
    ca_content_score,
    fit_cf_c3,
    predict_cf_c3,
    select_ca_matched_rows,
)


def _row(qid, candidate, label, morgan, target_key="T_KEEP", attach="C|N", endpoint="IC50"):
    return {
        "dataset": "cf_c3_test",
        "query_id": qid,
        "old_fragment_smiles": f"OF_{qid}",
        "candidate_smiles": candidate,
        "label": label,
        "morgan": morgan,
        "bit_corr": 0.5 * morgan,
        "dHeavy": 1.0 - morgan,
        "dRings": 0.0,
        "dMW": 1.0 - morgan,
        "dLogP": 1.0 - morgan,
        "dTPSA": 1.0 - morgan,
        "target_key": target_key,
        "attachment_signature": attach,
        "endpoint": endpoint,
        "core_key": f"core_{qid}",
    }


def test_ca_matched_selector_keeps_all_positives_and_nearest_negatives():
    df = pd.DataFrame(
        [
            _row("q1", "POS_A", 1, 0.70),
            _row("q1", "NEG_NEAR_A", 0, 0.69),
            _row("q1", "NEG_NEAR_B", 0, 0.73),
            _row("q1", "NEG_FAR", 0, 0.10),
            _row("q2", "POS_B", 1, 0.30),
            _row("q2", "NEG_NEAR_C", 0, 0.31),
            _row("q2", "NEG_FAR_B", 0, 0.90),
        ]
    )

    selected = select_ca_matched_rows(df, negatives_per_positive=1)

    assert set(selected.loc[selected["label"] == 1, "candidate_smiles"]) == {"POS_A", "POS_B"}
    assert "NEG_NEAR_A" in set(selected["candidate_smiles"])
    assert "NEG_NEAR_C" in set(selected["candidate_smiles"])
    assert "NEG_FAR" not in set(selected["candidate_smiles"])
    assert "NEG_FAR_B" not in set(selected["candidate_smiles"])


def test_cf_c3_feature_builder_does_not_use_candidate_query_or_frequency():
    rows = []
    for idx in range(6):
        qid = f"q{idx}"
        rows.extend(
            [
                _row(qid, f"CANDIDATE_POS_{idx}", 1, 0.75, target_key="T_A" if idx < 4 else "T_B"),
                _row(qid, f"CANDIDATE_NEG_{idx}", 0, 0.72, target_key="T_A" if idx < 4 else "T_B"),
                _row(qid, f"CANDIDATE_FAR_{idx}", 0, 0.10, target_key="T_A" if idx < 4 else "T_B"),
            ]
        )
    df = pd.DataFrame(rows)

    model = fit_cf_c3(df, c_value=0.1, negatives_per_positive=1, target_min_rows=2, target_min_positive=1)

    feature_text = " ".join(model.feature_names).lower()
    assert "candidate" not in feature_text
    assert "query" not in feature_text
    assert "freq" not in feature_text
    assert any("attachment_signature" in name for name in model.feature_names)
    assert model.training_rows == 12


def test_unseen_contexts_map_to_other_or_ignore_safely():
    train = pd.DataFrame(
        [
            _row("q1", "POS_A", 1, 0.80, target_key="T_A", attach="C|N", endpoint="IC50"),
            _row("q1", "NEG_A", 0, 0.78, target_key="T_A", attach="C|N", endpoint="IC50"),
            _row("q2", "POS_B", 1, 0.75, target_key="T_A", attach="C|N", endpoint="IC50"),
            _row("q2", "NEG_B", 0, 0.20, target_key="T_A", attach="C|N", endpoint="IC50"),
        ]
    )
    test = pd.DataFrame(
        [
            _row("q3", "POS_UNSEEN", 1, 0.76, target_key="T_UNSEEN", attach="N|S", endpoint="Ki"),
            _row("q3", "NEG_UNSEEN", 0, 0.21, target_key="T_UNSEEN", attach="N|S", endpoint="Ki"),
        ]
    )

    model = fit_cf_c3(train, c_value=0.1, negatives_per_positive=1, target_min_rows=2, target_min_positive=1)
    scores = predict_cf_c3(model, test)

    assert len(scores) == len(test)
    assert np.isfinite(scores).all()


def test_ca_bucket_assignment_is_train_edge_based():
    train = pd.DataFrame([_row(f"q{i}", f"C{i}", int(i % 2 == 0), 0.1 + i * 0.1) for i in range(10)])
    edges = ca_bucket_edges(ca_content_score(train), n_buckets=5)
    assigned, kept = add_cf_c3_context_columns(train, edges, target_min_rows=1000, target_min_positive=1000)

    assert assigned["ca_bucket"].nunique() <= 5
    assert set(assigned["target_context"]) == {"__OTHER__"}
    assert kept == set()


def test_cf_c3_variant_feature_names_match_ablation_policy():
    rows = []
    for idx in range(8):
        rows.extend(
            [
                _row(f"q{idx}", f"POS_{idx}", 1, 0.82, target_key="T_KEEP", attach="A|B", endpoint="IC50"),
                _row(f"q{idx}", f"NEG_NEAR_{idx}", 0, 0.79, target_key="T_KEEP", attach="A|B", endpoint="IC50"),
                _row(f"q{idx}", f"NEG_FAR_{idx}", 0, 0.15, target_key="T_DROP", attach="C|D", endpoint="Ki"),
            ]
        )
    train = pd.DataFrame(rows)

    full = fit_cf_c3(
        train,
        c_value=0.1,
        negatives_per_positive=1,
        target_min_rows=2,
        target_min_positive=1,
        config=CFC3_VARIANT_CONFIGS["cf_c3"],
    )
    no_ca_bucket = fit_cf_c3(
        train,
        c_value=0.1,
        negatives_per_positive=1,
        target_min_rows=2,
        target_min_positive=1,
        config=CFC3_VARIANT_CONFIGS["cf_c3_no_ca_bucket"],
    )
    no_context = fit_cf_c3(
        train,
        c_value=0.1,
        negatives_per_positive=1,
        target_min_rows=2,
        target_min_positive=1,
        config=CFC3_VARIANT_CONFIGS["cf_c3_no_context"],
    )
    no_target = fit_cf_c3(
        train,
        c_value=0.1,
        negatives_per_positive=1,
        target_min_rows=2,
        target_min_positive=1,
        config=CFC3_VARIANT_CONFIGS["cf_c3_no_target"],
    )

    assert any("ca_bucket" in name for name in full.feature_names)
    assert any("attachment_signature" in name for name in full.feature_names)
    assert any("target_context" in name for name in full.feature_names)
    assert not any("ca_bucket" in name for name in no_ca_bucket.feature_names)
    assert any("attachment_signature" in name for name in no_ca_bucket.feature_names)
    assert no_context.feature_names == [
        "morgan",
        "bit_corr",
        "dHeavy",
        "dRings",
        "dMW",
        "dLogP",
        "dTPSA",
    ]
    assert not any("target_context" in name for name in no_target.feature_names)
    assert any("ca_bucket" in name for name in no_target.feature_names)
    assert any("attachment_signature" in name for name in no_target.feature_names)


def test_no_ca_match_uses_more_training_rows_than_full_cf_c3():
    rows = []
    for idx in range(6):
        rows.extend(
            [
                _row(f"q{idx}", f"POS_{idx}", 1, 0.80),
                _row(f"q{idx}", f"NEG_NEAR_{idx}", 0, 0.79),
                _row(f"q{idx}", f"NEG_FAR_A_{idx}", 0, 0.20),
                _row(f"q{idx}", f"NEG_FAR_B_{idx}", 0, 0.10),
            ]
        )
    train = pd.DataFrame(rows)

    full = fit_cf_c3(
        train,
        c_value=0.1,
        negatives_per_positive=1,
        config=CFC3_VARIANT_CONFIGS["cf_c3"],
    )
    no_ca_match = fit_cf_c3(
        train,
        c_value=0.1,
        negatives_per_positive=1,
        config=CFC3_VARIANT_CONFIGS["cf_c3_no_ca_match"],
    )

    assert full.training_rows == 12
    assert no_ca_match.training_rows == len(train)
    assert no_ca_match.training_rows > full.training_rows
