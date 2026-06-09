from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from build_bindingdb_mmp_matrix import (
    build_candidate_matrix_from_fragments,
    fragment_smiles_brics,
    iter_active_measurements,
    parse_bindingdb_nm,
    pactivity_from_nm,
)


def test_parse_bindingdb_nm_and_pactivity():
    assert parse_bindingdb_nm(" 100 ") == ("=", 100.0)
    assert parse_bindingdb_nm("< 1000") == ("<", 1000.0)
    assert parse_bindingdb_nm(">1000") == (">", 1000.0)
    assert parse_bindingdb_nm("") is None
    assert round(pactivity_from_nm(100.0), 3) == 7.0


def test_iter_active_measurements_keeps_unambiguous_active_values():
    row = {
        "Ki (nM)": "100",
        "IC50 (nM)": ">1000",
        "Kd (nM)": "<500",
        "EC50 (nM)": "",
    }

    measurements = list(iter_active_measurements(row, active_threshold=6.0))

    assert [m["endpoint"] for m in measurements] == ["Ki", "Kd"]
    assert measurements[0]["pactivity"] == pytest.approx(7.0)
    assert measurements[1]["relation"] == "<"


def test_fragment_smiles_brics_finds_shared_core_for_acylanilides():
    fragments = fragment_smiles_brics(
        "CC(=O)Nc1ccccc1",
        mol_id="mol_a",
        core_min_heavy=6,
        fragment_min_heavy=1,
        fragment_max_heavy=12,
    )

    assert any(
        rec["core_smiles"] == "*Nc1ccccc1" and rec["fragment_smiles"] == "*C(C)=O"
        for rec in fragments
    )


def test_build_candidate_matrix_from_fragments_labels_active_active_replacements():
    fragment_records = [
        {
            "target_key": "P_TEST",
            "target_name": "Synthetic target",
            "endpoint": "Ki",
            "core_key": "core_a",
            "core_smiles": "*Nc1ccccc1",
            "attachment_signature": "C|N",
            "fragment_smiles": "*C(C)=O",
            "canonical_smiles": "CC(=O)Nc1ccccc1",
            "reactant_set_id": "a1",
            "pactivity": 7.0,
        },
        {
            "target_key": "P_TEST",
            "target_name": "Synthetic target",
            "endpoint": "Ki",
            "core_key": "core_a",
            "core_smiles": "*Nc1ccccc1",
            "attachment_signature": "C|N",
            "fragment_smiles": "*C(=O)CC",
            "canonical_smiles": "CCC(=O)Nc1ccccc1",
            "reactant_set_id": "a2",
            "pactivity": 7.1,
        },
        {
            "target_key": "P_TEST",
            "target_name": "Synthetic target",
            "endpoint": "Ki",
            "core_key": "core_b",
            "core_smiles": "*Nc1ccncc1",
            "attachment_signature": "C|N",
            "fragment_smiles": "*C(=O)c1ccccc1",
            "canonical_smiles": "O=C(Nc1ccncc1)c1ccccc1",
            "reactant_set_id": "b1",
            "pactivity": 7.2,
        },
        {
            "target_key": "P_TEST",
            "target_name": "Synthetic target",
            "endpoint": "Ki",
            "core_key": "core_b",
            "core_smiles": "*Nc1ccncc1",
            "attachment_signature": "C|N",
            "fragment_smiles": "*C(=O)c1ccccc1F",
            "canonical_smiles": "O=C(Nc1ccncc1)c1ccc(F)cc1",
            "reactant_set_id": "b2",
            "pactivity": 7.3,
        },
    ]

    matrix, audit = build_candidate_matrix_from_fragments(
        fragment_records,
        dataset="bindingdb_test",
        min_active_fragments_per_group=2,
        negative_ratio=1,
        min_candidates_per_query=2,
        random_seed=13,
    )

    assert audit["eligible_groups"] == 2
    assert set(["query_id", "old_fragment_smiles", "candidate_smiles", "label"]).issubset(matrix.columns)

    query = matrix[matrix["old_fragment_smiles"] == "*C(C)=O"]
    assert set(query["label"]) == {0, 1}
    positives = set(query.loc[query["label"] == 1, "candidate_smiles"])
    assert positives == {"*C(=O)CC"}
