from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_clustering import assign_candidate_clusters, ensure_candidate_cluster_column


def test_candidate_clustering_is_deterministic_under_input_order_changes():
    smiles = ["CCO", "CCN", "c1ccccc1", "c1ccncc1", "CCCl"]
    forward = assign_candidate_clusters(smiles, threshold=0.60)
    reversed_map = assign_candidate_clusters(list(reversed(smiles)), threshold=0.60)

    assert forward == reversed_map
    assert set(forward) == set(smiles)
    assert all(value.startswith("cluster_") for value in forward.values())


def test_existing_candidate_cluster_column_is_reused():
    df = pd.DataFrame(
        {
            "candidate_smiles": ["not_a_smiles", "also_not_a_smiles"],
            "candidate_cluster": ["manual_A", "manual_B"],
        }
    )

    out = ensure_candidate_cluster_column(df, threshold=0.60)

    assert list(out["candidate_cluster"]) == ["manual_A", "manual_B"]


def test_missing_candidate_cluster_requires_valid_rdkit_smiles():
    df = pd.DataFrame({"candidate_smiles": ["not_a_smiles"]})

    with pytest.raises(ValueError, match="Could not parse candidate SMILES"):
        ensure_candidate_cluster_column(df, threshold=0.60)
