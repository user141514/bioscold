#!/usr/bin/env python3
"""Deterministic candidate-fragment clustering for held-out diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_rdkit():
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem
        from rdkit.ML.Cluster import Butina
    except Exception as exc:  # pragma: no cover - depends on optional RDKit env
        raise ValueError(
            "RDKit is required to compute candidate clusters when the matrix lacks a candidate_cluster column."
        ) from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem, DataStructs, AllChem, Butina


def assign_candidate_clusters(smiles_list: list[str] | pd.Series, threshold: float = 0.60) -> dict[str, str]:
    """Assign stable Butina cluster ids keyed by the original candidate SMILES."""

    Chem, DataStructs, AllChem, Butina = _load_rdkit()
    unique_smiles = sorted(set(str(x) for x in smiles_list))
    canonical_by_original: dict[str, str] = {}
    fp_by_canonical = {}

    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse candidate SMILES for clustering: {smiles}")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        canonical_by_original[smiles] = canonical
        if canonical not in fp_by_canonical:
            fp_by_canonical[canonical] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

    canonical_smiles = sorted(fp_by_canonical)
    if not canonical_smiles:
        return {}
    if len(canonical_smiles) == 1:
        return {smiles: "cluster_000000" for smiles in unique_smiles}

    fps = [fp_by_canonical[smiles] for smiles in canonical_smiles]
    distances = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        distances.extend([1.0 - float(sim) for sim in sims])

    clusters = Butina.ClusterData(distances, len(fps), 1.0 - float(threshold), isDistData=True)
    cluster_members = [tuple(sorted(canonical_smiles[idx] for idx in cluster)) for cluster in clusters]
    cluster_members.sort(key=lambda members: (members[0], len(members), members))

    canonical_to_cluster = {}
    for cluster_idx, members in enumerate(cluster_members):
        cluster_id = f"cluster_{cluster_idx:06d}"
        for canonical in members:
            canonical_to_cluster[canonical] = cluster_id

    return {smiles: canonical_to_cluster[canonical_by_original[smiles]] for smiles in unique_smiles}


def ensure_candidate_cluster_column(df: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
    """Return a copy with candidate_cluster, reusing an existing column when present."""

    if "candidate_cluster" in df.columns:
        out = df.copy()
        out["candidate_cluster"] = out["candidate_cluster"].astype(str)
        return out
    if "candidate_smiles" not in df.columns:
        raise ValueError("candidate_smiles is required to compute candidate clusters.")
    mapping = assign_candidate_clusters(df["candidate_smiles"], threshold=threshold)
    out = df.copy()
    out["candidate_cluster"] = out["candidate_smiles"].astype(str).map(mapping)
    if out["candidate_cluster"].isna().any():
        raise ValueError("Failed to assign candidate_cluster for every row.")
    return out
