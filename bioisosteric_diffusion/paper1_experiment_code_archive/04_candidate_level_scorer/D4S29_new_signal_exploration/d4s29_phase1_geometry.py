#!/usr/bin/env python3
"""
D4S29 Phase 1: Geometry Signal Exploration
Test whether 3D conformer features (shape, electrostatic, PMI)
provide orthogonal signal beyond 2D/frequency features.
Focus on hard fragment *Cc1ccccc1 where DE/HGB both fail.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdShapeHelpers
import os, time, warnings
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S29_new_signal_exploration/results"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def clean_smiles(smi):
    """Remove attachment marker * for conformer generation."""
    return smi.replace("*", "[H]")  # replace * with H for geometry

def gen_conformer(mol, n_conf=1):
    """Generate 3D conformer, return mol with Hs."""
    mh = Chem.AddHs(mol)
    ok = AllChem.EmbedMolecule(mh, randomSeed=42, maxAttempts=50)
    if ok < 0:
        return None
    AllChem.MMFFOptimizeMolecule(mh)
    return mh

def compute_geometry_features(old_smi, cand_smi):
    """Compute 3D geometry features for an old-candidate pair."""
    try:
        old_mol = Chem.MolFromSmiles(clean_smiles(old_smi))
        cand_mol = Chem.MolFromSmiles(clean_smiles(cand_smi))
        if old_mol is None or cand_mol is None:
            return None

        old_h = gen_conformer(old_mol)
        cand_h = gen_conformer(cand_mol)
        if old_h is None or cand_h is None:
            return None

        features = {}

        # Shape similarity (3D Tanimoto using RDKit ShapeHelpers)
        try:
            shape_dist = rdShapeHelpers.ShapeTanimotoDist(old_h, cand_h)
            features['shape_tanimoto'] = 1.0 - shape_dist  # distance -> similarity
        except:
            features['shape_tanimoto'] = np.nan

        # Shape protrude distance
        try:
            protrude = rdShapeHelpers.ShapeProtrudeDist(old_h, cand_h)
            features['shape_protrude'] = protrude
        except:
            features['shape_protrude'] = np.nan

        # PMI (principal moments of inertia)
        for name, fn in [('pmi1', rdMolDescriptors.CalcPMI1),
                          ('pmi2', rdMolDescriptors.CalcPMI2),
                          ('pmi3', rdMolDescriptors.CalcPMI3)]:
            try:
                features[f'{name}_old'] = fn(old_h, 0)
                features[f'{name}_cand'] = fn(cand_h, 0)
                features[f'{name}_ratio'] = (features[f'{name}_cand'] /
                                              (features[f'{name}_old'] + 1e-10) - 1.0)
            except:
                features[f'{name}_old'] = np.nan
                features[f'{name}_cand'] = np.nan
                features[f'{name}_ratio'] = np.nan

        # Normalized PMI ratios (shape descriptors)
        for name, fn in [('npr1', rdMolDescriptors.CalcNPR1),
                          ('npr2', rdMolDescriptors.CalcNPR2)]:
            try:
                features[f'{name}_old'] = fn(old_h)
                features[f'{name}_cand'] = fn(cand_h)
                features[f'{name}_diff'] = features[f'{name}_cand'] - features[f'{name}_old']
            except:
                features[f'{name}_old'] = np.nan
                features[f'{name}_cand'] = np.nan
                features[f'{name}_diff'] = np.nan

        # Electrostatic: Gasteiger charge statistics
        try:
            AllChem.ComputeGasteigerCharges(old_h)
            old_charges = np.array([a.GetDoubleProp('_GasteigerCharge') for a in old_h.GetAtoms()])
            AllChem.ComputeGasteigerCharges(cand_h)
            cand_charges = np.array([a.GetDoubleProp('_GasteigerCharge') for a in cand_h.GetAtoms()])
            features['charge_mean_old'] = old_charges.mean()
            features['charge_mean_cand'] = cand_charges.mean()
            features['charge_mean_diff'] = cand_charges.mean() - old_charges.mean()
            features['charge_max_old'] = np.abs(old_charges).max()
            features['charge_max_cand'] = np.abs(cand_charges).max()
            features['charge_max_diff'] = features['charge_max_cand'] - features['charge_max_old']
            features['charge_std_old'] = old_charges.std()
            features['charge_std_cand'] = cand_charges.std()
        except:
            for k in ['charge_mean_old','charge_mean_cand','charge_mean_diff',
                       'charge_max_old','charge_max_cand','charge_max_diff',
                       'charge_std_old','charge_std_cand']:
                features[k] = np.nan

        # Asphericity and eccentricity
        try:
            features['asphericity_old'] = rdMolDescriptors.CalcAsphericity(old_h)
            features['asphericity_cand'] = rdMolDescriptors.CalcAsphericity(cand_h)
            features['asphericity_diff'] = (features['asphericity_cand'] -
                                             features['asphericity_old'])
            features['eccentricity_old'] = rdMolDescriptors.CalcEccentricity(old_h)
            features['eccentricity_cand'] = rdMolDescriptors.CalcEccentricity(cand_h)
            features['eccentricity_diff'] = (features['eccentricity_cand'] -
                                              features['eccentricity_old'])
        except:
            for k in ['asphericity_old','asphericity_cand','asphericity_diff',
                       'eccentricity_old','eccentricity_cand','eccentricity_diff']:
                features[k] = np.nan

        return features
    except Exception as e:
        return None


def main():
    log("=" * 60)
    log("D4S29 Phase 1: Geometry Signal Exploration")
    log("=" * 60)

    # Load data
    df = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    log(f"Loaded {len(df):,} rows, {df['query_id'].nunique():,} queries")

    # Focus: *Cc1ccccc1 (hard fragment) + top-2 other fragments for comparison
    target_frag = "*Cc1ccccc1"

    # Sample queries: 20 from *Cc1ccccc1, 5 from each other fragment
    np.random.seed(42)
    sampled_queries = []
    for frag in df['old_fragment_smiles'].unique():
        frag_qs = df[df['old_fragment_smiles'] == frag]['query_id'].unique()
        n_sample = min(30 if frag == target_frag else 5, len(frag_qs))
        sampled = np.random.choice(frag_qs, n_sample, replace=False)
        sampled_queries.extend(sampled)

    log(f"Sampled {len(sampled_queries)} queries from {df['old_fragment_smiles'].nunique()} fragments")

    # Compute geometry features for each query's candidates
    all_rows = []
    n_total = 0
    n_success = 0
    t0 = time.time()

    for i, qid in enumerate(sampled_queries):
        grp = df[df['query_id'] == qid]
        old_smi = grp['old_fragment_smiles'].iloc[0]

        # Process all candidates + all positives for this query
        for _, row in grp.iterrows():
            n_total += 1
            cand_smi = row['candidate_smiles']
            label = row['label']

            geo = compute_geometry_features(old_smi, cand_smi)
            if geo is None:
                continue
            n_success += 1

            geo['query_id'] = qid
            geo['old_fragment'] = old_smi
            geo['candidate_smiles'] = cand_smi
            geo['label'] = label
            geo['blend_score'] = row['blend_score']
            geo['blend_rank'] = row['blend_rank']
            geo['cont_prior_score'] = row['cont_prior_score']
            geo['backoff_logit_score'] = row['backoff_logit_score']
            geo['_tanimoto_similarity'] = row['_tanimoto_similarity']
            all_rows.append(geo)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = n_total / (elapsed + 0.01)
            log(f"  [{i+1}/{len(sampled_queries)}] {n_success}/{n_total} success, "
                f"{rate:.0f} pairs/min")

    geo_df = pd.DataFrame(all_rows)
    log(f"\nGeometry features computed: {n_success}/{n_total} ({100*n_success/n_total:.1f}%)")
    log(f"Time: {(time.time()-t0)/60:.1f} min")
    log(f"Feature dims: {len(geo_df.columns)}")

    # Save
    geo_df.to_csv(f"{OUT}/d4s29_geometry_sample.csv", index=False)
    log(f"Saved d4s29_geometry_sample.csv ({len(geo_df)} rows)")

    # ── Analysis: does geometry separate positives from negatives? ──
    log("\n=== GEOMETRY SIGNAL ANALYSIS ===")

    numeric_geo_cols = [c for c in geo_df.columns
                        if c not in ['query_id','old_fragment','candidate_smiles','label']
                        and geo_df[c].dtype in ['float64','float32','int64']
                        and not geo_df[c].isna().all()]

    for col in sorted(numeric_geo_cols):
        pos_vals = geo_df[geo_df['label'] == 1][col].dropna()
        neg_vals = geo_df[geo_df['label'] == 0][col].dropna()
        if len(pos_vals) < 3 or len(neg_vals) < 3:
            continue
        pos_mean = pos_vals.mean()
        neg_mean = neg_vals.mean()
        diff = pos_mean - neg_mean

        # Simple t-test
        from scipy import stats
        try:
            t_stat, p_val = stats.ttest_ind(pos_vals, neg_vals)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        except:
            p_val = 1.0
            sig = ""

        if abs(diff) > 1e-6 and p_val < 0.1:
            log(f"  {col:30s} pos={pos_mean:+.4f} neg={neg_mean:+.4f} "
                f"diff={diff:+.4f} p={p_val:.4f} {sig}")

    # Within-query ranking check for *Cc1ccccc1
    log("\n=== WITHIN-QUERY RANKING: *Cc1ccccc1 ===")
    cc1 = geo_df[geo_df['old_fragment'] == target_frag]
    cc1_queries = cc1['query_id'].unique()

    shape_hits = 0
    blend_hits = 0
    n_q = 0
    for qid in cc1_queries:
        grp = cc1[cc1['query_id'] == qid]
        n_q += 1

        # Rank by shape similarity
        if grp['shape_tanimoto'].notna().sum() > 0:
            shape_ranked = grp.sort_values('shape_tanimoto', ascending=False)
            labels = shape_ranked['label'].values
            pos_ranks = np.where(labels == 1)[0] + 1
            if len(pos_ranks) > 0 and pos_ranks.min() <= 10:
                shape_hits += 1

        # Rank by blend
        blend_ranked = grp.sort_values('blend_score', ascending=False)
        labels = blend_ranked['label'].values
        pos_ranks = np.where(labels == 1)[0] + 1
        if len(pos_ranks) > 0 and pos_ranks.min() <= 10:
            blend_hits += 1

    log(f"  Queries: {n_q}")
    log(f"  Shape Top10: {shape_hits/n_q:.4f} ({shape_hits}/{n_q})")
    log(f"  Blend Top10: {blend_hits/n_q:.4f} ({blend_hits}/{n_q})")

    # Check: shape + blend complementary?
    shape_only = 0
    blend_only = 0
    both_hit = 0
    neither = 0
    for qid in cc1_queries:
        grp = cc1[cc1['query_id'] == qid]
        s_hit = False; b_hit = False
        if grp['shape_tanimoto'].notna().sum() > 0:
            s_ranked = grp.sort_values('shape_tanimoto', ascending=False)
            s_labels = s_ranked['label'].values
            s_pos = np.where(s_labels == 1)[0] + 1
            s_hit = len(s_pos) > 0 and s_pos.min() <= 10
        b_ranked = grp.sort_values('blend_score', ascending=False)
        b_labels = b_ranked['label'].values
        b_pos = np.where(b_labels == 1)[0] + 1
        b_hit = len(b_pos) > 0 and b_pos.min() <= 10
        if s_hit and b_hit: both_hit += 1
        elif s_hit: shape_only += 1
        elif b_hit: blend_only += 1
        else: neither += 1

    log(f"  Both hit: {both_hit}, Shape-only: {shape_only}, "
        f"Blend-only: {blend_only}, Neither: {neither}")

    if shape_only > 0:
        log(f"  ✓ Shape provides orthogonal signal! ({shape_only} queries shape-only)")
    elif neither > 0 and shape_only == 0:
        log(f"  ✗ Shape does NOT help where blend fails")

    log(f"\nD4S29 Phase 1 complete.")
    log(f"Next: if geometry shows orthogonal signal → expand to full val")
    log(f"      if not → move to Phase 2 (activity coordinates)")

if __name__ == "__main__":
    main()
