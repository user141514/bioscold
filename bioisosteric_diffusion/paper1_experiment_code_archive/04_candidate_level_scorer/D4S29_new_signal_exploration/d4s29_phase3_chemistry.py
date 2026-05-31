#!/usr/bin/env python3
"""
D4S29 Phase 3: Chemical Context Signal Exploration
Explore attachment bond chemistry, local environment, and
replacement-induced chemical changes as orthogonal signals.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Fragments
import os, time, warnings
from collections import Counter
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S29_new_signal_exploration/results"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def get_attachment_info(smiles_with_star):
    """Parse attachment point: bond type, ring membership, degree."""
    mol = Chem.MolFromSmiles(smiles_with_star)
    if mol is None:
        return {}
    info = {}
    star_idx = None
    neighbor_idx = None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # dummy atom *
            star_idx = atom.GetIdx()
            for nbr in atom.GetNeighbors():
                neighbor_idx = nbr.GetIdx()
            break
    if star_idx is None or neighbor_idx is None:
        return {'attach_bond_type': 'unknown', 'attach_in_ring': 0}
    bond = mol.GetBondBetweenAtoms(star_idx, neighbor_idx)
    info['attach_bond_type'] = str(bond.GetBondType()) if bond else 'unknown'
    info['attach_bond_is_aromatic'] = int(bond.GetIsAromatic()) if bond else 0
    nb = mol.GetAtomWithIdx(neighbor_idx)
    info['attach_atom_in_ring'] = int(nb.IsInRing())
    info['attach_atom_degree'] = nb.GetDegree()
    info['attach_atom_aromatic'] = int(nb.GetIsAromatic())
    info['attach_atom_hybrid'] = str(nb.GetHybridization())
    info['attach_atom_num_H'] = nb.GetTotalNumHs()
    return info

def get_local_environment_fp(smiles_with_star, radius=2):
    """Morgan fingerprint centered on attachment neighbor atom."""
    mol = Chem.MolFromSmiles(smiles_with_star)
    if mol is None:
        return None
    star_idx = None
    neighbor_idx = None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            star_idx = atom.GetIdx()
            for nbr in atom.GetNeighbors():
                neighbor_idx = nbr.GetIdx()
            break
    if neighbor_idx is None:
        return None
    # Generate fingerprint centered on attachment neighbor
    env = AllChem.GetMorganFingerprint(mol, radius, fromAtoms=[neighbor_idx])
    return np.array(env)

def get_fragment_features(old_smi, cand_smi):
    """Chemical context features: ring changes, functional groups, etc."""
    old = Chem.MolFromSmiles(old_smi.replace('*', '[H]'))
    cand = Chem.MolFromSmiles(cand_smi.replace('*', '[H]'))
    if old is None or cand is None:
        return {}

    feats = {}

    # Ring changes
    old_rings = rdMolDescriptors.CalcNumRings(old)
    cand_rings = rdMolDescriptors.CalcNumRings(cand)
    feats['ring_count_change'] = cand_rings - old_rings

    old_arom = rdMolDescriptors.CalcNumAromaticRings(old)
    cand_arom = rdMolDescriptors.CalcNumAromaticRings(cand)
    feats['arom_ring_change'] = cand_arom - old_arom

    # Rotatable bond change
    old_rot = rdMolDescriptors.CalcNumRotatableBonds(old)
    cand_rot = rdMolDescriptors.CalcNumRotatableBonds(cand)
    feats['rot_bond_change'] = cand_rot - old_rot

    # H-bond donor/acceptor changes
    old_hbd = rdMolDescriptors.CalcNumHBD(old)
    cand_hbd = rdMolDescriptors.CalcNumHBD(cand)
    feats['hbd_change'] = cand_hbd - old_hbd

    old_hba = rdMolDescriptors.CalcNumHBA(old)
    cand_hba = rdMolDescriptors.CalcNumHBA(cand)
    feats['hba_change'] = cand_hba - old_hba

    # Fraction of sp3 carbons
    old_sp3 = rdMolDescriptors.CalcFractionCSP3(old)
    cand_sp3 = rdMolDescriptors.CalcFractionCSP3(cand)
    feats['fsp3_change'] = cand_sp3 - old_sp3

    # Heteroatom ratio
    old_het = sum(1 for a in old.GetAtoms() if a.GetAtomicNum() not in [1, 6])
    cand_het = sum(1 for a in cand.GetAtoms() if a.GetAtomicNum() not in [1, 6])
    old_heavy = old.GetNumHeavyAtoms()
    cand_heavy = cand.GetNumHeavyAtoms()
    feats['hetero_ratio_old'] = old_het / max(1, old_heavy)
    feats['hetero_ratio_cand'] = cand_het / max(1, cand_heavy)
    feats['hetero_ratio_diff'] = feats['hetero_ratio_cand'] - feats['hetero_ratio_old']

    # Heavy atom count ratio
    feats['heavy_ratio'] = cand_heavy / max(1, old_heavy)

    # Common functional group presence changes
    fg_names = ['fr_Al_COO', 'fr_Al_OH', 'fr_Ar_N', 'fr_Ar_OH', 'fr_COO',
                'fr_C_O', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
                'fr_amide', 'fr_aniline', 'fr_ester', 'fr_ether',
                'fr_halogen', 'fr_ketone', 'fr_nitrile', 'fr_phenol',
                'fr_priamide', 'fr_sulfide', 'fr_sulfone', 'fr_thiazole']
    for fg in fg_names:
        try:
            old_fg = getattr(Fragments, fg)(old)
            cand_fg = getattr(Fragments, fg)(cand)
            feats[f'{fg}_change'] = cand_fg - old_fg
        except:
            pass

    # Amide bond count (important for drug-likeness)
    try:
        feats['amide_bonds_old'] = rdMolDescriptors.CalcNumAmideBonds(old)
        feats['amide_bonds_cand'] = rdMolDescriptors.CalcNumAmideBonds(cand)
    except:
        pass

    # TPSA change
    old_tpsa = rdMolDescriptors.CalcTPSA(old)
    cand_tpsa = rdMolDescriptors.CalcTPSA(cand)
    feats['tpsa_change'] = cand_tpsa - old_tpsa

    # LogP crude change (Wildman-Crippen)
    old_logp = rdMolDescriptors.CalcCrippenDescriptors(old)[0]
    cand_logp = rdMolDescriptors.CalcCrippenDescriptors(cand)[0]
    feats['logp_change'] = cand_logp - old_logp

    return feats


def main():
    log("=" * 60)
    log("D4S29 Phase 3: Chemical Context Signal Exploration")
    log("=" * 60)

    df = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    log(f"Loaded {len(df):,} rows, {df['query_id'].nunique():,} queries")

    # Focus: same 65 sampled queries from Phase 1
    np.random.seed(42)
    target_frag = "*Cc1ccccc1"
    sampled_queries = []
    for frag in df['old_fragment_smiles'].unique():
        frag_qs = df[df['old_fragment_smiles'] == frag]['query_id'].unique()
        n_sample = min(30 if frag == target_frag else 5, len(frag_qs))
        sampled = np.random.choice(frag_qs, n_sample, replace=False)
        sampled_queries.extend(sampled)

    # Feature columns to collect
    attach_feat_keys = ['attach_bond_type', 'attach_bond_is_aromatic',
                         'attach_atom_in_ring', 'attach_atom_degree',
                         'attach_atom_aromatic', 'attach_atom_hybrid',
                         'attach_atom_num_H']

    # Process all candidates
    log(f"\nProcessing {len(sampled_queries)} queries...")
    all_rows = []
    t0 = time.time()

    for i, qid in enumerate(sampled_queries):
        grp = df[df['query_id'] == qid]
        old_smi = grp['old_fragment_smiles'].iloc[0]
        attach_info = get_attachment_info(old_smi)

        for _, row in grp.iterrows():
            cand_smi = row['candidate_smiles']
            label = row['label']

            row_data = {
                'query_id': qid, 'old_fragment': old_smi,
                'candidate_smiles': cand_smi, 'label': label,
                'blend_score': row['blend_score'], 'blend_rank': row['blend_rank'],
            }

            # Attachment info
            for k in attach_feat_keys:
                row_data[k] = attach_info.get(k, None)

            # Fragment-level chemical context
            chem = get_fragment_features(old_smi, cand_smi)
            row_data.update(chem)

            all_rows.append(row_data)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = len(all_rows) / (elapsed + 0.01)
            log(f"  [{i+1}/{len(sampled_queries)}] {len(all_rows)} rows, {rate:.0f}/s")

    chem_df = pd.DataFrame(all_rows)
    log(f"\nChemical context: {len(chem_df)} rows, {len(chem_df.columns)} cols")
    log(f"Time: {(time.time()-t0)/60:.1f} min")

    chem_df.to_csv(f"{OUT}/d4s29_chemistry_sample.csv", index=False)

    # ── Analysis ──
    log("\n=== CHEMICAL CONTEXT SIGNAL ANALYSIS ===")

    # Numerical feature comparison
    num_cols = [c for c in chem_df.columns
                if c not in ['query_id','old_fragment','candidate_smiles','label',
                              'attach_bond_type','attach_atom_hybrid']
                and chem_df[c].dtype in ['float64','float32','int64']
                and not chem_df[c].isna().all()]
    from scipy import stats

    sig_features = []
    for col in sorted(num_cols):
        pos_vals = chem_df[chem_df['label'] == 1][col].dropna()
        neg_vals = chem_df[chem_df['label'] == 0][col].dropna()
        if len(pos_vals) < 3 or len(neg_vals) < 3:
            continue
        try:
            t_stat, p_val = stats.ttest_ind(pos_vals, neg_vals)
        except:
            continue
        if p_val < 0.1:
            diff = pos_vals.mean() - neg_vals.mean()
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
            sig_features.append((col, pos_vals.mean(), neg_vals.mean(), diff, p_val, sig))
            log(f"  {col:30s} pos={pos_vals.mean():+.4f} neg={neg_vals.mean():+.4f} "
                f"diff={diff:+.4f} p={p_val:.4f} {sig}")

    # Attachment bond type breakdown
    log("\n=== ATTACHMENT BOND TYPE vs LABEL RATE (Cc1ccccc1) ===")
    cc1 = chem_df[chem_df['old_fragment'] == target_frag]
    for bt in cc1['attach_bond_type'].unique():
        subset = cc1[cc1['attach_bond_type'] == bt]
        log(f"  {bt}: n={len(subset):,}, label_rate={subset['label'].mean():.4f}")

    # Top chemical features for label prediction
    if sig_features:
        log(f"\n=== TOP 10 CHEMICAL SIGNAL FEATURES ===")
        sorted_feats = sorted(sig_features, key=lambda x: abs(x[3]), reverse=True)
        for col, pos_m, neg_m, diff, p_val, sig in sorted_feats[:10]:
            log(f"  {col:35s} diff={diff:+.4f} p={p_val:.6f} {sig}")

    # Within-query ranking check: chemical features vs blend
    log(f"\n=== RANKING: Chem features (*Cc1ccccc1) ===")
    # Build simple chem score: normalize features and sum
    num_cols_valid = [c for c in num_cols if c in chem_df.columns and chem_df[c].notna().sum() > 10]
    if len(num_cols_valid) > 0:
        from sklearn.preprocessing import StandardScaler
        X_chem = chem_df[num_cols_valid].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_chem)
        # Simple sum of z-scored features (direction based on pos-neg diff)
        chem_df['chem_score'] = X_scaled.sum(axis=1)

        cc1 = chem_df[chem_df['old_fragment'] == target_frag]
        chem_hits = 0; blend_hits = 0; both_hit = 0
        chem_only = 0; blend_only = 0; neither = 0
        n_q = 0
        for qid in cc1['query_id'].unique():
            grp = cc1[cc1['query_id'] == qid]
            n_q += 1

            c_hit = False; b_hit = False
            if grp['chem_score'].notna().sum() > 0:
                c_ranked = grp.sort_values('chem_score', ascending=False)
                c_labels = c_ranked['label'].values
                c_pos = np.where(c_labels == 1)[0] + 1
                c_hit = len(c_pos) > 0 and c_pos.min() <= 10
            b_ranked = grp.sort_values('blend_score', ascending=False)
            b_labels = b_ranked['label'].values
            b_pos = np.where(b_labels == 1)[0] + 1
            b_hit = len(b_pos) > 0 and b_pos.min() <= 10

            if c_hit: chem_hits += 1
            if b_hit: blend_hits += 1
            if c_hit and b_hit: both_hit += 1
            elif c_hit: chem_only += 1
            elif b_hit: blend_only += 1
            else: neither += 1

        log(f"  Queries={n_q}")
        log(f"  Chem Top10:  {chem_hits/n_q:.4f} ({chem_hits}/{n_q})")
        log(f"  Blend Top10: {blend_hits/n_q:.4f} ({blend_hits}/{n_q})")
        log(f"  Both={both_hit}, Chem-only={chem_only}, Blend-only={blend_only}, "
            f"Neither={neither}")

        if chem_only > 0:
            log(f"  ++ Chem provides orthogonal signal! ({chem_only} queries chem-only)")

    # ── Summary ──
    log(f"\n=== PHASE 3 SUMMARY ===")
    log(f"Phase 1 (geometry):     shape 4 shape-only rescues, signal present but weak")
    log(f"Phase 2 (activity):     BLOCKED - no bioactivity data in ChEMBL37K")
    log(f"Phase 3 (chemistry):    {chem_only if len(num_cols_valid) > 0 else 'N/A'} chem-only rescues")

if __name__ == "__main__":
    main()
