#!/usr/bin/env python3
"""
D4S27: Conditional Transform Prior via old_fragment + attachment_signature.

Key finding: val old_fragments are deliberately held-out (0 overlap with train).
P4(old+att+cand) exact match = 0%. We use:
  A) Similarity-transfer: nearest train old_fragment → transfer P4 stats
  B) Cluster-level prior (P3): cluster+candidate from D4S12
  C) Attachment baseline (P1): att+candidate
  D) PMI = logP(c|proxy_old,att) - logP(c|att)

Leakage: all counts from train only; val uses train-estimated priors only.
"""
import pandas as pd
import numpy as np
import gzip, os, json, time, sys, warnings
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = "E:/zuhui/bioisosteric_diffusion"
OUT = "E:/zuhui/bioisosteric_diffusion/goal/A_improve"

PRIOR_DIR = f"{PROJECT}/plan_results/routeA_chembl37k_d4s12_context_prior_adapter"
P4_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P4_old_attach_cand.csv.gz"
P3_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P3_cluster_cand.csv.gz"
P2_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P2_old_cand.csv.gz"
P1_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P1_attach_cand.csv.gz"
P0_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P0_candidate.csv.gz"
GLOBAL_STATS = f"{PRIOR_DIR}/d4s12_global_stats.json"

VAL_MATRIX = f"{PROJECT}/plan_results/routeA_chembl37k_d4s6_candidate_matrix_recovery/d4s6_phase2c_corrected_candidate_matrix_val.csv.gz"
VAL_META = f"{PROJECT}/plan_results/routeA_chembl37k_d4s2_listwise_reranker/matrices/val/d4s2_query_meta_val.csv"
TRAIN_META = f"{PROJECT}/plan_results/routeA_chembl37k_d4s2_listwise_reranker/matrices/train/d4s2_query_meta_train.csv"

os.makedirs(OUT, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
LOG = []
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.append(line)

def save_log():
    with open(os.path.join(OUT, "d4s27_run.log"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))

# ── Morgan Fingerprint ─────────────────────────────────────────────────────
def smiles_to_fp(smiles, radius=2, nBits=2048):
    """Morgan fingerprint as numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros(nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def batch_fingerprints(smiles_list, radius=2, nBits=2048):
    fps = np.zeros((len(smiles_list), nBits))
    for i, s in enumerate(smiles_list):
        fps[i] = smiles_to_fp(s, radius, nBits)
        if (i + 1) % 1000 == 0:
            log(f"  FPs: {i+1}/{len(smiles_list)}")
    return fps

# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(df, score_col, label_col='is_positive', group_col='query_id'):
    """Per-query ranking metrics."""
    def _qm(grp):
        grp = grp.sort_values(score_col, ascending=False)
        labels = grp[label_col].values
        ranks = np.where(labels == 1)[0] + 1
        num_pos = labels.sum()
        mrr = 1.0 / ranks[0] if len(ranks) > 0 else 0.0
        return pd.Series({
            'num_positives': num_pos, 'best_rank': ranks[0] if len(ranks) > 0 else 9999,
            'Top1': int(ranks[0] == 1) if len(ranks) > 0 else 0,
            'Top5': int(ranks[0] <= 5) if len(ranks) > 0 else 0,
            'Top10': int(ranks[0] <= 10) if len(ranks) > 0 else 0,
            'Top20': int(ranks[0] <= 20) if len(ranks) > 0 else 0,
            'MRR': mrr,
        })
    res = df.groupby(group_col, sort=False).apply(_qm).reset_index()
    return {
        'queries': len(res), 'rows': len(df),
        'Top1': float(res['Top1'].mean()), 'Top5': float(res['Top5'].mean()),
        'Top10': float(res['Top10'].mean()), 'Top20': float(res['Top20'].mean()),
        'MRR': float(res['MRR'].mean()),
    }, res


def borda_fusion(df, rank_cols, weights=None):
    n = len(rank_cols)
    if weights is None:
        weights = [1.0] * n
    weights = np.array(weights) / np.sum(weights)
    borda = np.zeros(len(df))
    for col, w in zip(rank_cols, weights):
        borda += w * df[col].values
    return -borda


def per_query_zscore(series, qids):
    df_tmp = pd.DataFrame({'v': np.asarray(series), 'q': np.asarray(qids)})
    m = df_tmp.groupby('q')['v'].transform('mean')
    s = df_tmp.groupby('q')['v'].transform('std')
    return (np.asarray(series) - m.values) / (s.values + 1e-6)


# ── Similarity-based P4 transfer ───────────────────────────────────────────
def build_similarity_transfer(val_df, p4_df, p1_df, train_meta_df):
    """
    For each val old_fragment, find the most similar TRAIN old_fragment
    (by Morgan fingerprint), then transfer its P4 statistics.
    Returns updated val_df with transferred P4 scores.
    """
    log("\n--- Building similarity-based P4 transfer ---")

    # Get unique train old_fragments from P4
    train_olds = sorted(p4_df['old_fragment_smiles'].unique())
    val_olds = sorted(val_df['old_fragment_smiles'].unique())
    log(f"  Train old_fragments: {len(train_olds)}, Val old_fragments: {len(val_olds)}")

    # Compute fingerprints
    log("  Computing train fingerprints...")
    train_fps = batch_fingerprints(train_olds)
    log("  Computing val fingerprints...")
    val_fps = batch_fingerprints(val_olds)

    # For each val old_fragment, find nearest train old_fragment
    log("  Finding nearest neighbors...")
    old_to_nearest = {}
    old_to_similarity = {}
    for i, val_old in enumerate(val_olds):
        sims = np.dot(train_fps, val_fps[i])  # Tanimoto for binary FPs
        best_idx = np.argmax(sims)
        old_to_nearest[val_old] = train_olds[best_idx]
        old_to_similarity[val_old] = sims[best_idx]
        log(f"    {val_old[:30]:30s} → {train_olds[best_idx][:30]:30s} sim={sims[best_idx]:.3f}")

    # Build transferred P4 lookup: (nearest_train_old, att, cand) → P4 stats
    log("  Building transferred P4 lookup...")
    transferred_p4 = {}
    for _, row in p4_df.iterrows():
        train_old = str(row['old_fragment_smiles'])
        att = str(row['attachment_signature'])
        cand = str(row['candidate_smiles'])
        # This train_old may be the nearest neighbor for some val_old
        transferred_p4[(train_old, att, cand)] = {
            'logit': row['logit_prior_a1'], 'rate': row['smoothed_rate_a1'],
            'exposure': row['exposure_count']
        }

    # Apply to val via merge: map val_old → nearest_train_old → merge with P4
    log("  Applying transferred P4 to val via merge...")
    val_df = val_df.copy()
    val_df['_nearest_old'] = val_df['old_fragment_smiles'].map(old_to_nearest)

    # Merge transferred P4 on (nearest_old, att, cand)
    p4m = p4_df[['old_fragment_smiles', 'attachment_signature', 'candidate_smiles',
                  'logit_prior_a1', 'smoothed_rate_a1', 'exposure_count']].copy()
    p4m.columns = ['_nearest_old', 'attachment_signature', 'candidate_smiles',
                   't_p4_logit', 't_p4_rate', 't_p4_exposure']
    val_df = val_df.merge(p4m, on=['_nearest_old', 'attachment_signature', 'candidate_smiles'], how='left')

    cov = val_df['t_p4_logit'].notna().mean()
    log(f"  Transferred P4 coverage: {cov*100:.1f}%")

    t_p4_logit = val_df['t_p4_logit'].values
    t_p4_rate = val_df['t_p4_rate'].values
    t_p4_exposure = val_df['t_p4_exposure'].fillna(0).values

    return val_df, t_p4_logit, t_p4_rate, t_p4_exposure, old_to_nearest, old_to_similarity


# ── Cluster-level (P3) + Attachment (P1) PMI ──────────────────────────────
def build_cluster_att_pmi(val_df, p3_df, p1_df):
    """
    Compute PMI between cluster-level prior (P3) and attachment-level prior (P1).
    Uses merge-based approach for speed.
    """
    log("\n--- Building cluster+attachment PMI ---")

    val_df = val_df.copy()
    val_df['_cluster_id'] = val_df['old_fragment_cluster_if_available'].fillna('UNKNOWN')

    # Merge P3 on (cluster_id, candidate)
    p3m = p3_df[['cluster_id', 'candidate_smiles', 'logit_prior_a1', 'smoothed_rate_a1']].copy()
    p3m.columns = ['_cluster_id', 'candidate_smiles', 'p3_logit', 'p3_rate']
    val_df = val_df.merge(p3m, on=['_cluster_id', 'candidate_smiles'], how='left')
    cov3 = val_df['p3_logit'].notna().mean()
    log(f"  P3 coverage: {cov3*100:.1f}%")

    # Compute P3-P1 PMI
    p1_logit_vals = val_df['p1_logit'].values
    p3_logit_vals = val_df['p3_logit'].values
    both = ~np.isnan(p3_logit_vals) & ~np.isnan(p1_logit_vals)
    val_df['p3_p1_pmi'] = np.nan
    val_df.loc[both, 'p3_p1_pmi'] = p3_logit_vals[both] - p1_logit_vals[both]
    log(f"  P3-P1 PMI coverage: {both.mean()*100:.1f}%")

    return val_df


# ── Stratified Report ──────────────────────────────────────────────────────
def stratified_report(df, score_col, baseline_col, label=""):
    log(f"\n{'='*60}")
    log(f"STRATIFIED: {score_col} vs {baseline_col} [{label}]")
    log(f"{'='*60}")

    base_m, base_q = compute_metrics(df, baseline_col)
    new_m, new_q = compute_metrics(df, score_col)

    log(f"  Base: Top10={base_m['Top10']:.4f} MRR={base_m['MRR']:.4f}")
    log(f"  New:  Top10={new_m['Top10']:.4f} MRR={new_m['MRR']:.4f}")
    log(f"  Δ:    Top10={new_m['Top10']-base_m['Top10']:+.4f} MRR={new_m['MRR']-base_m['MRR']:+.4f}")

    mq = base_q.merge(new_q, on='query_id', suffixes=('_base', '_new'))

    # Rescue
    rescue = (mq['best_rank_base'] > 10) & (mq['best_rank_base'] <= 20) & (mq['best_rank_new'] <= 10)
    log(f"  Rank 11-20 → Top10: {rescue.sum()}/{len(mq)} ({rescue.sum()/len(mq)*100:.1f}%)")

    miss = mq['best_rank_base'] > 10
    miss_hit = miss & (mq['best_rank_new'] <= 10)
    miss_impr = miss & (mq['best_rank_new'] > 10) & (mq['best_rank_new'] < mq['best_rank_base'])
    miss_worse = miss & (mq['best_rank_new'] >= mq['best_rank_base'])
    log(f"  Baseline misses: {miss.sum()}")
    log(f"    → Hit:     {miss_hit.sum()} ({miss_hit.sum()/max(miss.sum(),1)*100:.1f}%)")
    log(f"    → Improved: {miss_impr.sum()} ({miss_impr.sum()/max(miss.sum(),1)*100:.1f}%)")
    log(f"    → Worse:    {miss_worse.sum()} ({miss_worse.sum()/max(miss.sum(),1)*100:.1f}%)")

    hit = mq['best_rank_base'] <= 10
    hit_lost = hit & (mq['best_rank_new'] > 10)
    log(f"  Baseline hits: {hit.sum()}, lost: {hit_lost.sum()}")

    return mq


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("D4S27: Conditional Transform Prior (similarity-transfer)")
    log("=" * 60)
    log(f"Python: {sys.executable}")

    # 1. Load priors
    log("\n--- Loading D4S12 priors ---")
    t0 = time.time()
    p4 = pd.read_csv(P4_PRIOR)
    p3 = pd.read_csv(P3_PRIOR)
    p2 = pd.read_csv(P2_PRIOR)
    p1 = pd.read_csv(P1_PRIOR)
    p0 = pd.read_csv(P0_PRIOR)
    with open(GLOBAL_STATS) as f:
        gs = json.load(f)
    global_rate = gs['global_positive_rate']
    log(f"  P4={len(p4):,} P3={len(p3):,} P2={len(p2):,} P1={len(p1):,} P0={len(p0):,}")

    # 2. Load validation data
    log("\n--- Loading validation data ---")
    with gzip.open(VAL_MATRIX, 'rb') as f:
        val = pd.read_csv(f)
    val_meta = pd.read_csv(VAL_META)
    eval_qids = set(val_meta[val_meta['target_any_seen_vocab'] == 1]['query_id'])
    val_eval = val[val['query_id'].isin(eval_qids)].copy()
    log(f"  Eval: {len(val_eval):,} rows, {val_eval['query_id'].nunique():,} queries")

    # 3. Audit coverage
    log("\n--- Coverage Audit ---")
    train_olds = set(p4['old_fragment_smiles'].unique())
    val_olds = set(val_eval['old_fragment_smiles'].unique())
    log(f"  Train old_fragments: {len(train_olds)}")
    log(f"  Val old_fragments: {len(val_olds)}")
    log(f"  P4(old+att+cand) exact overlap: {len(train_olds & val_olds)} (0% coverage)")
    log(f"  P2(old+cand) exact overlap: {len(train_olds & val_olds)} (0% coverage)")

    val_clusters = set(val_eval['old_fragment_cluster_if_available'].dropna().unique())
    p3_clusters = set(p3['cluster_id'].unique())
    log(f"  P3(cluster+cand) overlap: {len(val_clusters & p3_clusters)}/{len(val_clusters)} clusters")

    # 4. Load train meta for similarity transfer
    log("\n--- Loading train meta for similarity transfer ---")
    train_meta = pd.read_csv(TRAIN_META, usecols=['query_id', 'old_fragment_smiles'])
    log(f"  Train meta: {len(train_meta)} queries")

    # 5. P1 baseline (attachment-level prior) — always works
    log("\n--- Building P1 lookup ---")
    p1_lookup = {}
    for _, row in p1.iterrows():
        p1_lookup[(str(row['attachment_signature']), str(row['candidate_smiles']))] = {
            'logit': row['logit_prior_a1'], 'rate': row['smoothed_rate_a1'],
            'exposure': row['exposure_count']
        }
    log(f"  P1: {len(p1_lookup):,} keys")

    # Apply P1 via merge
    p1m = p1[['attachment_signature', 'candidate_smiles', 'logit_prior_a1', 'smoothed_rate_a1']].copy()
    p1m.columns = ['attachment_signature', 'candidate_smiles', 'p1_logit', 'p1_rate']
    val_eval = val_eval.merge(p1m, on=['attachment_signature', 'candidate_smiles'], how='left')
    log(f"  P1 merged: {val_eval['p1_logit'].notna().mean()*100:.1f}% coverage")

    # 6. Similarity-transfer P4
    val_eval, t_p4_logit, t_p4_rate, t_p4_exposure, old_to_nearest, old_to_sim = \
        build_similarity_transfer(val_eval, p4, p1, train_meta)
    val_eval['t_p4_rate'] = t_p4_rate
    val_eval['t_p4_exposure'] = t_p4_exposure
    val_eval['t_p4_similarity'] = val_eval['old_fragment_smiles'].map(old_to_sim)

    # 7. Compute transferred-P4-based PMI
    log("\n--- Computing transferred PMI ---")
    p1_logit_arr = val_eval['p1_logit'].values
    both = ~np.isnan(t_p4_logit) & ~np.isnan(p1_logit_arr)
    t_pmi_logit = np.full(len(val_eval), np.nan)
    t_pmi_logit[both] = t_p4_logit[both] - p1_logit_arr[both]
    val_eval['t_pmi_logit'] = t_pmi_logit

    eps = 1e-10
    logP4_t = np.full(len(val_eval), np.nan)
    logP4_t[~np.isnan(t_p4_rate)] = np.log(np.clip(t_p4_rate[~np.isnan(t_p4_rate)], eps, 1-eps))
    logP1_arr = np.full(len(val_eval), np.nan)
    p1_rate_arr = val_eval['p1_rate'].values
    logP1_arr[~np.isnan(p1_rate_arr)] = np.log(np.clip(p1_rate_arr[~np.isnan(p1_rate_arr)], eps, 1-eps))
    both_rate = ~np.isnan(t_p4_rate) & ~np.isnan(p1_rate_arr)
    t_pmi_exact = np.full(len(val_eval), np.nan)
    t_pmi_exact[both_rate] = logP4_t[both_rate] - logP1_arr[both_rate]
    val_eval['t_pmi_exact'] = t_pmi_exact
    log(f"  Transferred PMI coverage: {both.mean()*100:.1f}%")

    # 8. Cluster-level (P3) priors
    val_eval = build_cluster_att_pmi(val_eval, p3, p1)

    # 9. Backoff: t_P4 → P3 → P1 → P0
    log("\n--- Building backoff prior ---")
    p3_logit_arr = val_eval['p3_logit'].values
    p4_avail = ~np.isnan(t_p4_logit)
    p3_avail = ~np.isnan(p3_logit_arr)
    p1_avail = ~np.isnan(p1_logit_arr)

    backoff = np.full(len(val_eval), np.nan)
    backoff[p4_avail] = t_p4_logit[p4_avail]
    fb3 = p3_avail & ~p4_avail
    backoff[fb3] = p3_logit_arr[fb3]
    fb1 = p1_avail & ~p4_avail & ~p3_avail
    backoff[fb1] = p1_logit_arr[fb1]
    val_eval['backoff_logit'] = backoff
    log(f"  Backoff coverage: {(~np.isnan(backoff)).mean()*100:.1f}%")

    # 10. Continuation prior: confidence-weighted transferred P4 + P1
    log("--- Continuation prior ---")
    cont = np.full(len(val_eval), np.nan)
    max_exp = max(t_p4_exposure.max(), 1)
    conf = np.minimum(t_p4_exposure / max(10, max_exp), 1.0)
    cont[p4_avail] = conf[p4_avail] * t_p4_logit[p4_avail] + (1-conf[p4_avail]) * p1_logit_arr[p4_avail]
    cont[fb3] = p3_logit_arr[fb3] - np.log(2)
    cont[fb1] = p1_logit_arr[fb1] - np.log(4)
    val_eval['cont_prior'] = cont
    log(f"  Cont prior coverage: {(~np.isnan(cont)).mean()*100:.1f}%")

    # Also: similarity-gated transferred PMI
    sim_gate = val_eval['t_p4_similarity'].values
    val_eval['t_pmi_gated'] = np.where(sim_gate >= 0.4, val_eval['t_pmi_logit'].values, np.nan)
    log(f"  Similarity-gated PMI (≥0.4): {(~np.isnan(val_eval['t_pmi_gated'].values)).mean()*100:.1f}%")

    # 11. Baseline
    log(f"\n{'='*60}")
    log("BASELINE (blend_score)")
    log(f"{'='*60}")
    base_m, base_q = compute_metrics(val_eval, 'blend_score')
    log(f"  Top1={base_m['Top1']:.4f} Top5={base_m['Top5']:.4f} Top10={base_m['Top10']:.4f} Top20={base_m['Top20']:.4f} MRR={base_m['MRR']:.4f}")

    GATE_TOP10 = 0.834018691588785
    gate_ok = abs(base_m['Top10'] - GATE_TOP10) < 1e-6
    log(f"  Gate: {'PASS' if gate_ok else 'FAIL'}")

    # 12. Evaluate all signals
    log(f"\n{'='*60}")
    log("SIGNAL EVALUATION")
    log(f"{'='*60}")

    all_results = {}
    for col, desc in [
        ('p1_logit', 'P1: logP(c|att) baseline'),
        ('p3_logit', 'P3: logP(c|cluster)'),
        ('p3_p1_pmi', 'P3-P1 PMI: cluster lift over att'),
        ('t_p4_logit', 'Transferred P4: NN(old)→P4 logit'),
        ('t_pmi_logit', 'Transferred PMI: NN(P4)-P1'),
        ('t_pmi_exact', 'Transferred PMI exact'),
        ('t_pmi_gated', 'Transferred PMI (sim≥0.4 gated)'),
        ('backoff_logit', 'Backoff: tP4→P3→P1'),
        ('cont_prior', 'Continuation prior'),
    ]:
        if col not in val_eval.columns:
            continue
        valid = val_eval[col].notna()
        if valid.mean() < 0.01:
            log(f"  {desc:45s} SKIP (<1% coverage)")
            continue
        m, q = compute_metrics(val_eval.loc[valid], col)
        all_results[col] = m
        log(f"  {desc:45s} Top1={m['Top1']:.4f} Top10={m['Top10']:.4f} MRR={m['MRR']:.4f}")

    # 13. Borda fusion
    log(f"\n{'='*60}")
    log("BORDA FUSION")
    log(f"{'='*60}")

    for bname, rank_cols in [
        ('Borda_DE_HGB', ['rank_DE', 'rank_HGB']),
        ('Borda_DE_HGB_attach', ['rank_DE', 'rank_HGB', 'rank_attach']),
    ]:
        val_eval[bname] = borda_fusion(val_eval, rank_cols)
        m, _ = compute_metrics(val_eval, bname)
        log(f"  {bname}: Top10={m['Top10']:.4f} MRR={m['MRR']:.4f}")

    for prior_col in ['t_pmi_logit', 't_pmi_gated', 'backoff_logit', 'cont_prior',
                       'p3_logit', 'p3_p1_pmi']:
        if prior_col not in val_eval.columns:
            continue
        pname = prior_col.replace('_logit', '').replace('_pmi', '')
        val_eval[f'r_{pname}'] = val_eval.groupby('query_id')[prior_col].rank(ascending=False, method='min')
        bname = f'Borda_DE_HGB_{pname}'
        val_eval[bname] = borda_fusion(val_eval, ['rank_DE', 'rank_HGB', f'r_{pname}'])
        m, _ = compute_metrics(val_eval, bname)
        all_results[bname] = m
        log(f"  {bname}: Top10={m['Top10']:.4f} MRR={m['MRR']:.4f}")

    # 14. Minimal blend
    log(f"\n{'='*60}")
    log("MINIMAL BLEND")
    log(f"{'='*60}")

    z_blend = per_query_zscore(val_eval['blend_score'], val_eval['query_id'])

    for prior_col in ['t_pmi_logit', 't_pmi_gated', 'backoff_logit', 'cont_prior',
                       'p3_logit', 'p3_p1_pmi']:
        if prior_col not in val_eval.columns:
            continue
        valid_mask = val_eval[prior_col].notna()
        if valid_mask.mean() < 0.05:
            continue
        fill_val = val_eval.loc[valid_mask, prior_col].min()
        z_prior = per_query_zscore(val_eval[prior_col].fillna(fill_val), val_eval['query_id'])
        for alpha in [0.95, 0.975, 0.99]:
            bname = f'blend_a{alpha}_{prior_col}'
            val_eval[bname] = alpha * z_blend + (1 - alpha) * z_prior
            m, _ = compute_metrics(val_eval, bname)
            delta = m['Top10'] - base_m['Top10']
            all_results[bname] = m
            log(f"  {bname}: Top10={m['Top10']:.6f} (Δ={delta:+.6f}) MRR={m['MRR']:.6f}")

    # 15. Stratified reports
    log(f"\n{'='*60}")
    log("STRATIFIED REPORTS")
    log(f"{'='*60}")
    for prior_col in ['t_pmi_logit', 't_pmi_gated', 'backoff_logit', 'p3_logit']:
        if prior_col in val_eval.columns and val_eval[prior_col].notna().mean() > 0.05:
            stratified_report(val_eval, prior_col, 'blend_score', prior_col)

    # 16. Save
    log(f"\n{'='*60}")
    log("SAVING")
    log(f"{'='*60}")

    all_results['baseline_blend'] = base_m
    metrics_list = []
    for k, v in all_results.items():
        d = {'method': k}
        d.update({kk: vv for kk, vv in v.items() if kk in ('Top1', 'Top5', 'Top10', 'Top20', 'MRR', 'queries')})
        metrics_list.append(d)
    pd.DataFrame(metrics_list).to_csv(os.path.join(OUT, 'd4s27_all_metrics.csv'), index=False)
    log(f"  Saved d4s27_all_metrics.csv")

    # Per-query metrics
    _, bq = compute_metrics(val_eval, 'blend_score')
    for prior_col in ['t_pmi_logit', 'backoff_logit', 'p3_logit']:
        if prior_col in val_eval.columns:
            _, pq = compute_metrics(val_eval, prior_col)
            bq.merge(pq, on='query_id', suffixes=('_blend', f'_{prior_col}')).to_csv(
                os.path.join(OUT, f'd4s27_query_{prior_col}.csv'), index=False)
    log(f"  Saved per-query comparisons")

    # 17. Summary
    log(f"\n{'='*60}")
    log("D4S27 SUMMARY")
    log(f"{'='*60}")

    best_delta, best_method = 0.0, 'blend_score'
    for k, v in all_results.items():
        d = v['Top10'] - base_m['Top10']
        if d > best_delta:
            best_delta, best_method = d, k

    log(f"  Baseline Top10:  {base_m['Top10']:.6f}")
    log(f"  Best method:     {best_method}")
    log(f"  Max delta Top10: {best_delta:+.6f}")
    log(f"  Beat baseline:   {'YES' if best_delta > 1e-8 else 'NO'}")

    # Signal correlation
    for col in ['t_pmi_logit', 'backoff_logit', 'p3_logit']:
        if col not in val_eval.columns:
            continue
        valid = val_eval[col].notna()
        if valid.sum() > 100:
            r = val_eval.loc[valid, col].corr(val_eval.loc[valid, 'blend_score'])
            log(f"  {col} vs blend_score r={r:.6f}")

    save_log()
    log("\nD4S27 complete. Output: " + OUT)


if __name__ == '__main__':
    main()
