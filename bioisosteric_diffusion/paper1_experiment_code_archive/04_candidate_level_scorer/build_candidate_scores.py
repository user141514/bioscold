#!/usr/bin/env python3
"""
D4S27 Phase 2: Build comprehensive candidate-level prior score/rank file.

FIXED: True Morgan Tanimoto (not bit intersection) for similarity-transfer P4.
Added: similarity histogram, nearest-old table, old-level stratification.
"""
import pandas as pd
import numpy as np
import gzip, os, json, time, sys, warnings
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
OUT = f"{PROJECT}/goal/A_improve/results"
os.makedirs(OUT, exist_ok=True)

PRIOR_DIR = f"{PROJECT}/plan_results/routeA_chembl37k_d4s12_context_prior_adapter"
P4_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P4_old_attach_cand.csv.gz"
P3_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P3_cluster_cand.csv.gz"
P2_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P2_old_cand.csv.gz"
P1_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P1_attach_cand.csv.gz"
P0_PRIOR = f"{PRIOR_DIR}/d4s12_prior_P0_candidate.csv.gz"
GLOBAL_STATS = f"{PRIOR_DIR}/d4s12_global_stats.json"

MATRIX_DIR = f"{PROJECT}/plan_results/routeA_chembl37k_d4s6_candidate_matrix_recovery"
VAL_MATRIX = f"{MATRIX_DIR}/d4s6_phase2c_corrected_candidate_matrix_val.csv.gz"
BLIND_MATRIX = f"{MATRIX_DIR}/d4s6_phase2b_corrected_candidate_matrix_blind.csv.gz"
VAL_META = f"{PROJECT}/plan_results/routeA_chembl37k_d4s2_listwise_reranker/matrices/val/d4s2_query_meta_val.csv"
BLIND_META = f"{PROJECT}/plan_results/routeA_chembl37k_d4s2_listwise_reranker/matrices/blind_test/d4s2_query_meta_blind_test.csv"

t0_total = time.time()
LOG = []

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
    LOG.append(msg)

def save_log():
    with open(f"{PROJECT}/goal/A_improve/d4s27_build.log", 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))

# ── Fingerprints ─────────────────────────────────────────────────────────────
def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros(nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def batch_fps(smiles_list, radius=2, nBits=2048, label="FPs"):
    fps = np.zeros((len(smiles_list), nBits))
    for i, s in enumerate(smiles_list):
        fps[i] = smiles_to_fp(s, radius, nBits)
        if (i + 1) % 5000 == 0:
            log(f"  {label}: {i+1}/{len(smiles_list)}")
    return fps

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load priors
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 60)
log("LOADING D4S12 PRIORS")
log("=" * 60)

p4 = pd.read_csv(P4_PRIOR)
p3 = pd.read_csv(P3_PRIOR)
p2 = pd.read_csv(P2_PRIOR)
p1 = pd.read_csv(P1_PRIOR)
p0 = pd.read_csv(P0_PRIOR)
with open(GLOBAL_STATS) as f:
    gs = json.load(f)

log(f"  P4={len(p4):,}  P3={len(p3):,}  P2={len(p2):,}  P1={len(p1):,}  P0={len(p0):,}")
log(f"  Global positive rate: {gs['global_positive_rate']:.4f}")

ALPHA = 'a1'

# ══════════════════════════════════════════════════════════════════════════════
# 2. Train old_fragment fingerprints
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("BUILDING TRUE TANIMOTO SIMILARITY-TRANSFER INDEX")
log("=" * 60)

train_olds = sorted(p4['old_fragment_smiles'].unique())
log(f"  Train old_fragments: {len(train_olds)}")
log("  Computing train fingerprints...")
train_fps = batch_fps(train_olds, label="train_FPs")
train_fp_sums = train_fps.sum(axis=1).astype(np.float64)

def build_similarity_transfer(df, label="val"):
    """True Morgan Tanimoto nearest-neighbor lookup."""
    log(f"\n  --- {label}: similarity transfer (TRUE Tanimoto) ---")
    val_olds = sorted(df['old_fragment_smiles'].unique())
    log(f"  {label} old_fragments: {len(val_olds)}")
    log(f"  Computing {label} fingerprints...")
    val_fps = batch_fps(val_olds, label=f"{label}_FPs")

    old_to_nearest = {}
    old_to_sim = {}
    sim_records = []

    for i, vo in enumerate(val_olds):
        intersection = np.dot(train_fps, val_fps[i]).astype(np.float64)
        val_sum = float(val_fps[i].sum())
        denom = train_fp_sums + val_sum - intersection
        denom[denom == 0] = 1.0
        sims = intersection / denom  # TRUE Morgan Tanimoto
        best = np.argmax(sims)
        best_train = train_olds[best]
        best_sim = float(sims[best])
        old_to_nearest[vo] = best_train
        old_to_sim[vo] = best_sim
        sim_records.append({
            'val_old_fragment': vo,
            'nearest_train_old': best_train,
            'tanimoto_similarity': best_sim,
        })

    # Save nearest-old table
    sim_df = pd.DataFrame(sim_records)
    sim_path = f"{OUT}/tanimoto_nearest_old_{label}.csv"
    sim_df.to_csv(sim_path, index=False)
    log(f"  Saved nearest-old table: {sim_path}")

    # Log similarity distribution
    sims_arr = np.array([r['tanimoto_similarity'] for r in sim_records])
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(sims_arr, bins=bins)
    log(f"  Similarity distribution ({len(sim_records)} old_fragments):")
    for j in range(len(bins)-1):
        if hist[j] > 0:
            log(f"    [{bins[j]:.1f}-{bins[j+1]:.1f}): {hist[j]}")
    log(f"  Mean similarity: {sims_arr.mean():.4f}, Median: {np.median(sims_arr):.4f}")
    log(f"  Min: {sims_arr.min():.4f}, Max: {sims_arr.max():.4f}")

    # Apply to dataframe
    df = df.copy()
    df['_nearest_train_old'] = df['old_fragment_smiles'].map(old_to_nearest)
    df['_tanimoto_similarity'] = df['old_fragment_smiles'].map(old_to_sim)

    # Merge P4 on (nearest_train_old, attachment_signature, candidate_smiles)
    p4m = p4[['old_fragment_smiles', 'attachment_signature', 'candidate_smiles',
              f'logit_prior_{ALPHA}', f'smoothed_rate_{ALPHA}',
              'exposure_count', 'positive_count']].copy()
    p4m.columns = ['_nearest_train_old', 'attachment_signature', 'candidate_smiles',
                   't_p4_logit', 't_p4_rate', 't_p4_exposure', 't_p4_positive']

    df = df.merge(p4m, on=['_nearest_train_old', 'attachment_signature', 'candidate_smiles'],
                  how='left')
    cov = df['t_p4_logit'].notna().mean()
    log(f"  Transferred P4 coverage: {cov*100:.1f}%")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 3. Compute all prior signals
# ══════════════════════════════════════════════════════════════════════════════
def compute_all_priors(df):
    log("\n  Computing prior signals...")

    # P1
    p1m = p1[['attachment_signature', 'candidate_smiles',
              f'logit_prior_{ALPHA}', f'smoothed_rate_{ALPHA}',
              'exposure_count', 'positive_count']].copy()
    p1m.columns = ['attachment_signature', 'candidate_smiles',
                   'p1_logit', 'p1_rate', 'p1_exposure', 'p1_positive']
    df = df.merge(p1m, on=['attachment_signature', 'candidate_smiles'], how='left')
    log(f"  P1 coverage: {df['p1_logit'].notna().mean()*100:.1f}%")

    # P3
    df['_cluster_id'] = df['old_fragment_cluster_if_available'].fillna('UNKNOWN')
    p3m = p3[['cluster_id', 'candidate_smiles',
              f'logit_prior_{ALPHA}', f'smoothed_rate_{ALPHA}',
              'exposure_count', 'positive_count']].copy()
    p3m.columns = ['_cluster_id', 'candidate_smiles',
                   'p3_logit', 'p3_rate', 'p3_exposure', 'p3_positive']
    df = df.merge(p3m, on=['_cluster_id', 'candidate_smiles'], how='left')
    log(f"  P3 coverage: {df['p3_logit'].notna().mean()*100:.1f}%")

    # P0
    p0m = p0[['candidate_smiles', f'logit_prior_{ALPHA}', f'smoothed_rate_{ALPHA}',
              'exposure_count', 'positive_count']].copy()
    p0m.columns = ['candidate_smiles', 'p0_logit', 'p0_rate', 'p0_exposure', 'p0_positive']
    df = df.merge(p0m, on='candidate_smiles', how='left')

    t4_avail = df['t_p4_logit'].notna().values
    p3_avail = df['p3_logit'].notna().values
    p1_avail = df['p1_logit'].notna().values
    p0_avail = df['p0_logit'].notna().values

    # Backoff: tP4 -> P3 -> P1 -> P0
    backoff = np.full(len(df), np.nan)
    backoff[t4_avail] = df.loc[t4_avail, 't_p4_logit'].values
    fb3 = p3_avail & ~t4_avail
    backoff[fb3] = df.loc[fb3, 'p3_logit'].values
    fb1 = p1_avail & ~t4_avail & ~p3_avail
    backoff[fb1] = df.loc[fb1, 'p1_logit'].values
    fb0 = p0_avail & ~t4_avail & ~p3_avail & ~p1_avail
    backoff[fb0] = df.loc[fb0, 'p0_logit'].values
    df['backoff_logit'] = backoff

    backoff_support = np.full(len(df), np.nan, dtype=float)
    backoff_support[t4_avail] = df.loc[t4_avail, 't_p4_exposure'].values
    backoff_support[fb3] = df.loc[fb3, 'p3_exposure'].values
    backoff_support[fb1] = df.loc[fb1, 'p1_exposure'].values
    backoff_support[fb0] = df.loc[fb0, 'p0_exposure'].values
    df['backoff_support'] = backoff_support

    backoff_rate = np.full(len(df), np.nan, dtype=float)
    backoff_rate[t4_avail] = df.loc[t4_avail, 't_p4_rate'].values
    backoff_rate[fb3] = df.loc[fb3, 'p3_rate'].values
    backoff_rate[fb1] = df.loc[fb1, 'p1_rate'].values
    backoff_rate[fb0] = df.loc[fb0, 'p0_rate'].values
    df['backoff_smoothed_rate'] = backoff_rate

    # Continuation prior
    t4_exposure_vals = df['t_p4_exposure'].fillna(0).values
    max_exp = max(float(np.nanmax(t4_exposure_vals)), 1.0)
    conf = np.minimum(t4_exposure_vals / max(10, max_exp), 1.0)
    cont = np.full(len(df), np.nan)
    cont[t4_avail] = (conf[t4_avail] * df.loc[t4_avail, 't_p4_logit'].values +
                      (1 - conf[t4_avail]) * df.loc[t4_avail, 'p1_logit'].values)
    cont[fb3] = df.loc[fb3, 'p3_logit'].values - np.log(2)
    cont[fb1] = df.loc[fb1, 'p1_logit'].values - np.log(4)
    cont[fb0] = df.loc[fb0, 'p0_logit'].values - np.log(8)
    df['cont_prior'] = cont

    # PMI variants
    both_t4p1 = t4_avail & p1_avail
    df['t_pmi_p4_p1'] = np.nan
    df.loc[both_t4p1, 't_pmi_p4_p1'] = (
        df.loc[both_t4p1, 't_p4_logit'].values - df.loc[both_t4p1, 'p1_logit'].values)
    both_p3p1 = p3_avail & p1_avail
    df['pmi_p3_p1'] = np.nan
    df.loc[both_p3p1, 'pmi_p3_p1'] = (
        df.loc[both_p3p1, 'p3_logit'].values - df.loc[both_p3p1, 'p1_logit'].values)
    both_p1p0 = p1_avail & p0_avail
    df['pmi_p1_p0'] = np.nan
    df.loc[both_p1p0, 'pmi_p1_p0'] = (
        df.loc[both_p1p0, 'p1_logit'].values - df.loc[both_p1p0, 'p0_logit'].values)

    log(f"  Backoff coverage: {(~np.isnan(backoff)).mean()*100:.1f}%")
    log(f"  Cont prior coverage: {(~np.isnan(cont)).mean()*100:.1f}%")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 4. Per-query ranks and features
# ══════════════════════════════════════════════════════════════════════════════
def add_ranks_and_query_features(df):
    log("\n  Computing per-query ranks and query features...")
    prior_cols = ['backoff_logit', 'cont_prior', 't_p4_logit', 'p1_logit',
                  'p3_logit', 't_pmi_p4_p1', 'pmi_p3_p1', 'pmi_p1_p0']
    for col in prior_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            df[f'{col}_rank'] = df.groupby('query_id')[col].rank(ascending=False, method='min')

    def _entropy(grp):
        vals = grp.dropna().values
        if len(vals) < 2:
            return np.nan
        vals_centered = vals - np.nanmax(vals)
        exp_vals = np.exp(np.clip(vals_centered, -50, 50))
        probs = exp_vals / exp_vals.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    ent = df.groupby('query_id')['backoff_logit'].apply(_entropy)
    df['query_prior_entropy'] = df['query_id'].map(ent)

    def _margin(grp):
        vals = grp.dropna().sort_values(ascending=False).values
        return float(vals[0] - vals[1]) if len(vals) >= 2 else (float(vals[0]) if len(vals) == 1 else np.nan)

    mar = df.groupby('query_id')['backoff_logit'].apply(_margin)
    df['query_prior_margin'] = df['query_id'].map(mar)

    def _top_support(grp):
        ranked = grp.sort_values('backoff_logit', ascending=False).head(20)
        return float(ranked['backoff_support'].max()) if ranked['backoff_support'].notna().any() else np.nan

    sup = df.groupby('query_id').apply(_top_support, include_groups=False).reset_index()
    sup.columns = ['query_id', 'query_prior_max_support']
    df = df.merge(sup, on='query_id', how='left')
    log(f"  Added ranks for {len(prior_cols)} prior columns + query features")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 5. Process one split
# ══════════════════════════════════════════════════════════════════════════════
def process_split(matrix_path, meta_path, split_label):
    log(f"\n{'='*60}")
    log(f"PROCESSING: {split_label}")
    log(f"{'='*60}")

    log(f"  Loading candidate matrix...")
    with gzip.open(matrix_path, 'rb') as f:
        df = pd.read_csv(f)
    log(f"  {len(df):,} rows, {df['query_id'].nunique():,} queries, {len(df.columns)} columns")

    log(f"  Loading query meta...")
    meta = pd.read_csv(meta_path)
    if 'target_any_seen_vocab' in meta.columns:
        eval_qids = set(meta[meta['target_any_seen_vocab'] == 1]['query_id'])
        df = df[df['query_id'].isin(eval_qids)].copy()
        log(f"  After target_any_seen_vocab filter: {len(df):,} rows, {df['query_id'].nunique():,} queries")

    meta_cols = ['query_id', 'replacement_frequency_mean', 'replacement_frequency_bin',
                 'hard_top10_miss_flag', 'num_positives', 'single_pos_flag', 'multi_pos_flag',
                 'old_fragment_cluster_id', 'target_any_seen_vocab', 'target_all_seen_vocab']
    meta_cols = [c for c in meta_cols if c in meta.columns]
    df = df.merge(meta[meta_cols], on='query_id', how='left')

    df = build_similarity_transfer(df, split_label)
    df = compute_all_priors(df)
    df = add_ranks_and_query_features(df)

    drop_cols = ['_nearest_train_old', '_cluster_id']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    rename_map = {
        'backoff_logit': 'backoff_logit_score', 'cont_prior': 'cont_prior_score',
        't_p4_logit': 't_p4_logit_score', 't_p4_rate': 't_p4_smoothed_rate',
        't_p4_exposure': 't_p4_support', 'p1_logit': 'p1_logit_score',
        'p1_rate': 'p1_smoothed_rate', 'p1_exposure': 'p1_support',
        'p3_logit': 'p3_logit_score', 'p3_rate': 'p3_smoothed_rate',
        'p3_exposure': 'p3_support', 'p0_logit': 'p0_logit_score',
        'p0_rate': 'p0_smoothed_rate', 'p0_exposure': 'p0_support',
        'is_positive': 'label',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    log(f"  Final: {len(df):,} rows, {len(df.columns)} columns")

    parquet_path = f"{OUT}/candidate_prior_scores_{split_label}.parquet"
    csv_path = f"{OUT}/candidate_prior_scores_{split_label}.csv.gz"
    log(f"  Saving parquet: {parquet_path}")
    df.to_parquet(parquet_path, index=False)
    log(f"  Saving CSV: {csv_path}")
    with gzip.open(csv_path, 'wt') as f:
        df.to_csv(f, index=False)

    log(f"  {split_label} DONE ({len(df.columns)} cols x {len(df):,} rows)")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
log("D4S27 Phase 2: Building candidate-level prior score files (TRUE Tanimoto)")
log(f"Output dir: {OUT}")

val_df = process_split(VAL_MATRIX, VAL_META, "val")
blind_df = process_split(BLIND_MATRIX, BLIND_META, "blind")

elapsed = time.time() - t0_total
log(f"\n{'='*60}")
log(f"Phase 2 COMPLETE in {elapsed/60:.1f} min")
log(f"{'='*60}")
save_log()
