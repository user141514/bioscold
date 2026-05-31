#!/usr/bin/env python3
"""
D4S27 Phase 3+4: Comprehensive Evaluation + Fusion (TRUE Tanimoto).

Evaluations:
  1. Standalone prior Top10/MRR
  2. Borda(DE,HGB,backoff), Borda(DE,HGB,cont), Borda(DE,HGB,backoff,cont)
  3. Validation-selected zscore blend
  4. Minimal gate/reranker (query-level reliability + prior margin/support/entropy)
  5. Overlap/rescue table
  6. Stratified reports (including old-level + similarity bins)
  7. Leakage verification
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import os, time, sys, warnings

warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
OUT = f"{PROJECT}/goal/A_improve/results"
os.makedirs(OUT, exist_ok=True)

LOG = []
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
    LOG.append(msg)
def save_log():
    with open(f"{PROJECT}/goal/A_improve/d4s27_eval.log", 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))

log("Loading candidate prior scores...")
val = pd.read_parquet(f"{OUT}/candidate_prior_scores_val.parquet")
log(f"  Val: {len(val):,} rows, {val['query_id'].nunique():,} queries, {len(val.columns)} cols")

# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(df, score_col, label_col='label', group_col='query_id'):
    def _qm(grp):
        grp = grp.sort_values(score_col, ascending=False)
        labels = grp[label_col].values
        ranks = np.where(labels == 1)[0] + 1
        return pd.Series({
            'num_pos': int(labels.sum()),
            'best_rank': ranks[0] if len(ranks) > 0 else 9999,
            'Top1': int(ranks[0] == 1) if len(ranks) > 0 else 0,
            'Top5': int(ranks[0] <= 5) if len(ranks) > 0 else 0,
            'Top10': int(ranks[0] <= 10) if len(ranks) > 0 else 0,
            'Top20': int(ranks[0] <= 20) if len(ranks) > 0 else 0,
            'MRR': 1.0 / ranks[0] if len(ranks) > 0 else 0.0,
        })
    res = df.groupby(group_col, sort=False).apply(_qm).reset_index()
    return {
        'queries': len(res), 'rows': len(df),
        'Top1': float(res['Top1'].mean()), 'Top5': float(res['Top5'].mean()),
        'Top10': float(res['Top10'].mean()), 'Top20': float(res['Top20'].mean()),
        'MRR': float(res['MRR'].mean()),
    }, res

def per_query_zscore(series, qids):
    df_tmp = pd.DataFrame({'v': np.asarray(series), 'q': np.asarray(qids)})
    m = df_tmp.groupby('q')['v'].transform('mean')
    s = df_tmp.groupby('q')['v'].transform('std')
    return (np.asarray(series) - m.values) / (s.values + 1e-6)

def borda_score(df, rank_cols):
    return -df[rank_cols].mean(axis=1)

GATE_TOP10 = 0.834018691588785

# Rank column map (post-rename scores use pre-rename rank names)
PRIOR_RANK_MAP = {
    'backoff_logit_score': 'backoff_logit_rank',
    'cont_prior_score': 'cont_prior_rank',
    't_p4_logit_score': 't_p4_logit_rank',
    'p3_logit_score': 'p3_logit_rank',
    'p1_logit_score': 'p1_logit_rank',
    't_pmi_p4_p1': 't_pmi_p4_p1_rank',
    'pmi_p3_p1': 'pmi_p3_p1_rank',
    'pmi_p1_p0': 'pmi_p1_p0_rank',
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("1. BASELINE VERIFICATION")
log("="*60)
base_m, base_q = compute_metrics(val, 'blend_score')
log(f"  Blend: Top1={base_m['Top1']:.4f} Top5={base_m['Top5']:.4f} "
    f"Top10={base_m['Top10']:.6f} Top20={base_m['Top20']:.4f} MRR={base_m['MRR']:.6f}")
gate_ok = abs(base_m['Top10'] - GATE_TOP10) < 1e-6
log(f"  Gate check: {'PASS' if gate_ok else 'FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. STANDALONE PRIOR TOP10/MRR
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("2. STANDALONE PRIOR PERFORMANCE")
log("="*60)

prior_signals = {
    'backoff_logit_score': 'Backoff logit (tP4->P3->P1)',
    'cont_prior_score': 'Continuation prior',
    't_p4_logit_score': 'Transferred P4 logit',
    'p1_logit_score': 'P1: logP(c|att)',
    'p3_logit_score': 'P3: logP(c|cluster)',
    't_pmi_p4_p1': 'PMI: transferred P4-P1',
    'pmi_p3_p1': 'PMI: P3-P1 (cluster lift)',
    'pmi_p1_p0': 'PMI: P1-P0 (attachment lift)',
}

standalone_results = {}
standalone_query = {}
for col, desc in prior_signals.items():
    if col not in val.columns:
        continue
    valid = val[col].notna()
    if valid.mean() < 0.01:
        continue
    m, q = compute_metrics(val.loc[valid], col)
    standalone_results[col] = m
    standalone_query[col] = q
    log(f"  {desc:45s} Top1={m['Top1']:.4f} Top10={m['Top10']:.4f} MRR={m['MRR']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. BORDA FUSION
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("3. BORDA FUSION")
log("="*60)

for bname, rank_cols in [('Borda_DE_HGB', ['rank_DE', 'rank_HGB'])]:
    val[f'{bname}'] = borda_score(val, rank_cols)
    m, _ = compute_metrics(val, f'{bname}')
    standalone_results[bname] = m
    log(f"  {bname:35s} Top10={m['Top10']:.4f} MRR={m['MRR']:.4f}")

borda_combos = [
    ('Borda_DE_HGB_backoff', 'backoff_logit_score'),
    ('Borda_DE_HGB_cont', 'cont_prior_score'),
    ('Borda_DE_HGB_backoff_cont', None),
    ('Borda_DE_HGB_tp4', 't_p4_logit_score'),
    ('Borda_DE_HGB_p3', 'p3_logit_score'),
    ('Borda_DE_HGB_p1', 'p1_logit_score'),
    ('Borda_DE_HGB_tpmi', 't_pmi_p4_p1'),
]

for bname, prior_col in borda_combos:
    if prior_col is None:
        rank_cols = ['rank_DE', 'rank_HGB', 'backoff_logit_rank', 'cont_prior_rank']
    else:
        rank_col = PRIOR_RANK_MAP.get(prior_col)
        if rank_col not in val.columns:
            continue
        rank_cols = ['rank_DE', 'rank_HGB', rank_col]

    val[f'{bname}'] = borda_score(val, rank_cols)
    m, _ = compute_metrics(val, f'{bname}')
    standalone_results[bname] = m
    log(f"  {bname:35s} Top10={m['Top10']:.4f} MRR={m['MRR']:.4f} d={m['Top10']-base_m['Top10']:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. ZSCORE BLEND
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("4. ZSCORE BLEND")
log("="*60)

z_blend = per_query_zscore(val['blend_score'], val['query_id'])
blend_results = []

for prior_col, pname in [
    ('backoff_logit_score', 'backoff'), ('cont_prior_score', 'cont'),
    ('t_p4_logit_score', 'tp4'), ('p3_logit_score', 'p3'),
    ('p1_logit_score', 'p1'), ('t_pmi_p4_p1', 'tpmi'),
]:
    if prior_col not in val.columns:
        continue
    valid = val[prior_col].notna()
    fill_val = val.loc[valid, prior_col].min() if valid.any() else 0
    z_prior = per_query_zscore(val[prior_col].fillna(fill_val), val['query_id'])
    for alpha in [0.90, 0.95, 0.975, 0.99, 0.999]:
        bc = alpha * z_blend + (1-alpha) * z_prior
        vt = val.copy(); vt['_b'] = bc
        m, _ = compute_metrics(vt, '_b')
        blend_results.append({
            'prior': pname, 'alpha': alpha,
            'Top10': m['Top10'], 'MRR': m['MRR'],
            'delta_Top10': m['Top10'] - base_m['Top10'],
        })

# 3-way: blend + backoff + cont
z_bo = per_query_zscore(val['backoff_logit_score'].fillna(val['backoff_logit_score'].min()), val['query_id'])
z_co = per_query_zscore(val['cont_prior_score'].fillna(val['cont_prior_score'].min()), val['query_id'])
for a1 in [0.90, 0.95, 0.98]:
    for f2 in [0.3, 0.5, 0.7]:
        a2 = (1-a1) * f2; a3 = (1-a1) * (1-f2)
        bc = a1 * z_blend + a2 * z_bo + a3 * z_co
        vt = val.copy(); vt['_b3'] = bc
        m, _ = compute_metrics(vt, '_b3')
        blend_results.append({
            'prior': f'3way_a1={a1}_f2={f2}', 'alpha': a1,
            'Top10': m['Top10'], 'MRR': m['MRR'],
            'delta_Top10': m['Top10'] - base_m['Top10'],
        })

blend_df = pd.DataFrame(blend_results)
blend_df.to_csv(f"{OUT}/zscore_blend_grid.csv", index=False)
best = blend_df.loc[blend_df['delta_Top10'].idxmax()]
log(f"  Best zscore blend: {best.to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. OVERLAP / RESCUE
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("5. RESCUE / OVERLAP")
log("="*60)

base_qr = base_q[['query_id', 'best_rank']].copy()
base_qr.columns = ['query_id', 'best_rank_base']

rescue_rows = []
for col, name in [('backoff_logit_score', 'backoff'), ('cont_prior_score', 'cont'),
                   ('t_p4_logit_score', 'tp4'), ('p3_logit_score', 'p3'),
                   ('p1_logit_score', 'p1')]:
    if col not in val.columns:
        continue
    m, q = compute_metrics(val, col)
    qr = q[['query_id', 'best_rank']].copy()
    qr.columns = ['query_id', f'best_rank_{name}']
    mq = base_qr.merge(qr, on='query_id')
    miss_bl = mq['best_rank_base'] > 10
    hit_pr = mq[f'best_rank_{name}'] <= 10
    hit_bl = mq['best_rank_base'] <= 10
    miss_pr = mq[f'best_rank_{name}'] > 10
    rescue_rows.append({
        'prior_signal': name, 'total_queries': len(mq),
        'baseline_hit': hit_bl.sum(), 'baseline_miss': miss_bl.sum(),
        'prior_standalone_hit': hit_pr.sum(),
        'rescue_miss_to_hit': (miss_bl & hit_pr).sum(),
        'rescue_pct': (miss_bl & hit_pr).sum() / max(miss_bl.sum(), 1) * 100,
        'lost_hit_to_miss': (hit_bl & miss_pr).sum(),
        'lost_pct': (hit_bl & miss_pr).sum() / max(hit_bl.sum(), 1) * 100,
    })

rescue_df = pd.DataFrame(rescue_rows)

# Cross-rescue: cont vs backoff
_, q_bo = compute_metrics(val, 'backoff_logit_score')
_, q_co = compute_metrics(val, 'cont_prior_score')
mq_cross = base_qr.merge(
    q_bo[['query_id', 'best_rank']].rename(columns={'best_rank': 'best_rank_backoff'}), on='query_id'
).merge(
    q_co[['query_id', 'best_rank']].rename(columns={'best_rank': 'best_rank_cont'}), on='query_id'
)

cont_rescue = (mq_cross['best_rank_base'] > 10) & (mq_cross['best_rank_cont'] <= 10)
backoff_rescue = (mq_cross['best_rank_base'] > 10) & (mq_cross['best_rank_backoff'] <= 10)
both_rescue = cont_rescue & backoff_rescue
log(f"  cont_prior rescues:       {cont_rescue.sum()}")
log(f"  backoff rescues:          {backoff_rescue.sum()}")
log(f"  both rescue (overlap):    {both_rescue.sum()}")
log(f"  cont-only:                {(cont_rescue & ~backoff_rescue).sum()}")
log(f"  backoff-only:             {(backoff_rescue & ~cont_rescue).sum()}")
log(f"  unique rescues:           {(cont_rescue | backoff_rescue).sum()}")

oracle_hit = ((mq_cross['best_rank_base'] <= 10) | (mq_cross['best_rank_cont'] <= 10) |
              (mq_cross['best_rank_backoff'] <= 10))
oracle_top10 = oracle_hit.mean()
log(f"  Oracle(blend|cont|backoff) Top10: {oracle_top10:.6f}")
log(f"  Oracle headroom: {oracle_top10 - base_m['Top10']:+.6f}")

rescue_df.to_csv(f"{OUT}/rescue_overlap.csv", index=False)

# Save query-level metrics for cont_prior and backoff
q_cont_out = q_co[['query_id', 'best_rank', 'Top1', 'Top5', 'Top10', 'Top20', 'MRR']].copy()
q_cont_out.columns = ['query_id', 'best_rank_cont', 'Top1_cont', 'Top5_cont', 'Top10_cont', 'Top20_cont', 'MRR_cont']
q_backoff_out = q_bo[['query_id', 'best_rank', 'Top1', 'Top5', 'Top10', 'Top20', 'MRR']].copy()
q_backoff_out.columns = ['query_id', 'best_rank_backoff', 'Top1_backoff', 'Top5_backoff', 'Top10_backoff', 'Top20_backoff', 'MRR_backoff']
q_base_out = base_q[['query_id', 'best_rank', 'Top1', 'Top5', 'Top10', 'Top20', 'MRR']].copy()
q_base_out.columns = ['query_id', 'best_rank_blend', 'Top1_blend', 'Top5_blend', 'Top10_blend', 'Top20_blend', 'MRR_blend']

q_all = q_base_out.merge(q_cont_out, on='query_id').merge(q_backoff_out, on='query_id')
q_all.to_csv(f"{OUT}/d4s27_query_all.csv", index=False)
log(f"  Saved query-level file with {len(q_all)} queries")

# ══════════════════════════════════════════════════════════════════════════════
# 6. STRATIFIED REPORTS (including old-level)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("6. STRATIFIED REPORTS")
log("="*60)

qmeta_cols = ['query_id', 'replacement_frequency_bin', 'hard_top10_miss_flag',
              'num_positives', 'single_pos_flag', 'multi_pos_flag']
qmeta = val[qmeta_cols].drop_duplicates(subset='query_id')
mq_full = mq_cross.merge(qmeta, on='query_id', how='left')

# Old-level stratification: map each query to its old_fragment
old_map = val[['query_id', 'old_fragment_smiles']].drop_duplicates()
mq_full = mq_full.merge(old_map, on='query_id', how='left')

# Add Tanimoto similarity (from build script output)
tanimoto_val = pd.read_csv(f"{OUT}/tanimoto_nearest_old_val.csv")
old_sim_map = dict(zip(tanimoto_val['val_old_fragment'], tanimoto_val['tanimoto_similarity']))
mq_full['tanimoto_to_nearest_train'] = mq_full['old_fragment_smiles'].map(old_sim_map)
mq_full['tanimoto_bin'] = pd.cut(mq_full['tanimoto_to_nearest_train'],
    bins=[0, 0.3, 0.4, 0.5, 0.6, 1.0],
    labels=['<0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '>0.6'])

# Support bins
mq_full['backoff_support_bin'] = pd.cut(
    mq_full['query_id'].map(val.groupby('query_id')['backoff_support'].max()),
    bins=[0, 10, 100, 1000, 10000, float('inf')],
    labels=['0-10', '10-100', '100-1K', '1K-10K', '10K+']
)

# Rank bins for baseline
mq_full['baseline_rank_bin'] = pd.cut(mq_full['best_rank_base'],
    bins=[0, 1, 5, 10, 20, float('inf')],
    labels=['rank=1', 'rank=2-5', 'rank=6-10', 'rank=11-20', 'rank>20'])

strata_rows = []

# Dimension 1: old_fragment level
for old_frag in sorted(mq_full['old_fragment_smiles'].dropna().unique()):
    sub = mq_full[mq_full['old_fragment_smiles'] == old_frag]
    sim = sub['tanimoto_to_nearest_train'].iloc[0] if len(sub) > 0 else np.nan
    base_hit = (sub['best_rank_base'] <= 10).mean()
    cont_hit = (sub['best_rank_cont'] <= 10).mean()
    bo_hit = (sub['best_rank_backoff'] <= 10).mean()
    cont_r = ((sub['best_rank_base'] > 10) & (sub['best_rank_cont'] <= 10)).sum()
    bo_r = ((sub['best_rank_base'] > 10) & (sub['best_rank_backoff'] <= 10)).sum()
    strata_rows.append({
        'dimension': 'old_fragment', 'category': old_frag[:40],
        'n_queries': len(sub), 'tanimoto': sim,
        'baseline_Top10': base_hit, 'cont_Top10': cont_hit, 'backoff_Top10': bo_hit,
        'cont_rescue': cont_r, 'backoff_rescue': bo_r,
    })

# Dimension 2: Tanimoto similarity bin
for tb in ['<0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '>0.6']:
    sub = mq_full[mq_full['tanimoto_bin'] == tb]
    if len(sub) < 5:
        continue
    base_hit = (sub['best_rank_base'] <= 10).mean()
    cont_r = ((sub['best_rank_base'] > 10) & (sub['best_rank_cont'] <= 10)).sum()
    bo_r = ((sub['best_rank_base'] > 10) & (sub['best_rank_backoff'] <= 10)).sum()
    strata_rows.append({
        'dimension': 'tanimoto_bin', 'category': tb,
        'n_queries': len(sub), 'tanimoto': np.nan,
        'baseline_Top10': base_hit,
        'cont_Top10': (sub['best_rank_cont'] <= 10).mean(),
        'backoff_Top10': (sub['best_rank_backoff'] <= 10).mean(),
        'cont_rescue': cont_r, 'backoff_rescue': bo_r,
    })

# Dimension 3: standard strata
for dim, lbl in [
    ('replacement_frequency_bin', 'Replacement Frequency'),
    ('hard_top10_miss_flag', 'Hard Top10 Miss'),
    ('single_pos_flag', 'Single Positive'),
    ('backoff_support_bin', 'Support Bin'),
    ('baseline_rank_bin', 'Baseline Rank Bin'),
]:
    for cat in mq_full[dim].dropna().unique():
        sub = mq_full[mq_full[dim] == cat]
        if len(sub) < 5:
            continue
        base_hit = (sub['best_rank_base'] <= 10).mean()
        rank11_20 = (sub['best_rank_base'] > 10) & (sub['best_rank_base'] <= 20)
        rank20p = sub['best_rank_base'] > 20
        strata_rows.append({
            'dimension': lbl, 'category': str(cat), 'n_queries': len(sub),
            'tanimoto': np.nan,
            'baseline_Top10': base_hit,
            'cont_Top10': (sub['best_rank_cont'] <= 10).mean(),
            'backoff_Top10': (sub['best_rank_backoff'] <= 10).mean(),
            'cont_rescue': ((sub['best_rank_base'] > 10) & (sub['best_rank_cont'] <= 10)).sum(),
            'backoff_rescue': ((sub['best_rank_base'] > 10) & (sub['best_rank_backoff'] <= 10)).sum(),
            'cont_11_20_rescue': (rank11_20 & (sub['best_rank_cont'] <= 10)).sum(),
            'backoff_11_20_rescue': (rank11_20 & (sub['best_rank_backoff'] <= 10)).sum(),
            'cont_20p_rescue': (rank20p & (sub['best_rank_cont'] <= 10)).sum(),
            'backoff_20p_rescue': (rank20p & (sub['best_rank_backoff'] <= 10)).sum(),
        })

strata_df = pd.DataFrame(strata_rows)
strata_df.to_csv(f"{OUT}/stratified_metrics.csv", index=False)
log(f"  {len(strata_rows)} strata rows saved (including old-level + tanimoto bins)")

# ══════════════════════════════════════════════════════════════════════════════
# 7. GATE / RERANKER
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("7. GATE / RERANKER (OOF 5-fold CV)")
log("="*60)

q_feat = val.groupby('query_id').agg(
    prior_entropy=('query_prior_entropy', 'first'),
    prior_margin=('query_prior_margin', 'first'),
    prior_max_support=('query_prior_max_support', 'first'),
    backoff_max=('backoff_logit_score', 'max'),
    backoff_std=('backoff_logit_score', 'std'),
    cont_max=('cont_prior_score', 'max'),
    cont_std=('cont_prior_score', 'std'),
    blend_max=('blend_score', 'max'),
).reset_index()

q_feat = q_feat.merge(mq_full[['query_id', 'best_rank_base', 'best_rank_cont',
                                'best_rank_backoff']], on='query_id')
q_feat['either_can_rescue'] = (
    ((q_feat['best_rank_base'] > 10) & (q_feat['best_rank_cont'] <= 10)) |
    ((q_feat['best_rank_base'] > 10) & (q_feat['best_rank_backoff'] <= 10))
).astype(int)

log(f"  either_can_rescue: {q_feat['either_can_rescue'].sum()} / {len(q_feat)}")

feat_cols = ['prior_entropy', 'prior_margin', 'prior_max_support',
             'backoff_max', 'backoff_std', 'cont_max', 'cont_std', 'blend_max']
for c in feat_cols:
    q_feat[c] = q_feat[c].fillna(q_feat[c].median())

gkf = GroupKFold(n_splits=5)
gate_preds = np.zeros(len(q_feat))
for fold, (tr, te) in enumerate(gkf.split(q_feat, groups=q_feat['query_id'])):
    Xtr = q_feat.iloc[tr][feat_cols].values
    Xte = q_feat.iloc[te][feat_cols].values
    ytr = q_feat.iloc[tr]['either_can_rescue'].values
    sc = StandardScaler()
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(sc.fit_transform(Xtr), ytr)
    gate_preds[te] = clf.predict_proba(sc.transform(Xte))[:, 1]

q_feat['gate_score'] = gate_preds

gate_eval = []
for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    q_gated = set(q_feat.loc[q_feat['gate_score'] > thr, 'query_id'])
    vt = val.copy()
    vt['_gs'] = -vt['blend_rank'].astype(float)
    gidx = vt['query_id'].isin(q_gated)
    b4 = -vt[['rank_DE', 'rank_HGB', 'cont_prior_rank', 'backoff_logit_rank']].mean(axis=1)
    vt.loc[gidx, '_gs'] = b4[gidx]
    m, _ = compute_metrics(vt, '_gs')
    gate_eval.append({
        'threshold': thr, 'n_gated': len(q_gated),
        'pct_gated': len(q_gated)/len(q_feat)*100,
        'Top10': m['Top10'], 'MRR': m['MRR'],
        'delta_Top10': m['Top10'] - base_m['Top10'],
    })

gate_df = pd.DataFrame(gate_eval)
gate_df.to_csv(f"{OUT}/gate_evaluation.csv", index=False)
best_gate = gate_df.loc[gate_df['delta_Top10'].idxmax()]
log(f"  Best gate: thr={best_gate['threshold']} n={best_gate['n_gated']:.0f} "
    f"Top10={best_gate['Top10']:.6f} d={best_gate['delta_Top10']:+.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("8. LEAKAGE VERIFICATION")
log("="*60)

train_olds = set(pd.read_csv(
    f"{PROJECT}/plan_results/routeA_chembl37k_d4s12_context_prior_adapter/d4s12_prior_P4_old_attach_cand.csv.gz"
)['old_fragment_smiles'].unique())
val_olds = set(val['old_fragment_smiles'].unique())
overlap = train_olds & val_olds
log(f"  Train old_fragments: {len(train_olds)}, Val: {len(val_olds)}")
log(f"  Overlap: {len(overlap)} - {'CLEAN' if len(overlap)==0 else 'LEAKAGE!'}")
log(f"  P4 exact match: 0% (by design)")
log(f"  Gate: 5-fold CV over queries - OOF")

# ══════════════════════════════════════════════════════════════════════════════
# 9. FUSION DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("9. FUSION DIAGNOSIS")
log("="*60)

for c1, c2 in [('backoff_logit_score', 'cont_prior_score'),
               ('backoff_logit_score', 'blend_score'),
               ('cont_prior_score', 'blend_score')]:
    vv = val[c1].notna() & val[c2].notna()
    r = val.loc[vv, c1].corr(val.loc[vv, c2])
    log(f"  corr({c1}, {c2}) = {r:.4f}")

best_ranks = mq_full[['best_rank_base', 'best_rank_backoff', 'best_rank_cont']].dropna()
for c1, c2 in [('best_rank_base', 'best_rank_backoff'),
               ('best_rank_base', 'best_rank_cont'),
               ('best_rank_backoff', 'best_rank_cont')]:
    r, p = spearmanr(best_ranks[c1], best_ranks[c2])
    log(f"  Spearman rank corr({c1}, {c2}) = {r:.4f}")

bo_ok = mq_full['best_rank_backoff'] <= 10
co_ok = mq_full['best_rank_cont'] <= 10
log(f"  Backoff-only Top10: {(bo_ok & ~co_ok).sum()}")
log(f"  Cont-only Top10: {(co_ok & ~bo_ok).sum()}")
log(f"  Both Top10: {(bo_ok & co_ok).sum()}")
log(f"  Complementary: {(bo_ok != co_ok).sum()} / {len(mq_full)} ({(bo_ok != co_ok).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("10. SUMMARY")
log("="*60)

all_results = dict(standalone_results)
all_results['baseline_blend'] = base_m

rows = []
for k, v in all_results.items():
    row = {'method': k}
    row.update({kk: vv for kk, vv in v.items() if kk in ('Top1','Top5','Top10','Top20','MRR','queries')})
    rows.append(row)
sum_df = pd.DataFrame(rows)
sum_df['delta_Top10'] = sum_df['Top10'] - base_m['Top10']
sum_df['delta_MRR'] = sum_df['MRR'] - base_m['MRR']
sum_df = sum_df.sort_values('delta_Top10', ascending=False)
sum_df.to_csv(f"{OUT}/summary_metrics.csv", index=False)

log(f"\n  Baseline Top10: {base_m['Top10']:.6f}")
log(f"  Best method: {sum_df.iloc[0]['method']} Top10={sum_df.iloc[0]['Top10']:.6f}")
log(f"  Max delta Top10: {sum_df['delta_Top10'].max():+.6f}")
log(f"  Oracle Top10: {oracle_top10:.6f} (+{oracle_top10 - base_m['Top10']:+.6f})")
log(f"  Beat baseline: {'YES' if sum_df['delta_Top10'].max() > 1e-8 else 'NO'}")

log(f"\n  OUTPUTS:")
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, f)) / (1024*1024)
    if sz > 0.1:
        log(f"    {f} ({sz:.1f} MB)")
    else:
        log(f"    {f}")

save_log()
log("\nPhase 3+4 COMPLETE.")
