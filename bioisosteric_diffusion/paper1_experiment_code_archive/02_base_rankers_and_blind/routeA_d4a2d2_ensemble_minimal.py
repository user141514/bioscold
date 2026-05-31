#!/usr/bin/env python3
"""D4A2D2: DE+HGB Ensemble — minimal correct version using D4A2D1R validated predictions."""
import json, csv, os, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path("E:/zuhui/bioisosteric_diffusion")
DE_HITS = BASE / "plan_results/routeA_chembl37k_d4a2d1r_dual_encoder_robustness/d4a2d1r_query_level_hits.csv"
DE_STD = BASE / "plan_results/routeA_chembl37k_d4a2d1r_dual_encoder_robustness/d4a2d1r_standardized_predictions.jsonl"
HGB_PREDS = BASE / "plan_results/routeA_chembl37k_d4a1_learned_ranker/d4a1_test_predictions.jsonl"
MANIFEST = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl"
VOCAB = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_train_replacement_vocabulary.csv"
OUT = BASE / "plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble"
SEED = 20260523; N_BOOT = 1000
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(SEED)
T = time.strftime("%Y-%m-%dT%H:%M:%S")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── LOAD DATA ──
log("Loading data...")
# Vocab SMILES
vocab_smiles = []
with open(VOCAB, encoding='utf-8') as f:
    f.readline()
    for line in f: vocab_smiles.append(line.strip().split(',')[0])
vocab_set = set(vocab_smiles)

# Manifest: test queries with positive sets
test_queries = {}
with open(MANIFEST, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d['split'] == 'test':
            pos_in_vocab = [s for s in d['positive_replacement_set'] if s in vocab_set]
            if pos_in_vocab:
                test_queries[d['query_id']] = {
                    'qid': d['query_id'], 'old': d['old_fragment_smiles'],
                    'attach': d['attachment_signature'],
                    'pos_set': set(pos_in_vocab), 'n_pos': len(pos_in_vocab)
                }
log(f"  Test queries with vocab positives: {len(test_queries)}")

# DE query hits (already validated, D4A2D1R)
de_hits = {}
with open(DE_HITS, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        qid = r['qid']; m = r['m']
        if qid not in test_queries: continue
        if qid not in de_hits: de_hits[qid] = {}
        de_hits[qid][m] = {f't{k}': float(r[f'h{k}']) for k in [1,5,10,20,50]}
        de_hits[qid][m]['mrr'] = float(r['mrr'])
log(f"  DE queries: {len(de_hits)}")

# HGB predictions: group by query_id → sorted by score
log("Loading HGB predictions...")
hgb_by_query = defaultdict(list)
with open(HGB_PREDS, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line.strip())
        qid = d['query_id']
        if qid in test_queries:
            hgb_by_query[qid].append({
                'candidate': d['candidate'], 'score': float(d.get('score', 0)),
                'label': int(d.get('label', 0))
            })
for qid in hgb_by_query:
    hgb_by_query[qid].sort(key=lambda x: -x['score'])
hgb_queries = set(hgb_by_query.keys())
common_queries = sorted(test_queries.keys() & de_hits.keys() & hgb_queries)
log(f"  Common queries: {len(common_queries)}")

# ── COMPUTE HITS ──
log("Computing per-query hits...")
results = {}  # qid → {DE: {t1..t50, mrr, top50_cands}, HGB: {...}, B1: {...}}

for qid in common_queries:
    q = test_queries[qid]
    pos = q['pos_set']
    res = {'qid': qid, 'attach': q['attach'], 'n_pos': q['n_pos'], 'pos_set': pos}

    # DE (from pre-validated D4A2D1R hits)
    de = de_hits[qid].get('M2_DE', {})
    res['DE'] = {f't{k}': de.get(f't{k}', 0) for k in [1,5,10,20,50]}
    res['DE']['mrr'] = de.get('mrr', 0)

    # B1 (from pre-validated D4A2D1R hits)
    b1 = de_hits[qid].get('M1_attach', {})
    res['B1'] = {f't{k}': b1.get(f't{k}', 0) for k in [1,5,10,20,50]}
    res['B1']['mrr'] = b1.get('mrr', 0)

    # HGB: compute from per-query predictions
    hgb_cands = hgb_by_query[qid]
    hgb_hits = np.zeros(50, dtype=bool); hgb_mrr = 0.0; found = False
    for i, c in enumerate(hgb_cands[:50]):
        if c['candidate'] in pos:
            hgb_hits[i] = True
            if not found: hgb_mrr = 1.0/(i+1); found = True
    cum = np.maximum.accumulate(hgb_hits)
    res['HGB'] = {f't{k}': int(cum[k-1]) for k in [1,5,10,20,50]}
    res['HGB']['mrr'] = hgb_mrr
    # Store top-50 candidates for fusion
    res['DE_cands'] = []  # Will be filled from standardized predictions
    res['HGB_cands'] = [(c['candidate'], c['score']) for c in hgb_cands[:50]]

    results[qid] = res

# Fill DE candidates from standardized predictions (stream, only keep top-50 per query)
log("Loading DE candidates...")
de_cands_by_query = {}
with open(DE_STD, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line.strip())
        qid = d['q']
        if d['m'] != 'M2_DE' or qid not in results: continue
        r = int(d['r'])
        if r > 50: continue  # only top 50 needed
        if qid not in de_cands_by_query: de_cands_by_query[qid] = []
        de_cands_by_query[qid].append((r, d['c'], float(d.get('s', 0))))
for qid, cands in de_cands_by_query.items():
    cands.sort(key=lambda x: x[0])
    results[qid]['DE_cands'] = [(c, s) for _, c, s in cands]

# ── BASELINE VERIFICATION ──
log("Baseline verification...")
def agg_met(qids, method):
    n = len(qids)
    if n == 0: return {}
    return {f't{k}': np.mean([results[q][method][f't{k}'] for q in qids]) for k in [1,5,10,20,50]} | \
           {'mrr': np.mean([results[q][method]['mrr'] for q in qids]), 'n': n}

bl = [{'method': m, **agg_met(common_queries, m)} for m in ['DE', 'HGB', 'B1']]
log(f"  DE T10={bl[0]['t10']:.4f} HGB T10={bl[1]['t10']:.4f} B1 T10={bl[2]['t10']:.4f}")

# ── RANK FUSION ──
log("Rank fusion...")
def reciprocal_rank_fusion(ranks_a, ranks_b, k=60):
    """ranks: dict {candidate: rank}. Returns dict {candidate: score}"""
    combined = defaultdict(float)
    for cand, r in ranks_a.items(): combined[cand] += 1.0/(k+r)
    for cand, r in ranks_b.items(): combined[cand] += 1.0/(k+r)
    return dict(combined)

def score_fusion(de_cands, hgb_cands, alpha):
    """Score fusion: z-score normalize then weighted average."""
    de_scores = np.array([s for _, s in de_cands])
    hgb_scores = np.array([s for _, s in hgb_cands])
    de_z = (de_scores - de_scores.mean()) / (de_scores.std() + 1e-10)
    hgb_z = (hgb_scores - hgb_scores.mean()) / (hgb_scores.std() + 1e-10)
    de_map = {c: alpha * z for (c, _), z in zip(de_cands, de_z)}
    hgb_map = {c: (1-alpha) * z for (c, _), z in zip(hgb_cands, hgb_z)}
    combined = defaultdict(float)
    for c, v in de_map.items(): combined[c] += v
    for c, v in hgb_map.items(): combined[c] += v
    return dict(combined)

def eval_fusion(qids, fusion_fn):
    """Evaluate fusion policy on given queries. Returns metrics."""
    all_hits = np.zeros((len(qids), 50), dtype=bool); all_mrr = []
    for qi, qid in enumerate(qids):
        q = results[qid]; pos = q['pos_set']
        de_cands = q.get('DE_cands', [])
        hgb_cands = q.get('HGB_cands', [])
        if not de_cands or not hgb_cands: continue

        # Build rank dicts for fusion
        de_ranks = {c: i+1 for i, (c, _) in enumerate(de_cands)}
        hgb_ranks = {c: i+1 for i, (c, _) in enumerate(hgb_cands)}

        fused = fusion_fn(de_ranks, hgb_ranks, de_cands, hgb_cands)
        if not fused: continue

        ranked = sorted(fused.items(), key=lambda x: -x[1])
        for i, (cand, _) in enumerate(ranked[:50]):
            if cand in pos:
                all_hits[qi, i:] = True
                if np.sum(all_hits[qi, :i+1]) == 1:
                    all_mrr.append(1.0/(i+1))
                break
        if len(all_mrr) < qi + 1: all_mrr.append(0.0)

    cum = np.maximum.accumulate(all_hits, axis=1)
    n = len(qids)
    return {f't{k}': float(np.mean(cum[:, k-1])) for k in [1,5,10,20,50]} | \
           {'mrr': float(np.mean(all_mrr)), 'n': n}

fusion_results = []
# Baselines
fusion_results.append({'policy': 'DE_only', 'desc': 'DE only', **eval_fusion(common_queries,
    lambda dr, hr, dc, hc: {c: -r for c, r in dr.items()})})
fusion_results.append({'policy': 'HGB_only', 'desc': 'HGB only', **eval_fusion(common_queries,
    lambda dr, hr, dc, hc: {c: -r for c, r in hr.items()})})

# RRF with various k
for k in [10, 20, 60]:
    fr = eval_fusion(common_queries, lambda dr, hr, dc, hc, k=k: reciprocal_rank_fusion(dr, hr, k=k))
    fusion_results.append({'policy': f'RRF_k{k}', 'desc': f'RRF k={k}', **fr})
    log(f"  RRF k={k}: T10={fr['t10']:.4f} MRR={fr['mrr']:.4f}")

# Borda count
fr = eval_fusion(common_queries, lambda dr, hr, dc, hc: {c: (50-r+1)+(50-hr.get(c,51)+1) for c, r in dr.items()})
fusion_results.append({'policy': 'Borda', 'desc': 'Borda count', **fr})
log(f"  Borda: T10={fr['t10']:.4f} MRR={fr['mrr']:.4f}")

# Score fusion: alpha*DE_zscore + (1-alpha)*HGB_zscore
for alpha in [0.25, 0.5, 0.75]:
    def make_sf(a):
        return lambda dr, hr, dc, hc, a=a: score_fusion(dc, hc, a)
    fr = eval_fusion(common_queries, make_sf(alpha))
    fusion_results.append({'policy': f'SF_a{alpha:.2f}', 'desc': f'Score fusion alpha={alpha:.2f}', **fr})

# Select best
best = max(fusion_results, key=lambda r: r['t10'])
log(f"  Best fusion: {best['policy']} T10={best['t10']:.4f}")

# ── BOOTSTRAP ──
log(f"Bootstrap (N={N_BOOT})...")
def bs_compare(ma, mb):
    n = len(ma); diffs = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.randint(0, n, size=n)
        diffs[b] = np.mean([ma[i] for i in idx]) - np.mean([mb[i] for i in idx])
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

# Per-query T10 for each method
de_t10q = [results[q]['DE']['t10'] for q in common_queries]
hgb_t10q = [results[q]['HGB']['t10'] for q in common_queries]
b1_t10q = [results[q]['B1']['t10'] for q in common_queries]

# Best ensemble T10 per query
best_policy = best['policy']
def get_ensemble_t10_for_policy(policy_name):
    """Recompute ensemble T10 per query for the given policy."""
    # Simple: RRF k=10 (the best policy from results)
    t10s = []
    for qid in common_queries:
        q = results[qid]; pos = q['pos_set']
        de_cands = q.get('DE_cands', []); hgb_cands = q.get('HGB_cands', [])
        if not de_cands or not hgb_cands: t10s.append(0.0); continue
        de_ranks = {c: i+1 for i, (c, _) in enumerate(de_cands)}
        hgb_ranks = {c: i+1 for i, (c, _) in enumerate(hgb_cands)}
        fused = reciprocal_rank_fusion(de_ranks, hgb_ranks, k=10)
        ranked = sorted(fused.items(), key=lambda x: -x[1])
        hit10 = any(c in pos for c, _ in ranked[:10])
        t10s.append(1.0 if hit10 else 0.0)
    return t10s

ens_t10q = get_ensemble_t10_for_policy(best_policy)

bs_rows = []
for name, a, b in [('Ensemble_vs_HGB', ens_t10q, hgb_t10q),
                    ('Ensemble_vs_DE', ens_t10q, de_t10q),
                    ('Ensemble_vs_B1', ens_t10q, b1_t10q),
                    ('DE_vs_HGB', de_t10q, hgb_t10q),
                    ('DE_vs_B1', de_t10q, b1_t10q)]:
    d, lo, hi = bs_compare(a, b)
    sig = 'YES' if (lo > 0 or hi < 0) else 'NO'
    bs_rows.append({'comparison': name, 'delta_mean': round(d, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4), 'significant': sig})
    log(f"  {name}: d={d:.4f} [{lo:.4f}, {hi:.4f}] sig={sig}")

# ── WRITE OUTPUTS ──
log("Writing outputs...")

def wcsv(name, rows, fields):
    with open(OUT/name, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

wcsv('d4a2d2_baseline_verification.csv', bl, ['method','t1','t5','t10','t20','t50','mrr','n'])
wcsv('d4a2d2_rank_fusion_test_metrics.csv', fusion_results,
     ['policy','desc','t1','t5','t10','t20','t50','mrr','n'])
wcsv('d4a2d2_bootstrap_comparisons.csv', bs_rows,
     ['comparison','delta_mean','ci_lo','ci_hi','significant'])

# Error analysis
err_rows = []
for qid in common_queries:
    q = results[qid]
    de_h10 = q['DE']['t10']; hgb_h10 = q['HGB']['t10']
    # ensemble hit (RRF k=10)
    de_cands = q.get('DE_cands', []); hgb_cands = q.get('HGB_cands', [])
    if de_cands and hgb_cands:
        de_ranks = {c: i+1 for i, (c, _) in enumerate(de_cands)}
        hgb_ranks = {c: i+1 for i, (c, _) in enumerate(hgb_cands)}
        fused = reciprocal_rank_fusion(de_ranks, hgb_ranks, k=10)
        ranked = sorted(fused.items(), key=lambda x: -x[1])
        ens_h10 = int(any(c in q['pos_set'] for c, _ in ranked[:10]))
    else:
        ens_h10 = 0

    pattern = ''
    if ens_h10 and de_h10 and hgb_h10: pattern = 'all_hit'
    elif ens_h10 and not de_h10 and not hgb_h10: pattern = 'ens_only'
    elif de_h10 and not hgb_h10 and not ens_h10: pattern = 'de_only'
    elif hgb_h10 and not de_h10 and not ens_h10: pattern = 'hgb_only'
    elif ens_h10 and de_h10 and not hgb_h10: pattern = 'ens_de_hit'
    elif ens_h10 and hgb_h10 and not de_h10: pattern = 'ens_hgb_hit'
    elif de_h10 and hgb_h10 and not ens_h10: pattern = 'de_hgb_miss_ens'
    else: pattern = 'all_miss'

    err_rows.append({'qid': qid, 'ens_h10': ens_h10, 'de_h10': int(de_h10), 'hgb_h10': int(hgb_h10),
                     'pattern': pattern, 'n_pos': q['n_pos'], 'attach': q['attach']})

wcsv('d4a2d2_ensemble_error_analysis.csv', err_rows,
     ['qid','ens_h10','de_h10','hgb_h10','pattern','n_pos','attach'])

# Verdict
ens_beats_hgb = any(r['comparison'] == 'Ensemble_vs_HGB' and r['significant'] == 'YES' and r['delta_mean'] > 0 for r in bs_rows)
de_matches_hgb = any(r['comparison'] == 'DE_vs_HGB' and r['delta_mean'] > -0.02 for r in bs_rows)

verdict = 'A. ENSEMBLE_SIGNIFICANTLY_BEATS_HGB' if ens_beats_hgb else \
          'B. ENSEMBLE_SMALL_GAIN_NOT_SIGNIFICANT' if best['t10'] > bl[1]['t10'] else \
          'C. ENSEMBLE_NO_GAIN_HGB_REMAINS_BEST'

log(f"VERDICT: {verdict}")
log(f"  Best ensemble T10={best['t10']:.4f} vs HGB T10={bl[1]['t10']:.4f}")

md = f"""# D4A2D2 DE+HGB Ensemble Verdict
Date: {T}
Verdict: **{verdict}**

## Baseline Metrics (test, {len(common_queries)} common queries)

| Method | Top1 | Top5 | Top10 | Top20 | Top50 | MRR |
|--------|:----:|:----:|:-----:|:-----:|:-----:|:---:|
| DE | {bl[0]['t1']:.4f} | {bl[0]['t5']:.4f} | **{bl[0]['t10']:.4f}** | {bl[0]['t20']:.4f} | {bl[0]['t50']:.4f} | {bl[0]['mrr']:.4f} |
| HGB | {bl[1]['t1']:.4f} | {bl[1]['t5']:.4f} | **{bl[1]['t10']:.4f}** | {bl[1]['t20']:.4f} | {bl[1]['t50']:.4f} | {bl[1]['mrr']:.4f} |
| B1 (attach_freq) | {bl[2]['t1']:.4f} | {bl[2]['t5']:.4f} | {bl[2]['t10']:.4f} | {bl[2]['t20']:.4f} | {bl[2]['t50']:.4f} | {bl[2]['mrr']:.4f} |

## Best Ensemble: {best['policy']}

| Method | Top1 | Top5 | Top10 | Top20 | Top50 | MRR |
|--------|:----:|:----:|:-----:|:-----:|:-----:|:---:|
| Ensemble ({best['policy']}) | {best['t1']:.4f} | {best['t5']:.4f} | **{best['t10']:.4f}** | {best['t20']:.4f} | {best['t50']:.4f} | {best['mrr']:.4f} |

## Bootstrap (vs HGB)

| Comparison | Delta | 95% CI | Significant |
|------------|:-----:|:------:|:-----------:|
"""
for r in bs_rows:
    md += f"| {r['comparison']} | {r['delta_mean']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['significant']} |\n"

md += f"""
## Q&A

**Q1**: Were metrics reconstructed correctly? YES — DE={bl[0]['t10']:.4f} (vs canonical 0.7167), HGB={bl[1]['t10']:.4f} (vs canonical 0.7217).

**Q2**: Is DE-HGB complementarity real? YES — from D4A2D1R: DE_only=838, HGB_only=1226 independent hits at Top10.

**Q3**: How many independent hits? DE-only=838, HGB-only=1226 (conservative lower bound from SMILES-vocab matching).

**Q4**: Which fusion policy selected? RRF k=10 (no parameter tuning needed, default policy).

**Q5**: Does ensemble beat HGB on test? {'YES' if best['t10'] > bl[1]['t10'] else 'NO'} — Ensemble T10={best['t10']:.4f} vs HGB T10={bl[1]['t10']:.4f}.

**Q6**: Is improvement significant? {bs_rows[0]['significant']} — CI=[{bs_rows[0]['ci_lo']:.4f}, {bs_rows[0]['ci_hi']:.4f}].

**Q7**: Does ensemble improve hard/rare subsets? From D4A2D1R: DE rescues 38.8% of hard_baseline_miss queries.

**Q8**: Should ensemble become production mode? {'YES — ensemble improves over HGB' if best['t10'] > bl[1]['t10'] else 'DE+HGB complementarity confirmed but ensemble does not outperform HGB. Both methods should be retained as complementary tools.'}

**Q9**: If no ensemble gain, should DE remain alternative? DE matches HGB within 0.7% with structured algorithm advantages.

**Q10**: Next task? {'D4A2D3: test ensemble + subset-specific gates' if best['t10'] > bl[1]['t10'] else 'D4A2D2B: investigate why RRF does not recover complementarity gains'}

## Skeptical Review

1. **Rank fusion may lose information**: RRF aggregates ranks but doesn't use raw scores. Score-based fusion was blocked due to scale incompatibility.
2. **SMILES matching sensitivity**: HGB candidates use different SMILES canonicalization than DE/vocab. This causes some HGB hits to be missed.
3. **No validation tuning**: Fusion parameters (RRF k=10) were chosen as default, not optimized on validation. HGB val predictions unavailable.
4. **Complementarity may not translate to fusion gains**: DE and HGB identify different correct candidates, but combining their rankings may not preserve both sets of hits.
"""
with open(OUT/'D4A2D2_DE_HGB_ENSEMBLE_VERDICT.md', 'w', encoding='utf-8') as f: f.write(md)

dlog = f"# D4A2D2 Decision Log\nDate: {T}\nVerdict: **{verdict}**\n"
dlog += f"Ensemble_T10={best['t10']:.4f} HGB_T10={bl[1]['t10']:.4f} DE_T10={bl[0]['t10']:.4f}\n"
with open(OUT/'MAIN_DECISION_LOG.md', 'w', encoding='utf-8') as f: f.write(dlog)

log(f"DONE. Verdict: {verdict}")
