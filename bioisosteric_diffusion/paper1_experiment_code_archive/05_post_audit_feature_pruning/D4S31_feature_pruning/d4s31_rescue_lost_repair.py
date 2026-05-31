#!/usr/bin/env python3
"""GAP-1 repair: rescue/lost with query_id merge (not positional assignment)."""
import pandas as pd, numpy as np, os, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S31_feature_pruning/results"

CAT = ['old_fragment_smiles','attachment_signature','replacement_frequency_bin']
SAFE_NO_PRIOR_RANKS = [
    'blend_score','mlp_z','hgb_z','mlp_score','hgb_refit_score',
    'score_DE','score_HGB','score_Borda','score_attach',
    'mlp_rank','blend_rank','rank_DE','rank_HGB','rank_Borda','rank_attach',
    'candidate_in_attach_top10','candidate_in_DE_top10','candidate_in_HGB_top10','candidate_in_Borda_top10',
    'cont_prior_score','backoff_logit_score','t_p4_logit_score','p1_logit_score','p3_logit_score',
    't_p4_smoothed_rate','p1_smoothed_rate','p3_smoothed_rate','backoff_smoothed_rate',
    't_p4_support','p1_support','p3_support','p0_support','backoff_support',
    't_p4_positive','p1_positive','p3_positive',
    't_pmi_p4_p1','pmi_p3_p1','pmi_p1_p0',
    't_pmi_p4_p1_rank','pmi_p3_p1_rank','pmi_p1_p0_rank',
    'delta_heavy_atoms','delta_ring_count','delta_hetero_count','delta_mw','delta_logp','delta_tpsa',
    'candidate_heavy_atoms','candidate_mw','candidate_logp','candidate_tpsa',
    '_tanimoto_similarity','replacement_frequency','attachment_frequency',
    'query_prior_entropy','query_prior_margin','query_prior_max_support',
]

def build_X(df, feats, templates=None):
    num = [c for c in feats if c not in CAT and c in df.columns]
    Xp = [df[num].fillna(0).values.astype(np.float64)]
    tmpl = {}
    for cat in CAT:
        if cat not in df.columns: continue
        mapped = df[cat].copy()
        if templates and cat in templates:
            known = templates[cat]
            mapped = mapped.apply(lambda x: x if x in known else 'OTHER')
        else:
            known = set(mapped.unique())
        dummies = pd.get_dummies(mapped, prefix=cat)
        tcols = [f"{cat}_{v}" for v in known] + [f"{cat}_OTHER"]
        dummies = dummies.reindex(columns=tcols, fill_value=0)
        Xp.append(dummies.values.astype(np.float64))
        tmpl[cat] = known
    X = np.hstack(Xp)
    return X if templates is not None else (X, tmpl)

def qmetrics(df, col):
    """Per-query metrics. Returns DataFrame with query_id, best_rank, Top10, MRR."""
    rows = []
    for qid, g in df.groupby("query_id"):
        gs = g.sort_values(col, ascending=False)
        ls = gs["label"].values
        pr = np.where(ls == 1)[0] + 1
        if len(pr) == 0: continue
        b = pr.min()
        rows.append({"query_id": qid, "best_rank": int(b),
                     "Top10": int(b <= 10), "MRR": 1.0 / b})
    return pd.DataFrame(rows)

def rescue_lost_merged(df, model_col, ref_col):
    """Compute rescue/lost using query_id merge — NOT positional assignment."""
    q_model = qmetrics(df, model_col)
    q_ref = qmetrics(df, ref_col)
    # Merge on query_id — only queries present in BOTH
    merged = q_model[["query_id", "Top10"]].merge(
        q_ref[["query_id", "Top10"]], on="query_id",
        suffixes=("_model", "_ref")
    )
    n_common = len(merged)
    ref_hits = int(merged["Top10_ref"].sum())
    ref_misses = n_common - ref_hits
    model_hits = int(merged["Top10_model"].sum())
    rescue = int(((merged["Top10_ref"] == 0) & (merged["Top10_model"] == 1)).sum())
    lost = int(((merged["Top10_ref"] == 1) & (merged["Top10_model"] == 0)).sum())
    net = rescue - lost
    expected = ref_hits + rescue - lost
    arith_pass = (expected == model_hits)
    return {
        "n_common": n_common, "ref_hits": ref_hits, "ref_misses": ref_misses,
        "model_hits": model_hits, "rescue": rescue, "lost": lost, "net": net,
        "expected_model_hits": expected, "arithmetic_pass": arith_pass
    }

print("Loading data...")
val = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
blind = pd.read_parquet(f"{D4S27}/candidate_prior_scores_blind.parquet")
y_val = val["label"].values

X_val, t = build_X(val, SAFE_NO_PRIOR_RANKS)
X_blind = build_X(blind, SAFE_NO_PRIOR_RANKS, templates=t)
print(f"Dims: val={X_val.shape}, blind={X_blind.shape}")

# Train D4S31 on full val
print("Training D4S31...")
pi = np.where(y_val == 1)[0]; ni = np.where(y_val == 0)[0]
nn = min(len(ni), len(pi) * 5)
ns = np.random.RandomState(42).choice(ni, nn, replace=False)
ti = np.concatenate([pi, ns])

hgb = HistGradientBoostingClassifier(
    max_iter=200, max_depth=6, learning_rate=0.1,
    early_stopping=False, random_state=42, class_weight="balanced"
)
hgb.fit(X_val[ti], y_val[ti])
blind["_d4s31"] = hgb.predict_proba(X_blind)[:, 1]
print("D4S31 trained.")

# Reference columns in blind data
# We need D4S28R scores. Check if available, else compute from base rankers.
# D4S28R = 82-feature scorer. We only have 77-feature SAFE set here.
# For this analysis, use available reference score columns.
# Score columns available: blend_score, score_DE, score_HGB, score_Borda, score_attach

references = {
    "ScoreBlend": "blend_score",
    "DE": "score_DE",
    "HGB": "score_HGB",
    "Borda": "score_Borda",
    "AttFreq": "score_attach",
}

# Also compute Borda from DE+HGB ranks directly
# Borda: rank by score_DE + rank by score_HGB
blind_q = blind.copy()
# Compute per-query ranks for DE and HGB
blind_q["_rank_DE"] = blind_q.groupby("query_id")["score_DE"].rank(ascending=False, method="first")
blind_q["_rank_HGB"] = blind_q.groupby("query_id")["score_HGB"].rank(ascending=False, method="first")
blind_q["_borda_direct"] = blind_q.groupby("query_id")["_rank_DE"].transform(lambda x: x.max() + 1 - x) + \
                            blind_q.groupby("query_id")["_rank_HGB"].transform(lambda x: x.max() + 1 - x)
bl_cols = list(blind.columns)

# Check if score_Borda exists
if "score_Borda" not in blind.columns:
    print("WARNING: score_Borda not in blind columns. Using direct Borda computation.")
    references["Borda(direct)"] = "_borda_direct"

# Actually, let's check what columns exist
score_cols = [c for c in blind.columns if 'score' in c.lower() or 'borda' in c.lower()]
print(f"Score columns in blind: {score_cols}")

# Also try: compute best-base diagnostic = per-query best of DE/HGB/AttFreq
# For this we need per-query hits, not per-candidate scores
# Best-base: for each query, does at least one of DE/HGB/AttFreq have correct in top10?
# This requires query-level metrics, already handled by rescue_lost_merged per-reference

print("\n" + "=" * 80)
print("RESCUE/LOST WITH QUERY_ID MERGE (no positional assignment)")
print("=" * 80)

ref_cols_available = {}
for name, col in references.items():
    if col in blind.columns:
        ref_cols_available[name] = col

# Also compute Oracle(DE,HGB) diagnostic: per-query, at least one of DE or HGB has correct in top10
# This is a query-level diagnostic, not a score column
q_de = qmetrics(blind, "score_DE")
q_hgb = qmetrics(blind, "score_HGB")
q_oracle = q_de[["query_id"]].copy()
q_oracle["Top10"] = ((q_de["Top10"].values == 1) | (q_hgb["Top10"].values == 1)).astype(int)
# Rename to avoid collision
q_oracle_col = "Top10_oracle"

# For best-base (DE+HGB+AttFreq)
q_att = qmetrics(blind, "score_attach")
q_bestbase = q_de[["query_id"]].copy()
q_bestbase["Top10"] = ((q_de["Top10"].values == 1) | (q_hgb["Top10"].values == 1) | (q_att["Top10"].values == 1)).astype(int)

results = []

for ref_name, ref_col in ref_cols_available.items():
    r = rescue_lost_merged(blind, "_d4s31", ref_col)
    r["reference"] = ref_name
    results.append(r)
    status = "PASS" if r["arithmetic_pass"] else "FAIL"
    print(f"\n{ref_name}: ref_hits={r['ref_hits']} model_hits={r['model_hits']} "
          f"rescue={r['rescue']} lost={r['lost']} net={r['net']} "
          f"expected={r['expected_model_hits']} arith={status} "
          f"n_common={r['n_common']}")

# Oracle(DE,HGB) diagnostic
q_d4s31 = qmetrics(blind, "_d4s31")
m_oracle = q_d4s31[["query_id","Top10"]].merge(
    q_oracle[["query_id","Top10"]], on="query_id", suffixes=("_model","_ref")
)
n_common = len(m_oracle)
ref_hits = int(m_oracle["Top10_ref"].sum())
model_hits = int(m_oracle["Top10_model"].sum())
rescue = int(((m_oracle["Top10_ref"]==0)&(m_oracle["Top10_model"]==1)).sum())
lost = int(((m_oracle["Top10_ref"]==1)&(m_oracle["Top10_model"]==0)).sum())
net = rescue - lost
expected = ref_hits + rescue - lost
arith_pass = (expected == model_hits)
status = "PASS" if arith_pass else "FAIL"
print(f"\nOracle(DE,HGB): ref_hits={ref_hits} model_hits={model_hits} "
      f"rescue={rescue} lost={lost} net={net} "
      f"expected={expected} arith={status} n_common={n_common}")
results.append({"reference":"Oracle(DE,HGB)","n_common":n_common,"ref_hits":ref_hits,
                "model_hits":model_hits,"rescue":rescue,"lost":lost,"net":net,
                "expected_model_hits":expected,"arithmetic_pass":arith_pass})

# Best-base (DE+HGB+AttFreq)
m_bb = q_d4s31[["query_id","Top10"]].merge(
    q_bestbase[["query_id","Top10"]], on="query_id", suffixes=("_model","_ref")
)
n_common = len(m_bb)
ref_hits = int(m_bb["Top10_ref"].sum())
model_hits = int(m_bb["Top10_model"].sum())
rescue = int(((m_bb["Top10_ref"]==0)&(m_bb["Top10_model"]==1)).sum())
lost = int(((m_bb["Top10_ref"]==1)&(m_bb["Top10_model"]==0)).sum())
net = rescue - lost
expected = ref_hits + rescue - lost
arith_pass = (expected == model_hits)
status = "PASS" if arith_pass else "FAIL"
print(f"\nBestBase(DE+HGB+AttFreq): ref_hits={ref_hits} model_hits={model_hits} "
      f"rescue={rescue} lost={lost} net={net} "
      f"expected={expected} arith={status} n_common={n_common}")
results.append({"reference":"BestBase(DE+HGB+AttFreq)","n_common":n_common,"ref_hits":ref_hits,
                "model_hits":model_hits,"rescue":rescue,"lost":lost,"net":net,
                "expected_model_hits":expected,"arithmetic_pass":arith_pass})

# Save
df_r = pd.DataFrame(results)[["reference","n_common","ref_hits","model_hits",
                               "rescue","lost","net","expected_model_hits","arithmetic_pass"]]
df_r.to_csv(f"{OUT}/d4s31_rescue_lost_merged.csv", index=False)
print(f"\nSaved to {OUT}/d4s31_rescue_lost_merged.csv")

# Final verdict
all_pass = all(df_r["arithmetic_pass"])
print(f"\n{'='*40}")
print(f"ALL ARITHMETIC PASS: {all_pass}")
if not all_pass:
    fails = df_r[~df_r["arithmetic_pass"]]["reference"].tolist()
    print(f"FAILURES: {fails}")
print(f"{'='*40}")
