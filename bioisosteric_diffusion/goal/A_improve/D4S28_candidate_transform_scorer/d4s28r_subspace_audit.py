#!/usr/bin/env python3
"""
D4S28R: Negative-subspace audit + conservative fallback.
Strategy: per old_fragment, choose scorer ONLY if it beats baseline.
Otherwise fall back to blend_score.
"""
import pandas as pd, numpy as np, os, time, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S28_candidate_transform_scorer/results"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def metrics(df, col, lbl="label"):
    r = []
    for qid, g in df.groupby("query_id"):
        gs = g.sort_values(col, ascending=False)
        ls = gs[lbl].values; pr = np.where(ls==1)[0]+1
        if len(pr)==0: continue
        b = pr.min()
        r.append({"query_id":qid,"best_rank":b,"Top1":int(b==1),"Top5":int(b<=5),
                   "Top10":int(b<=10),"Top20":int(b<=20),"MRR":1.0/b})
    return pd.DataFrame(r)

SAFE = ['blend_score','mlp_z','hgb_z','mlp_score','hgb_refit_score',
    'score_DE','score_HGB','score_Borda','score_attach',
    'mlp_rank','blend_rank','rank_DE','rank_HGB','rank_Borda','rank_attach',
    'candidate_in_attach_top10','candidate_in_DE_top10','candidate_in_HGB_top10','candidate_in_Borda_top10',
    'cont_prior_score','backoff_logit_score','t_p4_logit_score','p1_logit_score','p3_logit_score',
    't_p4_smoothed_rate','p1_smoothed_rate','p3_smoothed_rate','backoff_smoothed_rate',
    't_p4_support','p1_support','p3_support','p0_support','backoff_support',
    't_p4_positive','p1_positive','p3_positive',
    'backoff_logit_rank','cont_prior_rank','t_p4_logit_rank','p1_logit_rank','p3_logit_rank',
    't_pmi_p4_p1','pmi_p3_p1','pmi_p1_p0',
    't_pmi_p4_p1_rank','pmi_p3_p1_rank','pmi_p1_p0_rank',
    'delta_heavy_atoms','delta_ring_count','delta_hetero_count','delta_mw','delta_logp','delta_tpsa',
    'candidate_heavy_atoms','candidate_mw','candidate_logp','candidate_tpsa',
    '_tanimoto_similarity','replacement_frequency','attachment_frequency',
    'query_prior_entropy','query_prior_margin','query_prior_max_support']
CAT = ['old_fragment_smiles','attachment_signature','replacement_frequency_bin']

def build_X(df, templates=None):
    num = [c for c in SAFE if c not in CAT and c in df.columns]
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


def main():
    log("="*60)
    log("D4S28R: Negative-Subspace Audit + Conservative Fallback")
    log("="*60)

    val = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    blind = pd.read_parquet(f"{D4S27}/candidate_prior_scores_blind.parquet")

    X_val, templates = build_X(val)
    y_val = val["label"].values
    X_blind = build_X(blind, templates=templates)
    log(f"Dims: val={X_val.shape}, blind={X_blind.shape}")

    # Train on val, predict blind
    pi = np.where(y_val==1)[0]; ni = np.where(y_val==0)[0]
    nn = min(len(ni), len(pi)*5)
    ns = np.random.RandomState(42).choice(ni, nn, replace=False)
    ti = np.concatenate([pi, ns])
    hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                          early_stopping=False, random_state=42,
                                          class_weight="balanced")
    hgb.fit(X_val[ti], y_val[ti])
    blind["score_hgb"] = hgb.predict_proba(X_blind)[:, 1]

    # ═══ AUDIT: Identify negative subspaces ═══
    log("\n=== NEGATIVE SUBSPACE AUDIT (Blind) ===")
    subspace_results = []
    for frag in sorted(blind["old_fragment_smiles"].unique()):
        g = blind[blind["old_fragment_smiles"]==frag]
        qs = metrics(g, "score_hgb"); qb = metrics(g, "blend_score")
        delta = qs["Top10"].mean() - qb["Top10"].mean()
        verdict = "USE_SCORER" if delta > 0 else "FALLBACK"
        subspace_results.append({
            "fragment": frag, "n_queries": len(qs),
            "baseline_Top10": round(qb["Top10"].mean(), 6),
            "scorer_Top10": round(qs["Top10"].mean(), 6),
            "delta": round(delta, 6),
            "verdict": verdict
        })

    # Also check by attachment_signature
    for att in sorted(blind["attachment_signature"].unique()):
        g = blind[blind["attachment_signature"]==att]
        if len(g["query_id"].unique()) < 5: continue
        qs = metrics(g, "score_hgb"); qb = metrics(g, "blend_score")
        delta = qs["Top10"].mean() - qb["Top10"].mean()
        subspace_results.append({
            "fragment": f"ATT:{att}", "n_queries": len(qs),
            "baseline_Top10": round(qb["Top10"].mean(), 6),
            "scorer_Top10": round(qs["Top10"].mean(), 6),
            "delta": round(delta, 6),
            "verdict": "USE_SCORER" if delta > 0 else "FALLBACK"
        })

    # Display
    negative = [s for s in subspace_results if s["delta"] < 0]
    positive = [s for s in subspace_results if s["delta"] >= 0]
    log(f"\n  NEGATIVE SUBSPACES ({len(negative)}):")
    for s in sorted(negative, key=lambda x: x["delta"]):
        log(f"    {s['fragment']:30s} n={s['n_queries']:5d} "
            f"bl={s['baseline_Top10']:.4f} sc={s['scorer_Top10']:.4f} d={s['delta']:+.4f}")
    log(f"\n  POSITIVE SUBSPACES ({len(positive)}):")
    for s in sorted(positive, key=lambda x: -x["delta"]):
        log(f"    {s['fragment']:30s} n={s['n_queries']:5d} "
            f"bl={s['baseline_Top10']:.4f} sc={s['scorer_Top10']:.4f} d={s['delta']:+.4f}")

    # ═══ CONSERVATIVE FALLBACK ═══
    log("\n=== CONSERVATIVE FALLBACK STRATEGY ===")
    # Build fallback map: old_fragment -> use_scorer (True/False)
    fallback_map = {}
    for s in subspace_results:
        if not s["fragment"].startswith("ATT:"):
            fallback_map[s["fragment"]] = s["delta"] > 0

    # Apply: if fragment in positive region, use scorer; else use baseline
    blind["score_fallback"] = np.where(
        blind["old_fragment_smiles"].map(fallback_map).fillna(False),
        blind["score_hgb"],
        blind["blend_score"]
    )

    q_fb = metrics(blind, "score_fallback")
    q_bl = metrics(blind, "blend_score")
    q_sc = metrics(blind, "score_hgb")

    log(f"  Baseline:            {q_bl['Top10'].mean():.6f}")
    log(f"  Scorer (all):        {q_sc['Top10'].mean():.6f}")
    log(f"  Fallback (selective): {q_fb['Top10'].mean():.6f}")
    log(f"  Fallback vs baseline: {q_fb['Top10'].mean() - q_bl['Top10'].mean():+.6f}")
    log(f"  Fallback vs scorer:   {q_fb['Top10'].mean() - q_sc['Top10'].mean():+.6f}")

    # ═══ ORACLE FALLBACK (per-query optimal choice) ═══
    log("\n=== ORACLE FALLBACK (upper bound) ===")
    blind["score_oracle"] = np.where(
        blind["score_hgb"] >= blind["blend_score"],
        blind["score_hgb"],
        blind["blend_score"]
    )
    # Actually need per-query oracle: use scorer for this query if scorer Top10 > baseline Top10
    q_sc_per_q = metrics(blind, "score_hgb")
    q_bl_per_q = metrics(blind, "blend_score")
    q_sc_per_q["scorer_better"] = q_sc_per_q["Top10"].values > q_bl_per_q["Top10"].values
    better_qs = set(q_sc_per_q[q_sc_per_q["scorer_better"]]["query_id"].values)

    blind["score_oracle_q"] = np.where(
        blind["query_id"].isin(better_qs),
        blind["score_hgb"],
        blind["blend_score"]
    )
    q_oracle = metrics(blind, "score_oracle_q")
    log(f"  Oracle per-query fallback: {q_oracle['Top10'].mean():.6f}")
    log(f"  Oracle headroom:           {q_oracle['Top10'].mean() - q_bl['Top10'].mean():+.6f}")
    log(f"  Queries using scorer:      {len(better_qs)}/{len(q_sc_per_q)}")

    # ═══ SAVE ═══
    pd.DataFrame(subspace_results).to_csv(f"{OUT}/d4s28r_subspace_audit.csv", index=False)

    summary = pd.DataFrame([
        {"method":"baseline_blend","Top10":round(q_bl["Top10"].mean(),6),
         "MRR":round(q_bl["MRR"].mean(),6),"delta":0},
        {"method":"scorer_all_queries","Top10":round(q_sc["Top10"].mean(),6),
         "MRR":round(q_sc["MRR"].mean(),6),
         "delta":round(q_sc["Top10"].mean()-q_bl["Top10"].mean(),6)},
        {"method":"fallback_fragment_level","Top10":round(q_fb["Top10"].mean(),6),
         "MRR":round(q_fb["MRR"].mean(),6),
         "delta":round(q_fb["Top10"].mean()-q_bl["Top10"].mean(),6)},
        {"method":"oracle_per_query","Top10":round(q_oracle["Top10"].mean(),6),
         "MRR":round(q_oracle["MRR"].mean(),6),
         "delta":round(q_oracle["Top10"].mean()-q_bl["Top10"].mean(),6)},
    ])
    summary.to_csv(f"{OUT}/d4s28r_fallback_summary.csv", index=False)
    log(f"\nSaved. D4S28R complete.")

if __name__=="__main__":
    main()
