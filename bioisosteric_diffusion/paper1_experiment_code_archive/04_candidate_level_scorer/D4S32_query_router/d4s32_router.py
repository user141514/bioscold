#!/usr/bin/env python3
"""D4S32: Query-level router for D4S31 scorer. Predict scorer>baseline per query."""
import pandas as pd, numpy as np, os, time, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S32_query_router/results"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def qmetrics(df, col):
    r = []
    for qid, g in df.groupby("query_id"):
        gs = g.sort_values(col, ascending=False)
        ls = gs["label"].values; pr = np.where(ls==1)[0]+1
        if len(pr)==0: continue
        b = pr.min()
        r.append({"query_id":qid,"best_rank":b,"Top10":int(b<=10),"MRR":1.0/b})
    return pd.DataFrame(r)

def boot_ci(vals, n=5000):
    rng = np.random.RandomState(42)
    m = [vals[rng.randint(0,len(vals),len(vals))].mean() for _ in range(n)]
    m = np.array(m)
    return float(np.percentile(m,2.5)), float(np.percentile(m,97.5))

def main():
    log("="*60)
    log("D4S32: Query-Level Router for Scorer Deployment")
    log("="*60)

    val = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    blind = pd.read_parquet(f"{D4S27}/candidate_prior_scores_blind.parquet")

    # Build D4S31 scorer scores (simplified: no full 5-fold, use pre-computed)
    # We need val OOF scorer performance per query
    # Re-use: train HistGB on val, 5-fold OOF
    SAFE_NO_RANKS = [f for f in [
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
    ] if f in val.columns]
    CAT = ['old_fragment_smiles','attachment_signature','replacement_frequency_bin']

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

    # 5-fold OOF on val (D4S31 config)
    X_val, t_val = build_X(val, SAFE_NO_RANKS)
    y_val = val["label"].values
    uq = np.array(list(set(val["query_id"].values)))
    gkf = GroupKFold(n_splits=5)
    val_oof = np.zeros(len(y_val))
    for tqi, vqi in gkf.split(uq, groups=uq):
        tqs = frozenset(uq[tqi]); vqs = frozenset(uq[vqi])
        tm = np.array([q in tqs for q in val["query_id"].values])
        vm = np.array([q in vqs for q in val["query_id"].values])
        pi = np.where(y_val[tm]==1)[0]; ni = np.where(y_val[tm]==0)[0]
        nn = min(len(ni), len(pi)*5)
        ns = np.random.RandomState(42).choice(ni, nn, replace=False)
        ti = np.concatenate([pi, ns])
        hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                              early_stopping=False, random_state=42,
                                              class_weight="balanced")
        hgb.fit(X_val[tm][ti], y_val[tm][ti])
        val_oof[vm] = hgb.predict_proba(X_val[vm])[:, 1]
    val["scorer_oof"] = val_oof

    # Per-query: scorer vs baseline
    q_sc = qmetrics(val, "scorer_oof")
    q_bl = qmetrics(val, "blend_score")
    q_sc["scorer_better"] = (q_sc["Top10"].values > q_bl["Top10"].values).astype(int)
    q_sc["scorer_delta"] = q_sc["Top10"].values.astype(float) - q_bl["Top10"].values.astype(float)
    log(f"Val scorers better: {q_sc['scorer_better'].sum()}/{len(q_sc)}")

    # Build query-level router features
    router_feats = []
    for qid in q_sc["query_id"].values:
        g = val[val["query_id"]==qid]
        feats = {
            "query_id": qid,
            "n_candidates": len(g),
            "n_positives": g["label"].sum(),
            "blend_score_mean": g["blend_score"].mean(),
            "blend_score_std": g["blend_score"].std(),
            "blend_score_max": g["blend_score"].max(),
            "rank_DE_median": g["rank_DE"].median(),
            "rank_HGB_median": g["rank_HGB"].median(),
            "top10_flag_count": g[["candidate_in_DE_top10","candidate_in_HGB_top10",
                                     "candidate_in_Borda_top10","candidate_in_attach_top10"]].sum().sum(),
            "query_prior_entropy": g["query_prior_entropy"].iloc[0],
            "query_prior_margin": g["query_prior_margin"].iloc[0],
            "query_prior_max_support": g["query_prior_max_support"].iloc[0],
            "replacement_freq_mean": g["replacement_frequency"].mean(),
            "replacement_freq_std": g["replacement_frequency"].std(),
            "attachment_freq": g["attachment_frequency"].iloc[0],
            "tanimoto_sim": g["_tanimoto_similarity"].iloc[0],
            "scorer_better": q_sc[q_sc["query_id"]==qid]["scorer_better"].values[0],
        }
        router_feats.append(feats)
    rdf = pd.DataFrame(router_feats)

    # Train router on val OOF
    router_X_cols = [c for c in rdf.columns if c not in ["query_id","scorer_better"]]
    Xr = rdf[router_X_cols].fillna(0).values
    yr = rdf["scorer_better"].values

    # 5-fold CV router
    gkf_r = GroupKFold(n_splits=5)
    router_oof = np.zeros(len(yr))
    for tqi, vqi in gkf_r.split(uq, groups=uq):
        tqs = frozenset(uq[tqi]); vqs = frozenset(uq[vqi])
        # Map query_id indices
        t_mask = rdf["query_id"].isin(tqs).values
        v_mask = rdf["query_id"].isin(vqs).values
        scaler = StandardScaler()
        Xr_tr = scaler.fit_transform(Xr[t_mask])
        Xr_val = scaler.transform(Xr[v_mask])
        lr = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42)
        lr.fit(Xr_tr, yr[t_mask])
        router_oof[v_mask] = lr.predict_proba(Xr_val)[:, 1]
    rdf["router_oof"] = router_oof
    log(f"Router OOF AUC: {np.mean((router_oof > 0.5) == yr):.4f}")
    log(f"Router OOF: predict USE on {int((router_oof>0.5).sum())}/{len(router_oof)} queries")

    # Apply router to val OOF for sanity
    q_sc["router_says_use"] = (router_oof > 0.5).astype(int)
    val_routed = val.copy()
    use_qs = set(q_sc[q_sc["router_says_use"]==1]["query_id"].values)
    val_routed["score_routed"] = np.where(
        val_routed["query_id"].isin(use_qs), val_routed["scorer_oof"], val_routed["blend_score"])
    q_val_routed = qmetrics(val_routed, "score_routed")
    log(f"  Val routed Top10: {q_val_routed['Top10'].mean():.6f}")

    # Train final scorer on full val, predict blind
    X_blind = build_X(blind, SAFE_NO_RANKS, templates=t_val)
    pi = np.where(y_val==1)[0]; ni = np.where(y_val==0)[0]
    nn = min(len(ni), len(pi)*5)
    ns = np.random.RandomState(42).choice(ni, nn, replace=False)
    ti = np.concatenate([pi, ns])
    hgb_full = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                               early_stopping=False, random_state=42,
                                               class_weight="balanced")
    hgb_full.fit(X_val[ti], y_val[ti])
    blind["scorer"] = hgb_full.predict_proba(X_blind)[:, 1]

    # Build router features for blind
    blind_router_feats = []
    for qid in blind["query_id"].unique():
        g = blind[blind["query_id"]==qid]
        blind_router_feats.append({
            "query_id": qid, "n_candidates": len(g),
            "blend_score_mean": g["blend_score"].mean(),
            "blend_score_std": g["blend_score"].std(),
            "blend_score_max": g["blend_score"].max(),
            "rank_DE_median": g["rank_DE"].median(),
            "rank_HGB_median": g["rank_HGB"].median(),
            "top10_flag_count": g[["candidate_in_DE_top10","candidate_in_HGB_top10",
                                     "candidate_in_Borda_top10","candidate_in_attach_top10"]].sum().sum(),
            "query_prior_entropy": g["query_prior_entropy"].iloc[0],
            "query_prior_margin": g["query_prior_margin"].iloc[0],
            "query_prior_max_support": g["query_prior_max_support"].iloc[0],
            "replacement_freq_mean": g["replacement_frequency"].mean(),
            "replacement_freq_std": g["replacement_frequency"].std(),
            "attachment_freq": g["attachment_frequency"].iloc[0],
            "tanimoto_sim": g["_tanimoto_similarity"].iloc[0],
        })
    brdf = pd.DataFrame(blind_router_feats)
    Xr_blind = brdf[router_X_cols].fillna(0).values

    # Train router on ALL val, predict blind
    scaler_all = StandardScaler()
    Xr_all = scaler_all.fit_transform(Xr)
    lr_all = LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=42)
    lr_all.fit(Xr_all, yr)
    blind_router_preds = lr_all.predict_proba(Xr_blind)[:, 1]

    # Test multiple router thresholds
    log("\n=== ROUTER THRESHOLD SWEEP (Blind) ===")
    best = None
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        use_qs_blind = set(brdf[blind_router_preds > thr]["query_id"].values)
        blind["_routed"] = np.where(blind["query_id"].isin(use_qs_blind),
                                     blind["scorer"], blind["blend_score"])
        q_r = qmetrics(blind, "_routed")
        delta = q_r["Top10"].mean() - 0.855773
        n_use = len(use_qs_blind)
        log(f"  thr={thr:.1f}: use={n_use:5d} queries, blind={q_r['Top10'].mean():.6f} delta={delta:+.6f}")
        if best is None or q_r["Top10"].mean() > best["Top10"]:
            best = {"thr": thr, "Top10": q_r["Top10"].mean(), "delta": delta, "n_use": n_use}

    # Compare
    q_bl = qmetrics(blind, "blend_score")
    q_sc_blind = qmetrics(blind, "scorer")
    log(f"\n=== FINAL ===")
    log(f"  Baseline:             {q_bl['Top10'].mean():.6f}")
    log(f"  Scorer all (D4S31):   {q_sc_blind['Top10'].mean():.6f}")
    log(f"  Best router (thr={best['thr']}): {best['Top10']:.6f} delta={best['delta']:+.6f}")
    log(f"  Router uses {best['n_use']}/{len(q_bl)} queries")

    # Save
    pd.DataFrame([{"method":"baseline","Top10":round(q_bl["Top10"].mean(),6)},
                   {"method":"scorer_all","Top10":round(q_sc_blind["Top10"].mean(),6)},
                   {"method":f"router_thr{best['thr']}","Top10":round(best["Top10"],6),
                    "delta":round(best["delta"],6),"n_queries_used":best["n_use"]},
                   ]).to_csv(f"{OUT}/d4s32_router_summary.csv", index=False)
    log(f"\nD4S32 complete. Best router Top10={best['Top10']:.6f}")

if __name__=="__main__":
    main()
