#!/usr/bin/env python3
"""D4S31 Lockdown: drop prior_ranks, best hyperparams, bootstrap CI."""
import pandas as pd, numpy as np, os, time, json, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S31_feature_pruning/results"
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

CAT = ['old_fragment_smiles','attachment_signature','replacement_frequency_bin']
# Drop prior_ranks from safe set
SAFE_NO_PRIOR_RANKS = [
    'blend_score','mlp_z','hgb_z','mlp_score','hgb_refit_score',
    'score_DE','score_HGB','score_Borda','score_attach',
    'mlp_rank','blend_rank','rank_DE','rank_HGB','rank_Borda','rank_attach',
    'candidate_in_attach_top10','candidate_in_DE_top10','candidate_in_HGB_top10','candidate_in_Borda_top10',
    'cont_prior_score','backoff_logit_score','t_p4_logit_score','p1_logit_score','p3_logit_score',
    't_p4_smoothed_rate','p1_smoothed_rate','p3_smoothed_rate','backoff_smoothed_rate',
    't_p4_support','p1_support','p3_support','p0_support','backoff_support',
    't_p4_positive','p1_positive','p3_positive',
    # prior_ranks DROPPED
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


def main():
    log("="*60)
    log("D4S31 LOCKDOWN: drop prior_ranks + best hyperparams")
    log("="*60)

    val = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    blind = pd.read_parquet(f"{D4S27}/candidate_prior_scores_blind.parquet")
    y_val = val["label"].values

    X_val, t = build_X(val, SAFE_NO_PRIOR_RANKS)
    X_blind = build_X(blind, SAFE_NO_PRIOR_RANKS, templates=t)
    log(f"Dims: val={X_val.shape}, blind={X_blind.shape}")

    # 5-fold OOF
    log("\n--- Val 5-fold OOF ---")
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
    val["score_oof"] = val_oof
    q_val = qmetrics(val, "score_oof")
    q_val_bl = qmetrics(val, "blend_score")
    log(f"  Val OOF Top10: {q_val['Top10'].mean():.6f} (baseline {q_val_bl['Top10'].mean():.6f})")

    # Blind one-shot with best configs
    log("\n--- Blind One-Shot ---")
    pi = np.where(y_val==1)[0]; ni = np.where(y_val==0)[0]
    nn = min(len(ni), len(pi)*5)
    ns = np.random.RandomState(42).choice(ni, nn, replace=False)
    ti = np.concatenate([pi, ns])

    configs = [
        ("default", 200, 6, 0.10),
        ("best_found", 100, 4, 0.10),
        ("deep_conservative", 100, 4, 0.05),
    ]
    results = []
    for name, mi, md, lr in configs:
        hgb = HistGradientBoostingClassifier(max_iter=mi, max_depth=md, learning_rate=lr,
                                              early_stopping=False, random_state=42,
                                              class_weight="balanced")
        hgb.fit(X_val[ti], y_val[ti])
        blind["_s"] = hgb.predict_proba(X_blind)[:, 1]
        q = qmetrics(blind, "_s")
        delta = q["Top10"].mean() - 0.855773
        ci_lo, ci_hi = boot_ci(q["Top10"].values)
        log(f"  {name:20s} (iter={mi} depth={md} lr={lr}): "
            f"blind={q['Top10'].mean():.6f} [{ci_lo:.6f},{ci_hi:.6f}] delta={delta:+.6f}")

        # Delta CI
        q_bl = qmetrics(blind, "blend_score")
        delta_vals = q["Top10"].values - q_bl["Top10"].values
        d_lo, d_hi = boot_ci(delta_vals)
        results.append({"config": name, "Top10": round(q["Top10"].mean(),6),
                        "CI_lo": ci_lo, "CI_hi": ci_hi,
                        "delta": round(delta,6), "delta_CI_lo": d_lo, "delta_CI_hi": d_hi})

    # Best config strata
    best = results[0]  # use default for now
    q_best = qmetrics(blind, "_s")
    q_bl = qmetrics(blind, "blend_score")

    # Rescue
    q_best["bl_hit"] = q_bl["Top10"].values
    miss = q_best[q_best["bl_hit"]==0]
    hit = q_best[q_best["bl_hit"]==1]
    rescue = int((miss["Top10"]==1).sum())
    lost = int((hit["Top10"]==0).sum())
    log(f"\n  Rescue: {rescue}/{len(miss)}, Lost: {lost}/{len(hit)}")

    # Per-fragment blind strata
    log("\n--- Blind Per-Fragment (default config) ---")
    strata = []
    for frag in sorted(blind["old_fragment_smiles"].unique()):
        g = blind[blind["old_fragment_smiles"]==frag]
        qs = qmetrics(g, "_s"); qb = qmetrics(g, "blend_score")
        d = qs["Top10"].mean() - qb["Top10"].mean()
        log(f"  {frag:25s} n={len(qs):5d} bl={qb['Top10'].mean():.4f} sc={qs['Top10'].mean():.4f} d={d:+.4f}")
        strata.append({"fragment":frag,"n":len(qs),
                       "bl_Top10":round(qb["Top10"].mean(),6),
                       "sc_Top10":round(qs["Top10"].mean(),6),
                       "delta":round(d,6)})

    pd.DataFrame(strata).to_csv(f"{OUT}/d4s31_blind_strata.csv",index=False)
    pd.DataFrame(results).to_csv(f"{OUT}/d4s31_final_metrics.csv",index=False)

    # Compare with D4S28R
    log("\n=== COMPARISON ===")
    log(f"  D4S28R scorer_all:  blind=0.8851 delta=+0.029")
    log(f"  D4S31 drop_ranks:   blind={best['Top10']:.4f} delta={best['delta']:+.4f}")
    log(f"  Improvement:         +{best['Top10']-0.8851:.4f}")

    log(f"\nD4S31 LOCKDOWN complete.")

if __name__=="__main__":
    main()
