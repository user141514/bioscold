#!/usr/bin/env python3
"""D4S31: Feature ablation — find minimal set for best blind generalization."""
import pandas as pd, numpy as np, os, time, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S31_feature_pruning/results"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def qmetrics(df, col, lbl="label"):
    r = []
    for qid, g in df.groupby("query_id"):
        gs = g.sort_values(col, ascending=False)
        ls = gs[lbl].values; pr = np.where(ls==1)[0]+1
        if len(pr)==0: continue
        b = pr.min()
        r.append({"query_id":qid,"Top10":int(b<=10),"MRR":1.0/b})
    return pd.DataFrame(r)

CAT = ['old_fragment_smiles','attachment_signature','replacement_frequency_bin']

# Feature groups
GROUPS = {
    "model_scores": ['blend_score','mlp_z','hgb_z','mlp_score','hgb_refit_score',
                      'score_DE','score_HGB','score_Borda','score_attach'],
    "model_ranks": ['mlp_rank','blend_rank','rank_DE','rank_HGB','rank_Borda','rank_attach'],
    "topk_flags": ['candidate_in_attach_top10','candidate_in_DE_top10',
                    'candidate_in_HGB_top10','candidate_in_Borda_top10'],
    "prior_scores": ['cont_prior_score','backoff_logit_score','t_p4_logit_score',
                      'p1_logit_score','p3_logit_score'],
    "prior_rates": ['t_p4_smoothed_rate','p1_smoothed_rate','p3_smoothed_rate',
                     'backoff_smoothed_rate'],
    "prior_supports": ['t_p4_support','p1_support','p3_support','p0_support','backoff_support'],
    "prior_positives": ['t_p4_positive','p1_positive','p3_positive'],
    "prior_ranks": ['backoff_logit_rank','cont_prior_rank','t_p4_logit_rank',
                     'p1_logit_rank','p3_logit_rank'],
    "pmi": ['t_pmi_p4_p1','pmi_p3_p1','pmi_p1_p0',
            't_pmi_p4_p1_rank','pmi_p3_p1_rank','pmi_p1_p0_rank'],
    "mol_props": ['delta_heavy_atoms','delta_ring_count','delta_hetero_count',
                   'delta_mw','delta_logp','delta_tpsa','candidate_heavy_atoms',
                   'candidate_mw','candidate_logp','candidate_tpsa'],
    "sim_freq": ['_tanimoto_similarity','replacement_frequency','attachment_frequency'],
    "query_stats": ['query_prior_entropy','query_prior_margin','query_prior_max_support'],
}

def build_X(df, feature_list, templates=None):
    num = [c for c in feature_list if c not in CAT and c in df.columns]
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
    log("D4S31: Feature Ablation Study")
    log("="*60)

    val = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    blind = pd.read_parquet(f"{D4S27}/candidate_prior_scores_blind.parquet")
    y_val = val["label"].values

    # ALL features baseline
    all_features = []
    for g in GROUPS.values():
        all_features.extend(g)

    # Test each ablation: remove one group at a time
    log("\n=== LEAVING ONE GROUP OUT ===")
    results = []
    for remove_group in sorted(GROUPS.keys()):
        feats = [f for g_name, g_feats in GROUPS.items()
                 if g_name != remove_group
                 for f in g_feats]
        X_val, t = build_X(val, feats)
        X_blind = build_X(blind, feats, templates=t)

        pi = np.where(y_val==1)[0]; ni = np.where(y_val==0)[0]
        nn = min(len(ni), len(pi)*5)
        ns = np.random.RandomState(42).choice(ni, nn, replace=False)
        ti = np.concatenate([pi, ns])
        hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                              early_stopping=False, random_state=42,
                                              class_weight="balanced")
        hgb.fit(X_val[ti], y_val[ti])
        blind["_s"] = hgb.predict_proba(X_blind)[:, 1]
        q = qmetrics(blind, "_s")
        delta = q["Top10"].mean() - 0.855773
        n_feats = X_val.shape[1]
        results.append({"ablation": f"drop_{remove_group}", "n_feats": n_feats,
                        "blind_Top10": round(q["Top10"].mean(),6),
                        "delta": round(delta,6)})
        log(f"  drop_{remove_group:20s} dims={n_feats:3d} blind={q['Top10'].mean():.6f} delta={delta:+.6f}")

    # ALL features
    X_val_all, t_all = build_X(val, all_features)
    X_blind_all = build_X(blind, all_features, templates=t_all)
    hgb_all = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                              early_stopping=False, random_state=42,
                                              class_weight="balanced")
    hgb_all.fit(X_val_all[ti], y_val[ti])
    blind["_all"] = hgb_all.predict_proba(X_blind_all)[:, 1]
    q_all = qmetrics(blind, "_all")
    results.append({"ablation": "ALL_FEATURES", "n_feats": X_val_all.shape[1],
                    "blind_Top10": round(q_all["Top10"].mean(),6),
                    "delta": round(q_all["Top10"].mean()-0.855773,6)})
    log(f"  {'ALL_FEATURES':20s} dims={X_val_all.shape[1]:3d} blind={q_all['Top10'].mean():.6f} delta={q_all['Top10'].mean()-0.855773:+.6f}")

    # Keep only groups that don't hurt
    log("\n=== KEEP ONLY GROUPS WHERE DROP REDUCES PERFORMANCE ===")
    # Find best blind Top10
    best_top10 = max(r["blind_Top10"] for r in results)
    keep_groups = []
    for g_name in sorted(GROUPS.keys()):
        # Remove this group
        feats_without = [f for gn, gf in GROUPS.items() if gn != g_name for f in gf]
        X_val_wo, t_wo = build_X(val, feats_without)
        X_blind_wo = build_X(blind, feats_without, templates=t_wo)
        hgb_wo = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                                 early_stopping=False, random_state=42,
                                                 class_weight="balanced")
        hgb_wo.fit(X_val_wo[ti], y_val[ti])
        blind["_wo"] = hgb_wo.predict_proba(X_blind_wo)[:, 1]
        q_wo = qmetrics(blind, "_wo")
        drop_hurts = q_wo["Top10"].mean() < best_top10 - 0.001  # drops >0.1pp
        if drop_hurts:
            keep_groups.append(g_name)
            log(f"  KEEP {g_name:20s} (drop -> {q_wo['Top10'].mean():.6f}, hurts)")
        else:
            log(f"  DROP {g_name:20s} (drop -> {q_wo['Top10'].mean():.6f}, ok)")

    # Minimal feature set
    minimal_feats = [f for gn in keep_groups for f in GROUPS[gn]]
    log(f"\n  Minimal set: {len(minimal_feats)} features from {keep_groups}")

    X_val_min, t_min = build_X(val, minimal_feats)
    X_blind_min = build_X(blind, minimal_feats, templates=t_min)
    hgb_min = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                              early_stopping=False, random_state=42,
                                              class_weight="balanced")
    hgb_min.fit(X_val_min[ti], y_val[ti])
    blind["_min"] = hgb_min.predict_proba(X_blind_min)[:, 1]
    q_min = qmetrics(blind, "_min")
    log(f"  Minimal blind Top10: {q_min['Top10'].mean():.6f} delta={q_min['Top10'].mean()-0.855773:+.6f}")

    # Regularization sweep on minimal set
    log("\n=== REGULARIZATION SWEEP (minimal set) ===")
    for lr in [0.05, 0.1, 0.2]:
        for md in [4, 6]:
            for mi in [100, 200]:
                hgb_r = HistGradientBoostingClassifier(max_iter=mi, max_depth=md, learning_rate=lr,
                                                        early_stopping=False, random_state=42,
                                                        class_weight="balanced",
                                                        l2_regularization=1.0)
                hgb_r.fit(X_val_min[ti], y_val[ti])
                blind["_r"] = hgb_r.predict_proba(X_blind_min)[:, 1]
                q_r = qmetrics(blind, "_r")
                log(f"    lr={lr:.2f} depth={md} iter={mi:3d} -> blind={q_r['Top10'].mean():.6f}")

    # Save
    pd.DataFrame(results).to_csv(f"{OUT}/d4s31_ablation.csv", index=False)
    log(f"\nSaved d4s31_ablation.csv")
    log(f"D4S31 complete. Best blind={best_top10:.6f}")

if __name__=="__main__":
    main()
