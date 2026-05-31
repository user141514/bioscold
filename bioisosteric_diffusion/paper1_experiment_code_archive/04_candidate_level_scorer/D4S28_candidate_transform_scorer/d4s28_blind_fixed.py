#!/usr/bin/env python3
"""
D4S28 Blind (FIXED): proper cat_templates alignment val->blind.
"""
import pandas as pd, numpy as np, os, time, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
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

def train_hgb(X, y):
    pi = np.where(y==1)[0]; ni = np.where(y==0)[0]
    nn = min(len(ni), len(pi)*5)
    ns = np.random.RandomState(42).choice(ni, nn, replace=False)
    ti = np.concatenate([pi, ns])
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                        early_stopping=False, random_state=42,
                                        class_weight="balanced")
    m.fit(X[ti], y[ti])
    return m

def main():
    log("="*60)
    log("D4S28 BLIND — FIXED (proper cat_templates alignment)")
    log("="*60)
    val = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    blind = pd.read_parquet(f"{D4S27}/candidate_prior_scores_blind.parquet")
    log(f"Val: {len(val):,} rows x {len(val.columns)} cols, {val['query_id'].nunique():,} queries")
    log(f"Blind: {len(blind):,} rows x {len(blind.columns)} cols, {blind['query_id'].nunique():,} queries")

    X_val, templates = build_X(val)
    y_val = val["label"].values
    X_blind = build_X(blind, templates=templates)
    log(f"Dims: val={X_val.shape}, blind={X_blind.shape}")
    assert X_val.shape[1] == X_blind.shape[1], f"MISMATCH!"

    # 5-fold OOF on val
    log("\n--- Val 5-fold OOF ---")
    uq = np.array(list(set(val["query_id"].values)))
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_val))
    for fold, (tqi, vqi) in enumerate(gkf.split(uq, groups=uq)):
        tqs = frozenset(uq[tqi]); vqs = frozenset(uq[vqi])
        tm = np.array([q in tqs for q in val["query_id"].values])
        vm = np.array([q in vqs for q in val["query_id"].values])
        hgb = train_hgb(X_val[tm], y_val[tm])
        oof[vm] = hgb.predict_proba(X_val[vm])[:, 1]
        fh = 0; fn = 0
        for _, g in val.iloc[vm].groupby("query_id"):
            fn += 1; gs = g.copy(); gs["_s"] = oof[g.index]
            gs = gs.sort_values("_s", ascending=False)
            if (gs["label"].values==1).nonzero()[0].min()+1 <= 10: fh += 1
        log(f"  Fold{fold+1}: Top10={fh/fn:.4f} ({fh}/{fn})")
    val["score_oof"] = oof
    q_val = metrics(val, "score_oof")
    q_bl_val = metrics(val, "blend_score")
    log(f"Val OOF Top10: {q_val['Top10'].mean():.6f} (baseline {q_bl_val['Top10'].mean():.6f})")

    # Blind one-shot
    log("\n--- Blind One-Shot ---")
    hgb_full = train_hgb(X_val, y_val)
    blind["score_hgb"] = hgb_full.predict_proba(X_blind)[:, 1]
    q_blind = metrics(blind, "score_hgb")
    q_bl_blind = metrics(blind, "blend_score")
    log(f"Blind Top10:  {q_blind['Top10'].mean():.6f}")
    log(f"Baseline:     {q_bl_blind['Top10'].mean():.6f}")
    log(f"Delta:        {q_blind['Top10'].mean() - q_bl_blind['Top10'].mean():+.6f}")

    # Strata
    log("\n--- Blind Per-Fragment ---")
    rows = []
    for frag in sorted(blind["old_fragment_smiles"].unique()):
        g = blind[blind["old_fragment_smiles"]==frag]
        qs = metrics(g, "score_hgb"); qb = metrics(g, "blend_score")
        log(f"  {frag:25s} n={len(qs):5d} blend={qb['Top10'].mean():.4f} scorer={qs['Top10'].mean():.4f}")
        rows.append({"fragment":frag,"n":len(qs),
                     "blend_Top10":round(qb["Top10"].mean(),6),
                     "scorer_Top10":round(qs["Top10"].mean(),6)})
    pd.DataFrame(rows).to_csv(f"{OUT}/d4s28_blind_fixed_strata.csv", index=False)

    # Rescue analysis
    q_blind["blend_hit"] = q_bl_blind["Top10"].values
    miss = q_blind[q_blind["blend_hit"]==0]
    hit = q_blind[q_blind["blend_hit"]==1]
    rescue = int((miss["Top10"]==1).sum())
    lost = int((hit["Top10"]==0).sum())
    log(f"\n  Rescue: {rescue}/{len(miss)} ({rescue/len(miss)*100:.1f}%), Lost: {lost}/{len(hit)}")

    # Summary
    r = pd.DataFrame([
        {"method":"D4S28_HistGB_blind_FIXED","Top10":round(q_blind["Top10"].mean(),6),
         "MRR":round(q_blind["MRR"].mean(),6),
         "delta_blind":round(q_blind["Top10"].mean()-q_bl_blind["Top10"].mean(),6)},
        {"method":"D4S27_blend_blind","Top10":round(q_bl_blind["Top10"].mean(),6),
         "MRR":round(q_bl_blind["MRR"].mean(),6),"delta_blind":0},
        {"method":"D4S28_HistGB_val_OOF","Top10":round(q_val["Top10"].mean(),6),
         "MRR":round(q_val["MRR"].mean(),6),
         "delta_blind":round(q_val["Top10"].mean()-q_bl_val["Top10"].mean(),6)},
        {"method":"D4S28_HistGB_blind_BUGGY","Top10":0.711995,"MRR":0.336090,"delta_blind":-0.143778},
    ])
    r.to_csv(f"{OUT}/d4s28_blind_fixed_summary.csv",index=False)
    log(f"\nSaved. D4S28 blind FIXED complete.")

if __name__=="__main__":
    main()
