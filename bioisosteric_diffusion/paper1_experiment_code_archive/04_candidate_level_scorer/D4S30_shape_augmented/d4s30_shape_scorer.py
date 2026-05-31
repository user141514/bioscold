#!/usr/bin/env python3
"""
D4S30: Shape-Augmented Candidate Scorer
Pre-compute USR/USRCAT for all unique old_fragments + candidates,
add 3D shape features to HistGB pipeline.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
import os, time, warnings
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S30_shape_augmented/results"
os.makedirs(OUT, exist_ok=True)

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def clean_smiles(smi):
    return smi.replace("*", "[H]")

def compute_usr(smi, cache={}):
    """Compute USR descriptor with caching."""
    if smi in cache:
        return cache[smi]
    try:
        mol = Chem.MolFromSmiles(clean_smiles(smi))
        if mol is None:
            cache[smi] = None; return None
        mh = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        ok = AllChem.EmbedMolecule(mh, params)
        if ok < 0:
            cache[smi] = None; return None
        AllChem.MMFFOptimizeMolecule(mh)
        usr = np.array(rdMolDescriptors.GetUSR(mh), dtype=np.float64)
        cache[smi] = usr
        return usr
    except:
        cache[smi] = None
        return None

def usr_similarity(usr1, usr2):
    """USR similarity = 1 / (1 + Euclidean distance)."""
    if usr1 is None or usr2 is None:
        return 0.0
    dist = np.sqrt(np.sum((usr1 - usr2) ** 2))
    return 1.0 / (1.0 + dist)

def usr_shape_score(usr1, usr2):
    """USR shape score (first 3 moments, indices 0-2, 3-5, 6-8, 9-11)."""
    if usr1 is None or usr2 is None:
        return np.zeros(4)
    scores = []
    for i in range(4):
        start = i * 3
        d1 = np.array(usr1[start:start+3])
        d2 = np.array(usr2[start:start+3])
        dist = np.sqrt(np.sum((d1 - d2) ** 2))
        scores.append(1.0 / (1.0 + dist))
    return np.array(scores)

def compute_metrics(df, score_col, label_col="label"):
    results = []
    for qid, grp in df.groupby("query_id"):
        gs = grp.sort_values(score_col, ascending=False)
        labels = gs[label_col].values
        pos_ranks = np.where(labels == 1)[0] + 1
        if len(pos_ranks) == 0:
            continue
        best = pos_ranks.min()
        results.append({
            "query_id": qid, "best_rank": best,
            "Top1": int(best==1), "Top5": int(best<=5),
            "Top10": int(best<=10), "Top20": int(best<=20),
            "MRR": 1.0/best,
        })
    return pd.DataFrame(results)


def main():
    log("=" * 60)
    log("D4S30: Shape-Augmented Candidate Scorer")
    log("=" * 60)

    # Load data
    log("Loading val + blind parquet...")
    val = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    blind = pd.read_parquet(f"{D4S27}/candidate_prior_scores_blind.parquet")
    log(f"  Val: {len(val):,} rows, {val['query_id'].nunique():,} queries")
    log(f"  Blind: {len(blind):,} rows, {blind['query_id'].nunique():,} queries")

    # ── Pre-compute USR for all unique molecules ──
    all_old = set(val["old_fragment_smiles"].unique()) | set(blind["old_fragment_smiles"].unique())
    all_cand = set(val["candidate_smiles"].unique()) | set(blind["candidate_smiles"].unique())
    log(f"Pre-computing USR for {len(all_old)} old_fragments + {len(all_cand)} candidates...")

    usr_cache = {}
    t0 = time.time()
    for smi in sorted(all_old | all_cand):
        compute_usr(smi, cache=usr_cache)
    n_ok = sum(1 for v in usr_cache.values() if v is not None)
    log(f"  {n_ok}/{len(usr_cache)} USR computed in {time.time()-t0:.1f}s")

    # ── Add USR features to val and blind ──
    def add_usr_features(df):
        df = df.copy()
        usr_dists = []
        usr_shape_scores = []
        for _, row in df.iterrows():
            old_usr = usr_cache.get(row["old_fragment_smiles"])
            cand_usr = usr_cache.get(row["candidate_smiles"])
            usr_dists.append(usr_similarity(old_usr, cand_usr))
            usr_shape_scores.append(usr_shape_score(old_usr, cand_usr))
        df["usr_similarity"] = usr_dists
        scores_arr = np.array(usr_shape_scores)
        df["usr_shape_ctd"] = scores_arr[:, 0]  # CTD = distribution moments
        df["usr_shape_cst"] = scores_arr[:, 1]  # CST
        df["usr_shape_fct"] = scores_arr[:, 2]  # FCT
        df["usr_shape_ftf"] = scores_arr[:, 3]  # FTF
        return df

    val = add_usr_features(val)
    blind = add_usr_features(blind)

    # ── Safe features + USR ──
    SAFE_FEATURES = [
        'blend_score','mlp_z','hgb_z','mlp_score','hgb_refit_score',
        'score_DE','score_HGB','score_Borda',
        'mlp_rank','blend_rank','rank_DE','rank_HGB','rank_Borda',
        'candidate_in_DE_top10','candidate_in_HGB_top10','candidate_in_Borda_top10',
        'cont_prior_score','backoff_logit_score','t_p4_logit_score','p1_logit_score','p3_logit_score',
        't_p4_smoothed_rate','p1_smoothed_rate','p3_smoothed_rate','backoff_smoothed_rate',
        't_p4_support','p1_support','p3_support','p0_support','backoff_support',
        't_p4_positive','p1_positive','p3_positive',
        'backoff_logit_rank','cont_prior_rank',
        't_pmi_p4_p1','pmi_p3_p1','pmi_p1_p0',
        'delta_heavy_atoms','delta_ring_count','delta_hetero_count','delta_mw','delta_logp','delta_tpsa',
        'candidate_heavy_atoms','candidate_mw','candidate_logp','candidate_tpsa',
        '_tanimoto_similarity','replacement_frequency','attachment_frequency',
        'query_prior_entropy','query_prior_margin','query_prior_max_support',
    ]
    USR_FEATURES = ['usr_similarity', 'usr_shape_ctd', 'usr_shape_cst', 'usr_shape_fct', 'usr_shape_ftf']
    CAT_FEATURES = ['old_fragment_smiles','attachment_signature','replacement_frequency_bin']

    def build_X(df, cat_templates=None):
        num_cols = [c for c in SAFE_FEATURES + USR_FEATURES
                    if c not in CAT_FEATURES and c in df.columns]
        X_parts = [df[num_cols].fillna(0).values.astype(np.float64)]
        templates = {}
        for cat_col in CAT_FEATURES:
            if cat_col in df.columns:
                mapped = df[cat_col].copy()
                if cat_templates and cat_col in cat_templates:
                    known = cat_templates[cat_col]
                    mapped = mapped.apply(lambda x: x if x in known else 'OTHER')
                else:
                    known = set(mapped.unique())
                dummies = pd.get_dummies(mapped, prefix=cat_col)
                template_cols = [f"{cat_col}_{v}" for v in known]
                template_cols.append(f"{cat_col}_OTHER")
                dummies = dummies.reindex(columns=template_cols, fill_value=0)
                X_parts.append(dummies.values.astype(np.float64))
                templates[cat_col] = known
        X = np.hstack(X_parts)
        return X if cat_templates else (X, templates)

    X_val, cat_tmpl = build_X(val)
    y_val = val["label"].values
    queries_val = val["query_id"].values
    log(f"  X_val shape: {X_val.shape}")

    X_blind = build_X(blind, cat_templates=cat_tmpl)
    y_blind = blind["label"].values
    log(f"  X_blind shape: {X_blind.shape}")

    # ── 5-fold CV on val ──
    log("\n=== 5-FOLD CV (query-OOF) ===")
    uq = np.array(list(set(queries_val)))
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_val))

    for fold, (tqi, vqi) in enumerate(gkf.split(uq, groups=uq)):
        tqs = frozenset(uq[tqi]); vqs = frozenset(uq[vqi])
        tm = np.array([q in tqs for q in queries_val])
        vm = np.array([q in vqs for q in queries_val])
        Xt, yt = X_val[tm], y_val[tm]
        Xv = X_val[vm]
        pi = np.where(yt == 1)[0]; ni = np.where(yt == 0)[0]
        nn = min(len(ni), len(pi) * 5)
        ns = np.random.RandomState(42).choice(ni, nn, replace=False)
        ti = np.concatenate([pi, ns])
        hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                              early_stopping=False, random_state=42,
                                              class_weight="balanced")
        hgb.fit(Xt[ti], yt[ti])
        oof[vm] = hgb.predict_proba(Xv)[:, 1]
        # Quick fold check
        df_fold = val.iloc[vm].copy()
        df_fold["_oof"] = oof[vm]
        fold_hits = sum(1 for _, g in df_fold.groupby("query_id")
                        if (g.sort_values("_oof", ascending=False)["label"].values == 1).nonzero()[0].min() + 1 <= 10)
        n_fold_q = df_fold["query_id"].nunique()
        log(f"  Fold {fold+1}: Top10={fold_hits/n_fold_q:.4f} ({fold_hits}/{n_fold_q})")

    val["score_shape_oof"] = oof
    q_val = compute_metrics(val, "score_shape_oof")
    q_bl = compute_metrics(val, "blend_score")
    shape_top10 = q_val["Top10"].mean()

    log(f"\n  Shape-augmented OOF Top10: {shape_top10:.6f}")
    log(f"  Baseline blend Top10:      {q_bl['Top10'].mean():.6f}")
    log(f"  Delta:                     {shape_top10 - q_bl['Top10'].mean():+.6f}")

    # ── USR standalone ranking ──
    q_usr = compute_metrics(val, "usr_similarity")
    log(f"  USR-only Top10:           {q_usr['Top10'].mean():.6f}")
    log(f"  (D4S28 HistGB was:         0.937421)")

    # Fragment strata
    log("\n  Fragment strata (shape-augmented OOF):")
    for frag in sorted(val["old_fragment_smiles"].unique()):
        grp = val[val["old_fragment_smiles"] == frag]
        qf = compute_metrics(grp, "score_shape_oof")
        qbl = compute_metrics(grp, "blend_score")
        log(f"  {frag:25s} n={len(qf):5d}  baseline={qbl['Top10'].mean():.4f}  shape={qf['Top10'].mean():.4f}")

    # ── Blind one-shot ──
    log("\n=== BLIND ONE-SHOT ===")
    # Train on full val
    pi = np.where(y_val == 1)[0]; ni = np.where(y_val == 0)[0]
    nn = min(len(ni), len(pi) * 5)
    ns = np.random.RandomState(42).choice(ni, nn, replace=False)
    ti = np.concatenate([pi, ns])
    hgb_full = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                               early_stopping=False, random_state=42,
                                               class_weight="balanced")
    hgb_full.fit(X_val[ti], y_val[ti])
    blind_preds = hgb_full.predict_proba(X_blind)[:, 1]
    blind["score_shape"] = blind_preds
    q_blind = compute_metrics(blind, "score_shape")
    q_blind_bl = compute_metrics(blind, "blend_score")
    blind_top10 = q_blind["Top10"].mean()
    log(f"  Shape-augmented blind Top10: {blind_top10:.6f}")
    log(f"  Baseline blind Top10:        {q_blind_bl['Top10'].mean():.6f}")
    log(f"  Delta:                       {blind_top10 - q_blind_bl['Top10'].mean():+.6f}")

    # ── Save ──
    summary = pd.DataFrame([
        {"method": "D4S30_shape_augmented_val_OOF", "Top10": round(shape_top10, 6),
         "MRR": round(q_val["MRR"].mean(), 6),
         "delta_vs_blend": round(shape_top10 - q_bl['Top10'].mean(), 6)},
        {"method": "D4S30_shape_augmented_blind", "Top10": round(blind_top10, 6),
         "MRR": round(q_blind["MRR"].mean(), 6),
         "delta_vs_blend": round(blind_top10 - q_blind_bl['Top10'].mean(), 6)},
        {"method": "D4S27_blend_val", "Top10": round(q_bl['Top10'].mean(), 6),
         "MRR": round(q_bl["MRR"].mean(), 6), "delta_vs_blend": 0},
        {"method": "D4S27_blend_blind", "Top10": round(q_blind_bl['Top10'].mean(), 6),
         "MRR": round(q_blind_bl["MRR"].mean(), 6), "delta_vs_blend": 0},
        {"method": "D4S28_HistGB_val_OOF", "Top10": 0.937421, "MRR": 0.612301,
         "delta_vs_blend": 0.103402},
        {"method": "D4S28_HistGB_blind", "Top10": 0.711995, "MRR": 0.336090,
         "delta_vs_blend": -0.143778},
        {"method": "USR_only_val", "Top10": round(q_usr['Top10'].mean(), 6),
         "MRR": round(q_usr['MRR'].mean(), 6),
         "delta_vs_blend": round(q_usr['Top10'].mean() - q_bl['Top10'].mean(), 6)},
    ])
    summary.to_csv(f"{OUT}/d4s30_summary.csv", index=False)
    log(f"\n  Saved d4s30_summary.csv")

    log(f"\nD4S30 complete.")

if __name__ == "__main__":
    main()
