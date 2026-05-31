#!/usr/bin/env python3
"""
D4S28 Blind One-Shot Evaluation
Train HistGB on full val data (85 safe features), predict on blind.
"""
import pandas as pd
import numpy as np
import os, time, json, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27 = f"{PROJECT}/goal/A_improve/results"
OUT = f"{PROJECT}/goal/A_improve/D4S28_candidate_transform_scorer/results"
os.makedirs(OUT, exist_ok=True)

SAFE_FEATURES = [
    'blend_score','mlp_z','hgb_z','mlp_score','hgb_refit_score',
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
    'query_prior_entropy','query_prior_margin','query_prior_max_support',
]
CAT_FEATURES = ['old_fragment_smiles','attachment_signature','replacement_frequency_bin']

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)

def build_X(df, cat_templates=None):
    """Build feature matrix. If cat_templates provided, align one-hot columns."""
    numeric_cols = [c for c in SAFE_FEATURES if c not in CAT_FEATURES and c in df.columns]
    X_parts = [df[numeric_cols].fillna(0).values.astype(np.float64)]
    templates = {}
    for cat_col in CAT_FEATURES:
        if cat_col in df.columns:
            if cat_templates and cat_col in cat_templates:
                # Align to template categories, map unseen to 'OTHER'
                known = cat_templates[cat_col]
                mapped = df[cat_col].apply(lambda x: x if x in known else 'OTHER')
            else:
                mapped = df[cat_col]
                known = set(mapped.unique())
            dummies = pd.get_dummies(mapped, prefix=cat_col)
            if cat_templates and cat_col in cat_templates:
                # Keep only val template columns + OTHER
                template_cols = [f"{cat_col}_{v}" for v in known]
                template_cols.append(f"{cat_col}_OTHER")  # always include for alignment
                dummies = dummies.reindex(columns=template_cols, fill_value=0)
            else:
                known = set(mapped.unique())
                dummies[f"{cat_col}_OTHER"] = 0  # reserve OTHER slot
                templates[cat_col] = known
            X_parts.append(dummies.values.astype(np.float64))
    X = np.hstack(X_parts)
    if cat_templates is None:
        return X, templates
    return X

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
    t0 = time.time()
    log("=" * 60)
    log("D4S28 BLIND ONE-SHOT EVALUATION")
    log("=" * 60)

    # Load val and train
    log("Loading val parquet...")
    val = pd.read_parquet(f"{D4S27}/candidate_prior_scores_val.parquet")
    log(f"  Val: {len(val):,} rows, {val['query_id'].nunique():,} queries")

    log("Loading blind parquet...")
    blind = pd.read_parquet(f"{D4S27}/candidate_prior_scores_blind.parquet")
    log(f"  Blind: {len(blind):,} rows, {blind['query_id'].nunique():,} queries")

    # Build features
    log("Building feature matrices...")
    X_val, cat_templates = build_X(val)
    y_val = val["label"].values
    X_blind = build_X(blind, cat_templates=cat_templates)
    y_blind = blind["label"].values
    log(f"  X_val: {X_val.shape}, X_blind: {X_blind.shape}")
    assert X_val.shape[1] == X_blind.shape[1], f"Feature dim mismatch: {X_val.shape[1]} vs {X_blind.shape[1]}"

    # Subsample val negatives for training speed
    pos_idx = np.where(y_val == 1)[0]
    neg_idx = np.where(y_val == 0)[0]
    n_neg = min(len(neg_idx), len(pos_idx) * 5)
    ns = np.random.RandomState(42).choice(neg_idx, n_neg, replace=False)
    train_idx = np.concatenate([pos_idx, ns])
    log(f"  Train: {len(train_idx):,} rows (pos={len(pos_idx)}, neg={n_neg})")

    # Train HistGB on full val
    log("Training HistGB on full val data...")
    hgb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.1,
        early_stopping=False, random_state=42, class_weight="balanced",
    )
    hgb.fit(X_val[train_idx], y_val[train_idx])

    # Predict on blind
    log("Predicting on blind...")
    blind_preds = hgb.predict_proba(X_blind)[:, 1]
    blind["score_histgb"] = blind_preds

    # Evaluate
    log("\n=== BLIND RESULTS ===")
    q_blind = compute_metrics(blind, "score_histgb")
    q_bl = compute_metrics(blind, "blend_score")
    q_cont = compute_metrics(blind, "cont_prior_score")
    q_backoff = compute_metrics(blind, "backoff_logit_score")

    b_top10 = q_blind["Top10"].mean()
    bl_top10 = q_bl["Top10"].mean()
    log(f"  HistGB blind Top10:  {b_top10:.6f}")
    log(f"  Baseline blind Top10: {bl_top10:.6f}")
    log(f"  Delta:                {b_top10 - bl_top10:+.6f}")
    log(f"  HistGB blind MRR:     {q_blind['MRR'].mean():.6f}")

    # Strata
    log("\n=== BLIND STRATA ===")
    strata_rows = []
    for frag in sorted(blind["old_fragment_smiles"].unique()):
        grp = blind[blind["old_fragment_smiles"] == frag]
        qb = compute_metrics(grp, "score_histgb")
        qbl = compute_metrics(grp, "blend_score")
        n = len(qb)
        log(f"  {frag:25s} n={n:5d}  baseline={qbl['Top10'].mean():.4f}  scorer={qb['Top10'].mean():.4f}")
        strata_rows.append({
            "split": "blind", "dimension": "old_fragment", "category": frag,
            "n_queries": n, "baseline_Top10": round(qbl["Top10"].mean(), 6),
            "scorer_Top10": round(qb["Top10"].mean(), 6),
        })

    # Hard-miss strata
    hm_col = "hard_top10_miss_flag"
    if hm_col in blind.columns:
        for hmf, label in [(0, "No"), (1, "Yes")]:
            grp = blind[blind[hm_col] == hmf]
            if len(grp) == 0: continue
            qb = compute_metrics(grp, "score_histgb")
            qbl = compute_metrics(grp, "blend_score")
            n = len(qb)
            log(f"  hard_miss={label:4s}  n={n:5d}  baseline={qbl['Top10'].mean():.4f}  scorer={qb['Top10'].mean():.4f}")
            strata_rows.append({
                "split": "blind", "dimension": "hard_miss", "category": f"hard_miss={label}",
                "n_queries": n, "baseline_Top10": round(qbl["Top10"].mean(), 6),
                "scorer_Top10": round(qb["Top10"].mean(), 6),
            })

    pd.DataFrame(strata_rows).to_csv(f"{OUT}/d4s28_blind_strata.csv", index=False)

    # Rescue analysis
    q_blind["baseline_hit"] = q_bl["Top10"].values
    baseline_miss = q_blind[q_blind["baseline_hit"] == 0]
    rescue = int((baseline_miss["Top10"] == 1).sum())  # scorer hits where baseline missed
    n_miss = len(baseline_miss)
    baseline_hit = q_blind[q_blind["baseline_hit"] == 1]
    hits_lost = int((baseline_hit["Top10"] == 0).sum())
    log(f"\n  Baseline misses: {n_miss}, Rescued: {rescue} ({rescue/n_miss*100:.1f}%)")
    log(f"  Baseline hits: {len(baseline_hit)}, Lost: {hits_lost}")

    # Save summary
    summary = pd.DataFrame([{
        "method": "D4S28_HistGB_safe_blind",
        "n_queries": len(q_blind),
        "Top1": round(q_blind["Top1"].mean(), 6),
        "Top5": round(q_blind["Top5"].mean(), 6),
        "Top10": round(b_top10, 6),
        "Top20": round(q_blind["Top20"].mean(), 6),
        "MRR": round(q_blind["MRR"].mean(), 6),
        "delta_Top10_vs_blend": round(b_top10 - bl_top10, 6),
        "baseline_misses": n_miss,
        "rescued": rescue,
        "hits_lost": hits_lost,
    }, {
        "method": "D4S27_baseline_blend_blind",
        "n_queries": len(q_bl),
        "Top1": round(q_bl["Top1"].mean(), 6),
        "Top5": round(q_bl["Top5"].mean(), 6),
        "Top10": round(bl_top10, 6),
        "Top20": round(q_bl["Top20"].mean(), 6),
        "MRR": round(q_bl["MRR"].mean(), 6),
        "delta_Top10_vs_blend": 0,
        "baseline_misses": int((q_bl["Top10"] == 0).sum()),
        "rescued": 0,
        "hits_lost": 0,
    }])
    summary.to_csv(f"{OUT}/d4s28_blind_summary.csv", index=False)
    log(f"\n  Saved d4s28_blind_summary.csv")

    log(f"\nD4S28 blind one-shot complete in {(time.time()-t0)/60:.1f} min")
    return b_top10, bl_top10

if __name__ == "__main__":
    main()
