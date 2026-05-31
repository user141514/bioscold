#!/usr/bin/env python3
"""
D4S28: Candidate-Level Transform Scorer
========================================
Learn p(label=1 | old_fragment, attachment, candidate, scores, priors, features)
at candidate level, NOT query-level gate.

Strategy:
  - 5-fold GroupKFold over queries (val-internal CV)
  - Train binary classifier ranking by p(label=1)
  - Compare: baseline blend / D4S27 priors / scorer / scorer+blend / scorer rerank

Models (lightweight, ordered by complexity):
  1. LogisticRegression (balanced class_weight)
  2. SGDClassifier (modified_huber, pairwise pairs sampling)
  3. HistGradientBoostingClassifier (tree-based interactions)
"""
import pandas as pd
import numpy as np
import os, sys, time, json, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27_RESULTS = f"{PROJECT}/goal/A_improve/results"
OUT_DIR = f"{PROJECT}/goal/A_improve/D4S28_candidate_transform_scorer/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature definitions ──────────────────────────────────────────────
# All candidate-level numeric features from D4S27 parquet
NUMERIC_FEATURES = [
    # Learned model scores
    "blend_score", "mlp_z", "hgb_z", "score_DE", "score_HGB", "score_Borda",
    "mlp_score", "hgb_refit_score",
    # Learned model ranks
    "mlp_rank", "blend_rank", "rank_DE", "rank_HGB", "rank_Borda",
    # Prior scores (candidate-level)
    "cont_prior_score", "backoff_logit_score",
    "t_p4_logit_score", "p1_logit_score", "p3_logit_score",
    # Prior smoothed rates
    "t_p4_smoothed_rate", "p1_smoothed_rate", "p3_smoothed_rate",
    "backoff_smoothed_rate",
    # Prior supports
    "t_p4_support", "p1_support", "p3_support", "p0_support",
    "backoff_support",
    # Prior positive counts
    "t_p4_positive", "p1_positive", "p3_positive",
    # Prior ranks
    "backoff_logit_rank", "cont_prior_rank", "t_p4_logit_rank",
    "p1_logit_rank", "p3_logit_rank",
    # PMI scores
    "t_pmi_p4_p1", "pmi_p3_p1", "pmi_p1_p0",
    # PMI ranks
    "t_pmi_p4_p1_rank", "pmi_p3_p1_rank", "pmi_p1_p0_rank",
    # Molecular delta features
    "delta_heavy_atoms", "delta_ring_count", "delta_hetero_count",
    "delta_mw", "delta_logp", "delta_tpsa",
    # Candidate properties
    "candidate_heavy_atoms", "candidate_mw", "candidate_logp", "candidate_tpsa",
    # Similarity
    "_tanimoto_similarity",
    # Frequencies
    "replacement_frequency", "attachment_frequency",
    # Query-level features (same for all candidates in a query)
    "query_prior_entropy", "query_prior_margin", "query_prior_max_support",
]

CATEGORICAL_FEATURES = [
    "old_fragment_smiles",
    "attachment_signature",
    "replacement_frequency_bin",
]

# Features that would leak target information if used
EXCLUDED = [
    "label", "candidate_id", "query_id", "split", "candidate_smiles",
    "positive_set_size", "num_positives", "single_pos_flag",
    "multi_pos_flag", "compatibility_flag",
    "hard_top10_miss_flag",
    "candidate_in_attach_top10", "candidate_in_DE_top10",
    "candidate_in_HGB_top10", "candidate_in_Borda_top10",
    # Don't use these in prediction - they contain target-dependent info
    "rank_attach", "score_attach",
    "a4c_tier_if_available_diagnostic_only",
    "target_any_seen_vocab", "target_all_seen_vocab",
    "old_fragment_cluster_if_available", "old_fragment_cluster_id",
    "morgan_similarity_old_candidate_if_available",
    "replacement_frequency_mean",
    # Prior logits
    "p0_logit_score", "p0_smoothed_rate", "p0_support", "p0_positive",
    # redundant
    "candidate_ring_count", "candidate_hetero_count",
]


def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)


def load_data():
    """Load val parquet, keep only needed columns."""
    log("Loading D4S27 val parquet...")
    df = pd.read_parquet(f"{D4S27_RESULTS}/candidate_prior_scores_val.parquet")
    log(f"  Loaded {len(df):,} rows, {len(df.columns)} cols")
    return df


def prepare_features(df):
    """Encode features for ML models."""
    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["query_id", "label"]
    df = df[keep_cols].copy()

    # Fill NaN with 0
    for c in NUMERIC_FEATURES:
        if df[c].isna().any():
            df[c] = df[c].fillna(0.0)

    # One-hot encode categoricals
    encoders = {}
    encoded_dfs = [df[NUMERIC_FEATURES].values]

    for cat_col in CATEGORICAL_FEATURES:
        # Use top 50 categories, rest as 'OTHER'
        top_cats = df[cat_col].value_counts().head(50).index.tolist()
        encoded = pd.get_dummies(
            df[cat_col].apply(lambda x: x if x in top_cats else 'OTHER'),
            prefix=cat_col
        )
        encoded_dfs.append(encoded.values)
        encoders[cat_col] = top_cats

    X = np.hstack(encoded_dfs)
    y = df["label"].values
    queries = df["query_id"].values

    feature_names = (NUMERIC_FEATURES +
                     [f"{c}_{v}" for c in CATEGORICAL_FEATURES
                      for v in encoders.get(c, [])] +
                     [f"{c}_OTHER" for c in CATEGORICAL_FEATURES])

    log(f"  Feature matrix: {X.shape[1]} features, {X.shape[0]:,} rows")
    log(f"  Positive rate: {y.mean():.4f}")
    return X, y, queries, feature_names, df


def compute_metrics(df, score_col, label_col="label"):
    """Compute Top-N and MRR for a score column, grouped by query."""
    results = []
    for qid, grp in df.groupby("query_id"):
        grp_sorted = grp.sort_values(score_col, ascending=False)
        labels = grp_sorted[label_col].values
        pos_ranks = np.where(labels == 1)[0] + 1
        if len(pos_ranks) == 0:
            continue
        best = pos_ranks.min()
        hit1 = int(best == 1)
        hit5 = int(best <= 5)
        hit10 = int(best <= 10)
        hit20 = int(best <= 20)
        mrr = 1.0 / best
        results.append({
            "query_id": qid,
            "best_rank": best,
            "Top1": hit1, "Top5": hit5, "Top10": hit10, "Top20": hit20,
            "MRR": mrr,
            "num_candidates": len(grp),
            "num_positives": labels.sum(),
        })
    return pd.DataFrame(results)


def evaluate_model(model_name, df_val, score_col, label_col="label"):
    """Full evaluation of a scoring method."""
    qdf = compute_metrics(df_val, score_col, label_col)

    # overall
    n = len(qdf)
    top1 = qdf["Top1"].mean()
    top5 = qdf["Top5"].mean()
    top10 = qdf["Top10"].mean()
    top20 = qdf["Top20"].mean()
    mrr = qdf["MRR"].mean()

    # rescue analysis vs baseline
    qdf_blend = compute_metrics(df_val, "blend_score", label_col)
    qdf["baseline_hit"] = qdf_blend["Top10"].values

    baseline_miss = qdf[qdf["baseline_hit"] == 0]
    n_miss = len(baseline_miss)
    rescue_11_20 = int(((baseline_miss["best_rank"] >= 11) &
                         (baseline_miss["best_rank"] <= 20)).sum())
    rescue_20p = int((baseline_miss["best_rank"] > 20).sum())
    rescue_total = int((baseline_miss["best_rank"] <= 10).sum())

    # hit preservation
    baseline_hit_q = qdf[qdf["baseline_hit"] == 1]
    n_hit = len(baseline_hit_q)
    hits_lost = int((baseline_hit_q["best_rank"] > 10).sum())

    # stratified by old_fragment
    strata = []
    df_val_with_meta = df_val.copy()
    # Merge old_fragment and other strata from D4S27 parquet
    if "old_fragment_smiles" in df_val.columns:
        for frag, grp in df_val.groupby("old_fragment_smiles"):
            qdf_frag = compute_metrics(grp, score_col, label_col)
            n_frag = len(qdf_frag)
            strata.append({
                "dimension": "old_fragment",
                "category": frag,
                "n_queries": n_frag,
                "Top10": round(qdf_frag["Top10"].mean(), 6),
                "MRR": round(qdf_frag["MRR"].mean(), 6),
            })
    if "replacement_frequency_bin" in df_val.columns:
        for fb, grp in df_val.groupby("replacement_frequency_bin"):
            qdf_fb = compute_metrics(grp, score_col, label_col)
            n_fb = len(qdf_fb)
            strata.append({
                "dimension": "replacement_frequency",
                "category": fb,
                "n_queries": n_fb,
                "Top10": round(qdf_fb["Top10"].mean(), 6),
                "MRR": round(qdf_fb["MRR"].mean(), 6),
            })

    return {
        "model": model_name,
        "n_queries": n,
        "Top1": round(top1, 6),
        "Top5": round(top5, 6),
        "Top10": round(top10, 6),
        "Top20": round(top20, 6),
        "MRR": round(mrr, 6),
        "delta_Top10": round(top10 - 0.834019, 6),
        "baseline_miss": n_miss,
        "rescue_total": rescue_total,
        "rescue_11_20": rescue_11_20,
        "rescue_20p": rescue_20p,
        "hits_lost": hits_lost,
        "strata": strata,
        "query_df": qdf,
    }


def train_logreg(X_train, y_train, X_val, y_val):
    """Logistic Regression with balanced class weight."""
    log("  Training LogisticRegression...")
    # Subsample negatives for faster training (keep all positives)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_neg_sample = min(len(neg_idx), len(pos_idx) * 5)  # 5:1 neg:pos ratio
    if n_neg_sample < len(neg_idx):
        neg_sample = np.random.RandomState(42).choice(neg_idx, n_neg_sample, replace=False)
        train_idx = np.concatenate([pos_idx, neg_sample])
    else:
        train_idx = np.arange(len(y_train))

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        C=0.1, max_iter=500, class_weight="balanced",
        solver="saga", random_state=42, n_jobs=-1
    )
    model.fit(X_tr_s, y_tr)
    pred = model.predict_proba(X_val_s)[:, 1]
    return pred, model, scaler


def train_sgd_pairwise(X_train, y_train, queries_train, X_val, y_val):
    """SGD pairwise proxy: sample pos-neg pairs within same query."""
    log("  Training SGD pairwise proxy...")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    # Build query→index mapping for fast same-query neg sampling
    q_to_pos = {}
    q_to_neg = {}
    for pi in pos_idx:
        q = queries_train[pi]
        q_to_pos.setdefault(q, []).append(pi)
    for ni in neg_idx:
        q = queries_train[ni]
        q_to_neg.setdefault(q, []).append(ni)

    # Sample pairs: equal positive pairs (pos > neg) and negative pairs (neg > pos)
    alignments = []
    np.random.seed(42)
    max_pairs = 200000
    for qid in q_to_pos:
        if qid not in q_to_neg:
            continue
        q_pos = q_to_pos[qid]
        q_neg = q_to_neg[qid]
        n_sample = min(len(q_pos), len(q_neg), max_pairs // len(q_to_pos) + 1)
        for pi in np.random.choice(q_pos, n_sample, replace=True):
            ni = np.random.choice(q_neg, 1)[0]
            # positive pair: pos - neg, label=1 (pos is better)
            alignments.append((pi, ni, 1))
            # negative pair: neg - pos, label=0
            alignments.append((ni, pi, 0))

    alignments = np.array(alignments[:max_pairs])
    X_pair = X_train[alignments[:, 0].astype(int)] - X_train[alignments[:, 1].astype(int)]
    y_pair = alignments[:, 2].astype(int)

    scaler = StandardScaler()
    X_pair_s = scaler.fit_transform(X_pair)
    X_val_s = scaler.transform(X_val)

    model = SGDClassifier(
        loss="modified_huber", penalty="l2", alpha=0.0001,
        max_iter=100, random_state=42
    )
    model.fit(X_pair_s, y_pair)
    pred = model.decision_function(X_val_s)
    return pred, model, scaler


def train_histgb(X_train, y_train, X_val, y_val):
    """HistGradientBoosting with subsampled negatives."""
    log("  Training HistGradientBoosting...")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_neg_sample = min(len(neg_idx), len(pos_idx) * 5)
    if n_neg_sample < len(neg_idx):
        neg_sample = np.random.RandomState(42).choice(neg_idx, n_neg_sample, replace=False)
        train_idx = np.concatenate([pos_idx, neg_sample])
    else:
        train_idx = np.arange(len(y_train))

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]

    model = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.1,
        early_stopping=False, random_state=42,
        class_weight="balanced",
    )
    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_val)[:, 1]
    return pred, model, None


def zscore_blend(scores1, scores2, alpha=0.5):
    """Z-score blend two score arrays."""
    z1 = (scores1 - np.mean(scores1)) / (np.std(scores1) + 1e-10)
    z2 = (scores2 - np.mean(scores2)) / (np.std(scores2) + 1e-10)
    return alpha * z1 + (1 - alpha) * z2


def rerank_within_baseline(df_val, scorer_col, baseline_col, top_k):
    """
    Rerank: keep baseline ordering, but within baseline top-K,
    re-rank by scorer. Candidates outside top-K keep baseline order.
    """
    df = df_val.copy()
    df["_baseline_rank"] = df.groupby("query_id")[baseline_col].rank(ascending=False)
    # Within top-K: use scorer score; outside: use negative rank (preserves order)
    df["_rerank_score"] = np.where(
        df["_baseline_rank"] <= top_k,
        df[scorer_col],
        -df["_baseline_rank"]
    )
    return df


def main():
    log("=" * 60)
    log("D4S28: Candidate-Level Transform Scorer")
    log("=" * 60)

    df = load_data()
    X, y, queries, feature_names, df_feat = prepare_features(df)

    # ── Verify D4S27 baseline ─────────────────────────────────────────
    log("\n=== VERIFY D4S27 BASELINE ===")
    eval_blend = evaluate_model("D4S27_blend", df, "blend_score")
    log(f"  baseline blend  Top10={eval_blend['Top10']:.6f}  "
        f"MRR={eval_blend['MRR']:.6f}")

    eval_cont = evaluate_model("D4S27_cont", df, "cont_prior_score")
    log(f"  cont_prior      Top10={eval_cont['Top10']:.6f}  "
        f"rescue={eval_cont['rescue_total']}")

    eval_backoff = evaluate_model("D4S27_backoff", df, "backoff_logit_score")
    log(f"  backoff         Top10={eval_backoff['Top10']:.6f}  "
        f"rescue={eval_backoff['rescue_total']}")

    # ── 5-fold GroupKFold CV ──────────────────────────────────────────
    log("\n=== 5-FOLD GROUP K-FOLD CV ===")
    gkf = GroupKFold(n_splits=5)
    unique_queries = df_feat["query_id"].unique()
    query_to_idx = {q: i for i, q in enumerate(df_feat["query_id"])}

    all_metrics = [eval_blend, eval_cont, eval_backoff]
    oof_preds_logreg = np.zeros(len(df))
    oof_preds_sgd = np.zeros(len(df))
    oof_preds_histgb = np.zeros(len(df))

    # GroupKFold.split needs X, y, groups — pass query_id as groups
    for fold, (train_q_idx, val_q_idx) in enumerate(gkf.split(unique_queries,
                                                                groups=unique_queries)):
        train_qs = set(unique_queries[train_q_idx])
        val_qs = set(unique_queries[val_q_idx])

        train_mask = df_feat["query_id"].isin(train_qs).values
        val_mask = df_feat["query_id"].isin(val_qs).values

        log(f"\n  Fold {fold+1}/5: train={train_mask.sum():,}  val={val_mask.sum():,}")

        # --- LogisticRegression ---
        pred_lr, _, _ = train_logreg(
            X[train_mask], y[train_mask],
            X[val_mask], y[val_mask]
        )
        oof_preds_logreg[val_mask] = pred_lr

        # --- SGD pairwise ---
        pred_sgd, _, _ = train_sgd_pairwise(
            X[train_mask], y[train_mask], queries[train_mask],
            X[val_mask], y[val_mask]
        )
        oof_preds_sgd[val_mask] = pred_sgd

        # --- HistGB ---
        pred_hgb, _, _ = train_histgb(
            X[train_mask], y[train_mask],
            X[val_mask], y[val_mask]
        )
        oof_preds_histgb[val_mask] = pred_hgb

    # ── Evaluate OOF predictions ──────────────────────────────────────
    log("\n=== OOF EVALUATION ===")

    df_eval = df_feat.copy()
    df_eval["scorer_logreg"] = oof_preds_logreg
    df_eval["scorer_sgd"] = oof_preds_sgd
    df_eval["scorer_histgb"] = oof_preds_histgb

    for name, col in [("LogReg", "scorer_logreg"),
                       ("SGD_pairwise", "scorer_sgd"),
                       ("HistGB", "scorer_histgb")]:
        eval_m = evaluate_model(f"D4S28_{name}", df_eval, col)
        log(f"  {name:20s}  Top10={eval_m['Top10']:.6f}  "
            f"d={eval_m['delta_Top10']:+.6f}  "
            f"rescue={eval_m['rescue_total']}  "
            f"lost={eval_m['hits_lost']}")
        all_metrics.append(eval_m)

        # scorer + blend zscore
        alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        best_alpha = None
        best_top10 = 0
        for alpha in alpha_grid:
            blended = zscore_blend(df_eval[col].values,
                                   df_eval["blend_score"].values, alpha)
            df_eval[f"{name}_blend_{alpha}"] = blended
            eval_za = evaluate_model(f"D4S28_{name}_zblend_a{alpha}", df_eval,
                                     f"{name}_blend_{alpha}")
            if eval_za["Top10"] > best_top10:
                best_top10 = eval_za["Top10"]
                best_alpha = alpha
        log(f"    +blend best: alpha={best_alpha} Top10={best_top10:.6f}")

        # scorer rerank within baseline Top20 / Top30
        for top_k in [20, 30]:
            df_rerank = rerank_within_baseline(df_eval, col, "blend_score", top_k)
            eval_rr = evaluate_model(
                f"D4S28_{name}_rerankTop{top_k}", df_rerank, "_rerank_score"
            )
            log(f"    rerank Top{top_k}: Top10={eval_rr['Top10']:.6f}  "
                f"d={eval_rr['delta_Top10']:+.6f}")
            all_metrics.append(eval_rr)

        # Best blend
        best_blend_eval = evaluate_model(
            f"D4S28_{name}_best_blend", df_eval,
            f"{name}_blend_{best_alpha}"
        )
        all_metrics.append(best_blend_eval)

    # ── Best model (HistGB) detailed strata ───────────────────────────
    log("\n=== STRATIFIED: Best Model (HistGB) ===")
    best_col = "scorer_histgb"
    for frag, grp in df_eval.groupby("old_fragment_smiles"):
        qdf = compute_metrics(grp, best_col)
        # also get baseline on this fragment
        qdf_bl = compute_metrics(grp, "blend_score")
        qdf_co = compute_metrics(grp, "cont_prior_score")
        qdf_bo = compute_metrics(grp, "backoff_logit_score")
        n = len(qdf)
        log(f"  {frag:25s}  n={n:5d}  "
            f"baseline={qdf_bl['Top10'].mean():.4f}  "
            f"scorer={qdf['Top10'].mean():.4f}  "
            f"cont={qdf_co['Top10'].mean():.4f}  "
            f"backoff={qdf_bo['Top10'].mean():.4f}")

    # ── Hard-miss stratified ──────────────────────────────────────────
    log("\n=== HARD-MISS STRATIFIED ===")
    # Need to load hard_miss from original parquet
    df_orig = pd.read_parquet(f"{D4S27_RESULTS}/candidate_prior_scores_val.parquet",
                              columns=["query_id", "hard_top10_miss_flag"])
    # Get unique query-level hard miss flag
    q_hard = df_orig.groupby("query_id")["hard_top10_miss_flag"].max().reset_index()
    df_eval_hm = df_eval.merge(q_hard, on="query_id", how="left")
    for hm_flag, label in [(0, "No"), (1, "Yes")]:
        grp = df_eval_hm[df_eval_hm["hard_top10_miss_flag"] == hm_flag]
        qdf = compute_metrics(grp, best_col)
        qdf_bl = compute_metrics(grp, "blend_score")
        n = len(qdf)
        log(f"  hard_miss={label}  n={n:5d}  "
            f"baseline={qdf_bl['Top10'].mean():.4f}  "
            f"scorer={qdf['Top10'].mean():.4f}")

    # ── Save outputs ──────────────────────────────────────────────────
    log("\n=== SAVING OUTPUTS ===")

    # summary_metrics.csv
    rows = []
    for m in all_metrics:
        rows.append({
            "model": m["model"],
            "n_queries": m["n_queries"],
            "Top1": m["Top1"], "Top5": m["Top5"],
            "Top10": m["Top10"], "Top20": m["Top20"],
            "MRR": m["MRR"],
            "delta_Top10": m["delta_Top10"],
            "rescue_total": m["rescue_total"],
            "rescue_11_20": m.get("rescue_11_20", 0),
            "rescue_20p": m.get("rescue_20p", 0),
            "hits_lost": m.get("hits_lost", 0),
        })
    sm = pd.DataFrame(rows)
    sm.to_csv(f"{OUT_DIR}/d4s28_summary_metrics.csv", index=False)
    log(f"  Saved d4s28_summary_metrics.csv ({len(rows)} methods)")

    # d4s28_query_all.csv (best model scores)
    qa = compute_metrics(df_eval, best_col)
    qa_bl = compute_metrics(df_eval, "blend_score")
    qa_co = compute_metrics(df_eval, "cont_prior_score")
    qa_bo = compute_metrics(df_eval, "backoff_logit_score")

    qa_out = qa[["query_id", "best_rank", "Top1", "Top5", "Top10", "Top20", "MRR"]].copy()
    qa_out.rename(columns={"best_rank": "best_rank_scorer",
                            "Top1": "Top1_scorer",
                            "Top5": "Top5_scorer",
                            "Top10": "Top10_scorer",
                            "Top20": "Top20_scorer",
                            "MRR": "MRR_scorer"}, inplace=True)
    for name, qdf_ref in [("blend", qa_bl), ("cont", qa_co), ("backoff", qa_bo)]:
        for col_ref in ["best_rank", "Top1", "Top5", "Top10", "Top20", "MRR"]:
            qa_out[f"{col_ref}_{name}"] = qdf_ref[col_ref].values
    qa_out.to_csv(f"{OUT_DIR}/d4s28_query_all.csv", index=False)
    log(f"  Saved d4s28_query_all.csv ({len(qa_out)} queries)")

    # d4s28_candidate_oof.csv.gz — use concatenation not merge (avoids memory blowup)
    df_oof_scores = df_eval[["query_id", "label", "blend_score",
                              "cont_prior_score", "backoff_logit_score",
                              "scorer_logreg", "scorer_sgd", "scorer_histgb"]].copy()
    df_meta = pd.read_parquet(
        f"{D4S27_RESULTS}/candidate_prior_scores_val.parquet",
        columns=["candidate_id", "candidate_smiles",
                 "old_fragment_smiles", "replacement_frequency_bin",
                 "hard_top10_miss_flag"]
    )
    # Concat horizontally (same row order, no merge needed)
    df_oof = pd.concat([df_oof_scores.reset_index(drop=True),
                         df_meta.reset_index(drop=True)], axis=1)
    df_oof.to_csv(f"{OUT_DIR}/d4s28_candidate_oof.csv.gz",
                  index=False, compression="gzip")
    log(f"  Saved d4s28_candidate_oof.csv.gz ({len(df_oof):,} rows)")

    # ── Final summary ─────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("D4S28 SUMMARY")
    log("=" * 60)
    log(f"  Baseline Top10: 0.834019")
    for m in all_metrics:
        if "D4S28_" in m["model"] and "blend" not in m["model"]:
            log(f"  {m['model']:40s} Top10={m['Top10']:.6f}  "
                f"d={m['delta_Top10']:+.6f}  "
                f"rescue={m.get('rescue_total',0)}  lost={m.get('hits_lost',0)}")

    beat = any(m["delta_Top10"] > 0 for m in all_metrics if "D4S28_" in m["model"])
    if beat:
        best = max((m for m in all_metrics if "D4S28_" in m["model"]),
                   key=lambda m: m["Top10"])
        log(f"\n  BEAT BASELINE: {best['model']} Top10={best['Top10']:.6f}")
    else:
        log(f"\n  NO METHOD BEAT BASELINE (0.834019)")
        # Check *Cc1ccccc1 improvement
        cc1 = df_eval[df_eval["old_fragment_smiles"] == "*Cc1ccccc1"]
        qdf_cc1_scorer = compute_metrics(cc1, best_col)
        qdf_cc1_bl = compute_metrics(cc1, "blend_score")
        log(f"  *Cc1ccccc1: baseline={qdf_cc1_bl['Top10'].mean():.4f}  "
            f"scorer={qdf_cc1_scorer['Top10'].mean():.4f}")

        # Check other fragments not degraded
        other = df_eval[df_eval["old_fragment_smiles"] != "*Cc1ccccc1"]
        qdf_other_scorer = compute_metrics(other, best_col)
        qdf_other_bl = compute_metrics(other, "blend_score")
        log(f"  Other fragments: baseline={qdf_other_bl['Top10'].mean():.4f}  "
            f"scorer={qdf_other_scorer['Top10'].mean():.4f}")

    log("\nD4S28 COMPLETE.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
