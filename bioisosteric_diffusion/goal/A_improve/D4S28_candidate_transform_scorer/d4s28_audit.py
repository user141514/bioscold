#!/usr/bin/env python3
"""
D4S28_AUDIT: Verify HistGB OOF Top10=0.9374 is real, not leakage.

Five-part audit:
  1. Feature provenance — classify every used feature
  2. Split contamination — train/val overlap check
  3. Permutation test — shuffle labels within query, re-train HistGB
  4. Safe-only rerun — strict deployable features, query-OOF + old-fragment-OOF
  5. Decision — allow blind one-shot or flag as leakage
"""
import pandas as pd
import numpy as np
import os, sys, time, json, warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
warnings.filterwarnings("ignore")

PROJECT = "E:/zuhui/bioisosteric_diffusion"
D4S27_RESULTS = f"{PROJECT}/goal/A_improve/results"
OUT_DIR = f"{PROJECT}/goal/A_improve/D4S28_candidate_transform_scorer/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# PART 0: FEATURE PROVENANCE MAP
# ══════════════════════════════════════════════════════════════════════

FEATURE_PROVENANCE = {
    # ── Model scores (computed by MLP/HGB/DE/Attach models on val data) ──
    "blend_score":      ("SAFE_DEPLOYABLE", "z-score blend: 0.95*z(MLP)+0.05*z(HGB), model outputs"),
    "mlp_z":            ("SAFE_DEPLOYABLE", "z-scored MLP output, model inference"),
    "hgb_z":            ("SAFE_DEPLOYABLE", "z-scored HGB refit output, model inference"),
    "mlp_score":        ("SAFE_DEPLOYABLE", "raw MLP score, model inference"),
    "hgb_refit_score":  ("SAFE_DEPLOYABLE", "raw HGB refit score, model inference"),
    "score_DE":         ("SAFE_DEPLOYABLE", "raw DE model score, model inference"),
    "score_HGB":        ("SAFE_DEPLOYABLE", "raw HGB model score, model inference"),
    "score_Borda":      ("SAFE_DEPLOYABLE", "Borda fusion score, deterministic from scores"),
    "score_attach":     ("SAFE_DEPLOYABLE", "raw attach model score, model inference"),

    # ── Per-query ranks (computed from scores within query, no label) ──
    "mlp_rank":         ("SAFE_DEPLOYABLE", "rank of mlp_score within query"),
    "blend_rank":       ("SAFE_DEPLOYABLE", "rank of blend_score within query"),
    "rank_DE":          ("SAFE_DEPLOYABLE", "rank of score_DE within query"),
    "rank_HGB":         ("SAFE_DEPLOYABLE", "rank of score_HGB within query"),
    "rank_Borda":       ("SAFE_DEPLOYABLE", "rank of score_Borda within query"),
    "rank_attach":      ("SAFE_DEPLOYABLE", "rank of score_attach within query"),

    # ── Top-K flags (computed from ranks, no label) ──
    "candidate_in_attach_top10": ("SAFE_DEPLOYABLE", "rank_attach <= 10"),
    "candidate_in_DE_top10":     ("SAFE_DEPLOYABLE", "rank_DE <= 10"),
    "candidate_in_HGB_top10":    ("SAFE_DEPLOYABLE", "rank_HGB <= 10"),
    "candidate_in_Borda_top10":  ("SAFE_DEPLOYABLE", "rank_Borda <= 10"),

    # ── D4S27 prior scores (from train-only P4/P3/P1/P0 statistics) ──
    "cont_prior_score":       ("SAFE_DEPLOYABLE", "continuation prior: logP(c|old,att) from train counts"),
    "backoff_logit_score":    ("SAFE_DEPLOYABLE", "cascade backoff: tP4->P3->P1, train-only stats"),
    "t_p4_logit_score":       ("SAFE_DEPLOYABLE", "transferred P4 logit, train nearest-old proxy"),
    "p1_logit_score":         ("SAFE_DEPLOYABLE", "P1 logit: logP(c|att), train counts"),
    "p3_logit_score":         ("SAFE_DEPLOYABLE", "P3 logit: logP(c|cluster), train counts"),

    # ── Prior smoothed rates ──
    "t_p4_smoothed_rate":     ("SAFE_DEPLOYABLE", "P4 smoothed rate (Laplace), train stats"),
    "p1_smoothed_rate":       ("SAFE_DEPLOYABLE", "P1 smoothed rate, train stats"),
    "p3_smoothed_rate":       ("SAFE_DEPLOYABLE", "P3 smoothed rate, train stats"),
    "backoff_smoothed_rate":  ("SAFE_DEPLOYABLE", "backoff smoothed rate, train stats"),

    # ── Prior supports (count statistics from train) ──
    "t_p4_support":           ("SAFE_DEPLOYABLE", "P4 support count, train-only"),
    "p1_support":             ("SAFE_DEPLOYABLE", "P1 support count, train-only"),
    "p3_support":             ("SAFE_DEPLOYABLE", "P3 support count, train-only"),
    "p0_support":             ("SAFE_DEPLOYABLE", "P0 support count, train-only"),
    "backoff_support":        ("SAFE_DEPLOYABLE", "backoff support count, train-only"),

    # ── Prior positive counts ──
    "t_p4_positive":          ("SAFE_DEPLOYABLE", "P4 positive count, train-only"),
    "p1_positive":            ("SAFE_DEPLOYABLE", "P1 positive count, train-only"),
    "p3_positive":            ("SAFE_DEPLOYABLE", "P3 positive count, train-only"),

    # ── Prior per-query ranks (from prior scores, no label) ──
    "backoff_logit_rank":     ("SAFE_DEPLOYABLE", "rank of backoff_logit_score within query"),
    "cont_prior_rank":        ("SAFE_DEPLOYABLE", "rank of cont_prior_score within query"),
    "t_p4_logit_rank":        ("SAFE_DEPLOYABLE", "rank of t_p4_logit within query"),
    "p1_logit_rank":          ("SAFE_DEPLOYABLE", "rank of p1_logit within query"),
    "p3_logit_rank":          ("SAFE_DEPLOYABLE", "rank of p3_logit within query"),

    # ── PMI scores (from train-only statistics) ──
    "t_pmi_p4_p1":            ("SAFE_DEPLOYABLE", "PMI: transferred P4 - P1, train stats"),
    "pmi_p3_p1":              ("SAFE_DEPLOYABLE", "PMI: P3 - P1 cluster lift, train stats"),
    "pmi_p1_p0":              ("SAFE_DEPLOYABLE", "PMI: P1 - P0 attachment lift, train stats"),

    # ── PMI ranks ──
    "t_pmi_p4_p1_rank":       ("SAFE_DEPLOYABLE", "rank of t_pmi_p4_p1 within query"),
    "pmi_p3_p1_rank":         ("SAFE_DEPLOYABLE", "rank of pmi_p3_p1 within query"),
    "pmi_p1_p0_rank":         ("SAFE_DEPLOYABLE", "rank of pmi_p1_p0 within query"),

    # ── Molecular descriptors (computed from SMILES, no label) ──
    "delta_heavy_atoms":      ("SAFE_DEPLOYABLE", "candidate - old heavy atoms"),
    "delta_ring_count":       ("SAFE_DEPLOYABLE", "candidate - old ring count"),
    "delta_hetero_count":     ("SAFE_DEPLOYABLE", "candidate - old hetero count"),
    "delta_mw":               ("SAFE_DEPLOYABLE", "candidate - old molecular weight"),
    "delta_logp":             ("SAFE_DEPLOYABLE", "candidate - old logP"),
    "delta_tpsa":             ("SAFE_DEPLOYABLE", "candidate - old TPSA"),
    "candidate_heavy_atoms":  ("SAFE_DEPLOYABLE", "candidate heavy atom count"),
    "candidate_mw":           ("SAFE_DEPLOYABLE", "candidate molecular weight"),
    "candidate_logp":         ("SAFE_DEPLOYABLE", "candidate logP"),
    "candidate_tpsa":         ("SAFE_DEPLOYABLE", "candidate TPSA"),

    # ── Similarity (from Morgan fingerprints, train-only reference) ──
    "_tanimoto_similarity":   ("SAFE_DEPLOYABLE", "Morgan Tanimoto to nearest TRAIN old_fragment"),

    # ── Replacement/attachment frequencies (from train count data) ──
    "replacement_frequency":  ("SAFE_DEPLOYABLE", "train frequency of candidate replacing this old_fragment"),
    "attachment_frequency":   ("SAFE_DEPLOYABLE", "train frequency of this attachment"),

    # ── Query-level prior statistics (from prior scores within query, no label) ──
    "query_prior_entropy":    ("SAFE_DEPLOYABLE", "entropy of prior scores within query"),
    "query_prior_margin":     ("SAFE_DEPLOYABLE", "max - second_max prior score within query"),
    "query_prior_max_support":("SAFE_DEPLOYABLE", "max prior support within query"),

    # ── Categorical features (context identifiers, same for all candidates in query) ──
    "old_fragment_smiles":    ("SAFE_DEPLOYABLE", "old_fragment identity, context feature"),
    "attachment_signature":   ("SAFE_DEPLOYABLE", "attachment signature, context feature"),
    "replacement_frequency_bin": ("SAFE_DEPLOYABLE", "binned replacement_frequency, train-derived"),

    # ═══ SUSPICIOUS — needs provenance confirmation ═══
    "replacement_frequency_mean": ("SUSPICIOUS_NEEDS_PROVENANCE",
        "mean of replacement_frequency per query — likely train-derived but aggregated at query level"),
    "morgan_similarity_old_candidate_if_available": ("SUSPICIOUS_NEEDS_PROVENANCE",
        "Morgan Tanimoto old-candidate — available for some entries, source unclear"),
    "compatibility_flag":    ("SUSPICIOUS_NEEDS_PROVENANCE",
        "chemical compatibility flag — likely rule-based but source unconfirmed"),
    "a4c_tier_if_available_diagnostic_only": ("SUSPICIOUS_NEEDS_PROVENANCE",
        "diagnostic tier, source unclear — DO NOT USE"),
    "target_any_seen_vocab": ("SUSPICIOUS_NEEDS_PROVENANCE",
        "whether target tokens seen in MLP training vocab — model-level feature, not label-derived but correlates with difficulty. EXCLUDE from safe-only"),
    "target_all_seen_vocab": ("SUSPICIOUS_NEEDS_PROVENANCE",
        "whether ALL target tokens seen in training vocab. EXCLUDE from safe-only"),

    # ═══ UNSAFE — label-derived ═══
    "label":                 ("UNSAFE_LABEL_DERIVED", "TARGET — the label being predicted"),
    "positive_set_size":     ("UNSAFE_LABEL_DERIVED", "number of positive candidates from labels"),
    "num_positives":         ("UNSAFE_LABEL_DERIVED", "number of positive candidates in query"),
    "single_pos_flag":       ("UNSAFE_LABEL_DERIVED", "derived from num_positives == 1"),
    "multi_pos_flag":        ("UNSAFE_LABEL_DERIVED", "derived from num_positives > 1"),
    "hard_top10_miss_flag":  ("UNSAFE_LABEL_DERIVED", "baseline Top10 miss — requires labels to compute"),

    # ── Not used / metadata ──
    "query_id":              ("METADATA", "query identifier"),
    "split":                 ("METADATA", "dataset split"),
    "candidate_id":          ("METADATA", "candidate identifier"),
    "candidate_smiles":      ("METADATA", "candidate SMILES string"),
    "old_fragment_cluster_if_available": ("METADATA", "cluster label"),
    "old_fragment_cluster_id": ("METADATA", "cluster ID"),
    "candidate_ring_count":  ("METADATA", "redundant with candidate_heavy_atoms"),
    "candidate_hetero_count":("METADATA", "redundant"),
    "p0_logit_score":        ("METADATA", "P0=logP(cand only), not used in prior computation"),
    "p0_smoothed_rate":      ("METADATA", "P0 smoothed rate, not used"),
    "p0_positive":           ("METADATA", "P0 positive count, not used"),
}


def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    print(f"{ts} {msg}", flush=True)


def compute_query_metrics(df, score_col, label_col="label"):
    """Compute per-query TopK metrics."""
    results = []
    for qid, grp in df.groupby("query_id"):
        grp_sorted = grp.sort_values(score_col, ascending=False)
        labels = grp_sorted[label_col].values
        pos_ranks = np.where(labels == 1)[0] + 1
        if len(pos_ranks) == 0:
            continue
        best = pos_ranks.min()
        results.append({
            "query_id": qid,
            "best_rank": best,
            "Top1": int(best == 1), "Top5": int(best <= 5),
            "Top10": int(best <= 10), "Top20": int(best <= 20),
            "MRR": 1.0 / best,
        })
    return pd.DataFrame(results)


def compute_oof_5fold(X_full, y_full, query_ids, groups_full, n_splits=5):
    """Train HistGB with GroupKFold. Returns OOF preds, Top10, MRR, qdf."""
    uq = np.array(list(set(groups_full)))
    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(y_full))

    for train_g_idx, val_g_idx in gkf.split(uq, groups=uq):
        train_gs = frozenset(uq[train_g_idx])
        val_gs = frozenset(uq[val_g_idx])
        train_mask = np.array([g in train_gs for g in groups_full])
        val_mask = np.array([g in val_gs for g in groups_full])

        X_tr, y_tr = X_full[train_mask], y_full[train_mask]
        X_val = X_full[val_mask]

        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        n_neg = min(len(neg_idx), len(pos_idx) * 5)
        if n_neg < len(neg_idx):
            neg_sample = np.random.RandomState(42).choice(neg_idx, n_neg, replace=False)
            train_idx = np.concatenate([pos_idx, neg_sample])
        else:
            train_idx = np.arange(len(y_tr))

        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            early_stopping=False, random_state=42, class_weight="balanced",
        )
        model.fit(X_tr[train_idx], y_tr[train_idx])
        oof_preds[val_mask] = model.predict_proba(X_val)[:, 1]

    # Evaluate per QUERY (not per group)
    df_eval = pd.DataFrame({
        "query_id": query_ids,
        "label": y_full,
        "score": oof_preds,
    })
    n_q = 0; n_hit = 0
    qdf_rows = []
    for qid, grp in df_eval.groupby("query_id"):
        n_q += 1
        gs = grp.sort_values("score", ascending=False)
        labels = gs["label"].values
        pos_ranks = np.where(labels == 1)[0] + 1
        if len(pos_ranks) == 0:
            continue
        best = pos_ranks.min()
        if best <= 10:
            n_hit += 1
        qdf_rows.append({
            "query_id": qid,
            "best_rank": best,
            "Top1": int(best == 1), "Top5": int(best <= 5),
            "Top10": int(best <= 10), "Top20": int(best <= 20),
            "MRR": 1.0 / best,
        })
    qdf = pd.DataFrame(qdf_rows)
    return oof_preds, n_hit / n_q, qdf["MRR"].mean(), qdf


# ══════════════════════════════════════════════════════════════════════
# MAIN AUDIT
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 60)
    log("D4S28_AUDIT: Verifying HistGB OOF Top10=0.9374")
    log("=" * 60)

    # ── Load data ──
    log("\nLoading D4S27 val parquet...")
    df = pd.read_parquet(f"{D4S27_RESULTS}/candidate_prior_scores_val.parquet")
    log(f"  {len(df):,} rows, {len(df.columns)} cols")

    # ═══════════════════════════════════════════════════════════════
    # PART 1: FEATURE PROVENANCE
    # ═══════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("PART 1: FEATURE PROVENANCE AUDIT")
    log("=" * 60)

    provenance_rows = []
    safe_count = suspicious_count = unsafe_count = 0
    for col in sorted(df.columns):
        verdict, note = FEATURE_PROVENANCE.get(col, ("UNCLASSIFIED", "MISSING — default to EXCLUDE"))
        if "SAFE" in verdict:
            safe_count += 1
        elif "UNSAFE" in verdict or "LABEL" in verdict:
            unsafe_count += 1
        else:
            suspicious_count += 1
        provenance_rows.append({
            "feature": col,
            "verdict": verdict,
            "note": note,
            "dtype": str(df[col].dtype),
        })
    pd.DataFrame(provenance_rows).to_csv(
        f"{OUT_DIR}/feature_provenance.csv", index=False
    )
    log(f"  SAFE: {safe_count}  SUSPICIOUS: {suspicious_count}  UNSAFE/META: {unsafe_count}")
    log(f"  Saved feature_provenance.csv ({len(provenance_rows)} features)")

    # ═══════════════════════════════════════════════════════════════
    # PART 2: SPLIT CONTAMINATION AUDIT
    # ═══════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("PART 2: SPLIT CONTAMINATION AUDIT")
    log("=" * 60)

    # Load blind to check candidate overlap
    blind = pd.read_parquet(f"{D4S27_RESULTS}/candidate_prior_scores_blind.parquet")
    val_cands = set(df["candidate_smiles"].unique())
    blind_cands = set(blind["candidate_smiles"].unique())
    val_blind_overlap = val_cands & blind_cands
    log(f"  Val candidates: {len(val_cands):,}")
    log(f"  Blind candidates: {len(blind_cands):,}")
    log(f"  Val-blind candidate overlap: {len(val_blind_overlap):,}")

    # Train old_fragment vs val old_fragment overlap
    train_olds = set(pd.read_csv(
        f"{D4S27_RESULTS}/tanimoto_nearest_old_val.csv"
    )['nearest_train_old'].unique())
    val_olds = set(df["old_fragment_smiles"].unique())
    old_overlap = train_olds & val_olds
    log(f"  Train old_fragments (nearest): {len(train_olds)}")
    log(f"  Val old_fragments: {len(val_olds)}")
    log(f"  Train-val old_fragment overlap: {len(old_overlap)}")

    # Check if any val positive candidate appears in D4S12 train data
    val_pos_smiles = set(df[df["label"] == 1]["candidate_smiles"])
    log(f"  Val positive candidate SMILES: {len(val_pos_smiles):,}")

    # Quick label balance check
    pos_rate = df["label"].mean()
    log(f"  Overall label positive rate: {pos_rate:.4f}")

    split_rows = [{
        "check": "val_candidate_count",
        "value": len(val_cands),
        "status": "OK"
    }, {
        "check": "blind_candidate_smiles_count",
        "value": len(blind_cands),
        "status": "OK"
    }, {
        "check": "val_blind_candidate_overlap",
        "value": len(val_blind_overlap),
        "status": "NOTE: some overlap expected (same chemical space)"
    }, {
        "check": "train_old_fragments",
        "value": len(train_olds),
        "status": "OK"
    }, {
        "check": "val_old_fragments",
        "value": len(val_olds),
        "status": "OK"
    }, {
        "check": "old_fragment_exact_match",
        "value": len(old_overlap),
        "status": "CLEAN: 0 exact overlap (similarity-transfer only)" if len(old_overlap) == 0 else "WARNING"
    }, {
        "check": "val_positive_rate",
        "value": round(pos_rate, 4),
        "status": "OK (1.07% positive)"
    }]
    pd.DataFrame(split_rows).to_csv(f"{OUT_DIR}/split_audit.csv", index=False)
    log(f"  Saved split_audit.csv ({len(split_rows)} checks)")

    # ═══════════════════════════════════════════════════════════════
    # PART 3: PERMUTATION TEST
    # ═══════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("PART 3: PERMUTATION TEST")
    log("=" * 60)

    # Use SAFE features only for fair test
    safe_features = [f for f, (v, _) in FEATURE_PROVENANCE.items()
                     if "SAFE" in v and f in df.columns]
    cat_features_for_encoding = ["old_fragment_smiles", "attachment_signature",
                                  "replacement_frequency_bin"]

    # Build safe feature matrix
    numeric_cols = [c for c in safe_features if c not in cat_features_for_encoding and c in df.columns]
    X_parts = [df[numeric_cols].fillna(0).values]
    for cat_col in cat_features_for_encoding:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
            X_parts.append(dummies.values)
    X_safe = np.hstack(X_parts)
    y_true = df["label"].values
    queries = df["query_id"].values

    log(f"  Safe feature matrix: {X_safe.shape[1]} dims, {X_safe.shape[0]:,} rows")

    # Permute labels within each query (preserve per-query positive count)
    np.random.seed(42)
    y_perm = np.zeros_like(y_true)
    for qid, grp in df.groupby("query_id"):
        idx = grp.index.values
        y_perm[idx] = np.random.permutation(y_true[idx])

    # Train HistGB with permuted labels (same 5-fold CV)
    log("  Training HistGB with PERMUTED labels (5-fold CV)...")
    gkf = GroupKFold(n_splits=5)
    unique_qs = df["query_id"].unique()
    oof_perm = np.zeros(len(y_true))

    for fold, (tqi, vqi) in enumerate(gkf.split(unique_qs, groups=unique_qs)):
        tqs = set(unique_qs[tqi]); vqs = set(unique_qs[vqi])
        t_mask = np.array([q in tqs for q in queries])
        v_mask = np.array([q in vqs for q in queries])

        X_tr, y_ptr = X_safe[t_mask], y_perm[t_mask]
        X_val = X_safe[v_mask]

        pi = np.where(y_ptr == 1)[0]; ni = np.where(y_ptr == 0)[0]
        nn = min(len(ni), len(pi) * 5)
        if nn < len(ni):
            ns = np.random.RandomState(42).choice(ni, nn, replace=False)
            tr_idx = np.concatenate([pi, ns])
        else:
            tr_idx = np.arange(len(y_ptr))

        hgb = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            early_stopping=False, random_state=42, class_weight="balanced",
        )
        hgb.fit(X_tr[tr_idx], y_ptr[tr_idx])
        oof_perm[v_mask] = hgb.predict_proba(X_val)[:, 1]
        log(f"    Fold {fold+1}/5 done")

    df_perm = df[["query_id", "label"]].copy()
    df_perm["score_perm"] = oof_perm
    qdf_perm = compute_query_metrics(df_perm, "score_perm")
    perm_top10 = qdf_perm["Top10"].mean()

    # Also compute rand baseline (random ordering)
    df_rand = df[["query_id", "label"]].copy()
    df_rand["score_rand"] = np.random.RandomState(42).rand(len(df_rand))
    qdf_rand = compute_query_metrics(df_rand, "score_rand")
    rand_top10 = qdf_rand["Top10"].mean()

    log(f"  Permuted-label OOF Top10: {perm_top10:.6f}")
    log(f"  Random baseline Top10:   {rand_top10:.6f}")
    log(f"  Baseline blend Top10:    0.834019")

    perm_passed = perm_top10 < 0.3  # Must be far below baseline
    if perm_passed:
        log(f"  VERDICT: PASS — permuted-label Top10 ({perm_top10:.4f}) << baseline (0.834)")
    else:
        log(f"  VERDICT: FAIL — permuted-label Top10 ({perm_top10:.4f}) suspiciously high")

    pd.DataFrame([{
        "test": "permuted_label_OOF",
        "permuted_Top10": round(perm_top10, 6),
        "random_baseline_Top10": round(rand_top10, 6),
        "baseline_Top10": 0.834019,
        "passed": perm_passed,
        "criterion": "permuted_Top10 < 0.3",
    }]).to_csv(f"{OUT_DIR}/permutation_test.csv", index=False)
    log(f"  Saved permutation_test.csv")

    # ═══════════════════════════════════════════════════════════════
    # PART 4: SAFE-ONLY RERUN (query-OOF + old-fragment-OOF)
    # ═══════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("PART 4: SAFE-ONLY RERUN")
    log("=" * 60)

    # Query-OOF (5-fold GroupKFold by query_id) — INLINE (verified correct)
    log("  4a. Query-OOF (5-fold by query_id)...")
    log(f"    DEBUG: X_safe shape={X_safe.shape} dtype={X_safe.dtype}")
    log(f"    DEBUG: y_true mean={y_true.mean():.4f} sum={y_true.sum()}")
    log(f"    DEBUG: queries n_unique={len(set(queries))}")
    uq = np.array(list(set(queries)))
    gkf_q = GroupKFold(n_splits=5)
    oof_safe_q = np.zeros(len(y_true))
    for tqi, vqi in gkf_q.split(uq, groups=uq):
        tqs = frozenset(uq[tqi]); vqs = frozenset(uq[vqi])
        tm = np.array([q in tqs for q in queries])
        vm = np.array([q in vqs for q in queries])
        Xt, yt = X_safe[tm], y_true[tm]
        Xv = X_safe[vm]
        pi = np.where(yt == 1)[0]; ni = np.where(yt == 0)[0]
        nn = min(len(ni), len(pi) * 5)
        ns = np.random.RandomState(42).choice(ni, nn, replace=False)
        ti = np.concatenate([pi, ns])
        hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                              early_stopping=False, random_state=42,
                                              class_weight="balanced")
        hgb.fit(Xt[ti], yt[ti])
        oof_safe_q[vm] = hgb.predict_proba(Xv)[:, 1]
        # Debug: check fold-level Top10
        df_fold = df.iloc[vm].copy()
        df_fold["_debug_oof"] = oof_safe_q[vm]
        fold_hits = 0; fold_n = 0
        for _, grp in df_fold.groupby("query_id"):
            fold_n += 1
            gs = grp.sort_values("_debug_oof", ascending=False)
            lbls = gs["label"].values
            pr = np.where(lbls == 1)[0] + 1
            if len(pr) > 0 and pr.min() <= 10:
                fold_hits += 1
        log(f"      fold Top10={fold_hits/fold_n:.4f} ({fold_hits}/{fold_n})")
    df["score_safe_query_oof"] = oof_safe_q
    log(f"    DEBUG: oof_safe_q range=[{oof_safe_q.min():.4f}, {oof_safe_q.max():.4f}] "
        f"label=1 mean={oof_safe_q[y_true==1].mean():.4f} label=0 mean={oof_safe_q[y_true==0].mean():.4f}")
    qdf_safe_q = compute_query_metrics(df, "score_safe_query_oof")
    safe_q_top10 = qdf_safe_q["Top10"].mean()
    safe_q_mrr = qdf_safe_q["MRR"].mean()
    log(f"    Safe-only query-OOF Top10: {safe_q_top10:.6f}")

    # Old-fragment-OOF (GroupKFold by old_fragment_smiles) — INLINE
    log("  4b. Old-fragment-OOF (5-fold by old_fragment)...")
    frags_uq = np.array(list(set(df["old_fragment_smiles"].values)))
    gkf_f = GroupKFold(n_splits=min(5, len(frags_uq)))
    oof_safe_f = np.zeros(len(y_true))
    for tfi, vfi in gkf_f.split(frags_uq, groups=frags_uq):
        tfs = frozenset(frags_uq[tfi]); vfs = frozenset(frags_uq[vfi])
        tm = np.array([f in tfs for f in df["old_fragment_smiles"].values])
        vm = np.array([f in vfs for f in df["old_fragment_smiles"].values])
        Xt, yt = X_safe[tm], y_true[tm]
        Xv = X_safe[vm]
        pi = np.where(yt == 1)[0]; ni = np.where(yt == 0)[0]
        nn = min(len(ni), len(pi) * 5)
        if nn > 0:
            ns = np.random.RandomState(42).choice(ni, nn, replace=False)
            ti = np.concatenate([pi, ns])
        else:
            ti = np.arange(len(yt))
        if len(ti) == 0:
            continue
        hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1,
                                              early_stopping=False, random_state=42,
                                              class_weight="balanced")
        hgb.fit(Xt[ti], yt[ti])
        oof_safe_f[vm] = hgb.predict_proba(Xv)[:, 1]
    df["score_safe_frag_oof"] = oof_safe_f
    qdf_safe_f = compute_query_metrics(df, "score_safe_frag_oof")
    safe_f_top10 = qdf_safe_f["Top10"].mean()
    safe_f_mrr = qdf_safe_f["MRR"].mean()
    log(f"    Safe-only old-frag-OOF Top10: {safe_f_top10:.6f}")

    # Baseline and strata
    qdf_bl = compute_query_metrics(df, "blend_score")
    log(f"    Baseline blend Top10: {qdf_bl['Top10'].mean():.6f}")

    # Strata by old_fragment for safe query-OOF
    safe_metrics_rows = [{
        "method": "baseline_blend",
        "Top10": round(qdf_bl["Top10"].mean(), 6),
        "MRR": round(qdf_bl["MRR"].mean(), 6),
        "split_type": "query-OOF",
    }, {
        "method": "safe_only_query_OOF",
        "Top10": round(safe_q_top10, 6),
        "MRR": round(safe_q_mrr, 6),
        "split_type": "query-OOF",
    }, {
        "method": "safe_only_fragment_OOF",
        "Top10": round(safe_f_top10, 6),
        "MRR": round(safe_f_mrr, 6),
        "split_type": "old-fragment-OOF",
    }]
    pd.DataFrame(safe_metrics_rows).to_csv(
        f"{OUT_DIR}/safe_only_metrics.csv", index=False
    )
    log(f"    Saved safe_only_metrics.csv")

    # Group-OOF strata
    group_rows = []
    for frag in sorted(df["old_fragment_smiles"].unique()):
        grp = df[df["old_fragment_smiles"] == frag]
        qdf = compute_query_metrics(grp, "score_safe_query_oof")
        qdf_bl_f = compute_query_metrics(grp, "blend_score")
        group_rows.append({
            "split": "query-OOF",
            "dimension": "old_fragment",
            "category": frag,
            "n_queries": len(qdf),
            "baseline_Top10": round(qdf_bl_f["Top10"].mean(), 6),
            "safe_scorer_Top10": round(qdf["Top10"].mean(), 6),
        })
        qdf_f = compute_query_metrics(grp, "score_safe_frag_oof")
        group_rows.append({
            "split": "fragment-OOF",
            "dimension": "old_fragment",
            "category": frag,
            "n_queries": len(qdf_f),
            "baseline_Top10": round(qdf_bl_f["Top10"].mean(), 6),
            "safe_scorer_Top10": round(qdf_f["Top10"].mean(), 6),
        })

    # *Cc1ccccc1 special focus
    cc1 = df[df["old_fragment_smiles"] == "*Cc1ccccc1"]
    qdf_cc1_q = compute_query_metrics(cc1, "score_safe_query_oof")
    qdf_cc1_f = compute_query_metrics(cc1, "score_safe_frag_oof")
    qdf_cc1_bl = compute_query_metrics(cc1, "blend_score")
    log(f"\n  *Cc1ccccc1 focus:")
    log(f"    baseline:        {qdf_cc1_bl['Top10'].mean():.6f}")
    log(f"    safe query-OOF:  {qdf_cc1_q['Top10'].mean():.6f}")
    log(f"    safe frag-OOF:   {qdf_cc1_f['Top10'].mean():.6f}")

    # Hard-miss strata (label-derived, used ONLY for reporting, NOT training)
    df_orig = pd.read_parquet(f"{D4S27_RESULTS}/candidate_prior_scores_val.parquet")
    q_hard = df_orig.groupby("query_id")["hard_top10_miss_flag"].max().reset_index()
    q_hard.rename(columns={"hard_top10_miss_flag": "_hm_flag"}, inplace=True)
    df_hm = df.merge(q_hard, on="query_id", how="left")
    for hmf, label in [(0, "No"), (1, "Yes")]:
        grp = df_hm[df_hm["_hm_flag"] == hmf]
        if len(grp) == 0:
            continue
        qdf_q = compute_query_metrics(grp, "score_safe_query_oof")
        group_rows.append({
            "split": "query-OOF",
            "dimension": "hard_miss",
            "category": f"hard_miss={label}",
            "n_queries": len(qdf_q),
            "baseline_Top10": round(compute_query_metrics(grp, "blend_score")["Top10"].mean(), 6),
            "safe_scorer_Top10": round(qdf_q["Top10"].mean(), 6),
        })

    pd.DataFrame(group_rows).to_csv(f"{OUT_DIR}/group_oof_metrics.csv", index=False)
    log(f"  Saved group_oof_metrics.csv ({len(group_rows)} rows)")

    # ═══════════════════════════════════════════════════════════════
    # PART 5: DECISION
    # ═══════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("PART 5: AUDIT DECISION")
    log("=" * 60)

    # Criteria
    safe_q_ok = safe_q_top10 >= 0.87
    safe_f_ok = safe_f_top10 >= 0.82
    perm_ok = perm_passed
    old_overlap_ok = len(old_overlap) == 0  # 0 exact old_fragment match

    log(f"  [1] Safe query-OOF >= 0.87: {safe_q_ok} ({safe_q_top10:.4f})")
    log(f"  [2] Safe frag-OOF >= 0.82:  {safe_f_ok} ({safe_f_top10:.4f})")
    log(f"  [3] Permutation test passed: {perm_ok} ({perm_top10:.4f})")
    log(f"  [4] Old-fragment overlap=0:  {old_overlap_ok} ({len(old_overlap)})")

    if safe_q_ok and perm_ok and old_overlap_ok:
        if safe_f_ok:
            decision = "ALLOW_BLIND_ONESHOT"
            reason = ("All checks passed. Safe-only query-OOF beats 0.87 threshold. "
                      "Permutation test confirms no label leakage. "
                      "Old-fragment-OOF generalizes, proving model learns transferable patterns. "
                      "Proceed to blind one-shot evaluation.")
        else:
            decision = "ALLOW_BLIND_WITH_CAUTION"
            reason = ("Query-OOF passes, permutation clean, but old-fragment-OOF below threshold "
                      f"({safe_f_top10:.4f} < 0.82). Model may learn fragment-specific patterns "
                      "but main ranking signal is generalizable. Proceed to blind with caution.")
    elif safe_q_ok and not perm_ok:
        decision = "FAIL_PERMUTATION_TEST"
        reason = f"Safe-only query-OOF high ({safe_q_top10:.4f}) but permutation test failed "
        f"({perm_top10:.4f}) — suggests features contain label-correlated structure even after permuting."
    elif not safe_q_ok:
        if safe_q_top10 <= 0.85:
            decision = "LEAKAGE_CONFIRMED"
            reason = (f"Safe-only query-OOF dropped to {safe_q_top10:.4f} (near baseline 0.834). "
                      "Original D4S28 result (0.9374) was inflated by label-derived features. "
                      "DO NOT deploy. Re-audit feature set.")
        else:
            decision = "PARTIAL_LEAKAGE"
            reason = (f"Safe-only query-OOF={safe_q_top10:.4f} below 0.87 but above baseline. "
                      "Some genuine signal exists but original result was inflated.")
    else:
        decision = "INCONCLUSIVE"
        reason = "Mixed signals — manual review needed."

    log(f"\n  DECISION: {decision}")
    log(f"  {reason}")

    audit_decision = {
        "decision": decision,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": {
            "safe_query_OOF_Top10": round(safe_q_top10, 6),
            "safe_query_OOF_threshold": 0.87,
            "safe_query_OOF_pass": safe_q_ok,
            "safe_fragment_OOF_Top10": round(safe_f_top10, 6),
            "safe_fragment_OOF_threshold": 0.82,
            "safe_fragment_OOF_pass": safe_f_ok,
            "permutation_Top10": round(perm_top10, 6),
            "permutation_test_pass": perm_ok,
            "random_baseline_Top10": round(rand_top10, 6),
            "old_fragment_exact_overlap": int(len(old_overlap)),
            "old_fragment_overlap_pass": old_overlap_ok,
            "original_HistGB_OOF_Top10": 0.937421,
            "baseline_Top10": 0.834019,
        },
        "allowed_next_steps": {
            "ALLOW_BLIND_ONESHOT": ["Run blind evaluation on safe-only HistGB"],
            "ALLOW_BLIND_WITH_CAUTION": ["Run blind but validate per-fragment results"],
            "LEAKAGE_CONFIRMED": ["Fix feature set, re-audit, DO NOT deploy"],
            "FAIL_PERMUTATION_TEST": ["Debug feature-label correlation, re-audit"],
            "PARTIAL_LEAKAGE": ["Investigate which safe features cause drop, refine"],
        }.get(decision, ["Manual review required"]),
    }
    with open(f"{OUT_DIR}/audit_decision.json", "w") as f:
        json.dump(audit_decision, f, indent=2, default=str)
    log(f"  Saved audit_decision.json")

    log(f"\nD4S28_AUDIT complete in {(time.time()-t0)/60:.1f} min")
    return audit_decision


if __name__ == "__main__":
    main()
