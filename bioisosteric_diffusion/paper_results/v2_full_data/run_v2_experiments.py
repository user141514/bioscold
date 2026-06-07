#!/usr/bin/env python3
"""V2 full-data experiments — OPTIMIZED.

Key changes vs run_v2_experiments.py:
- C3F uses fold-local OF×candidate success matrix (not global freq proxy)
- OF-OF Morgan similarity precomputed once
- CA/C3F scoring fully vectorized
- Static features + labels precomputed once, reused across all seeds/folds

Outputs (all in v2_full_data/):
  e1_main_10seed_summary.csv
  e1_main_10seed_detail.csv
  e1_main_paired_tests.csv
  e2_ablation_full_summary.csv
  e2_ablation_full_detail.csv
  e3_tuning_ledger.csv
  e4_coldstart_audit.csv
  e5_rank_diag.csv
  e6_hgb_vs_lr.csv
"""

import json, glob, time, itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

SEED = 20260603
FEATURE_NAMES = ["morgan", "bit_corr", "dHeavy", "dRings", "dMW", "dLogP", "dTPSA", "freq"]
OUT = Path("paper_results/v2_full_data")

# Grids (locked)
CA_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
C3F_K = [3, 5, 7, 10]
C3F_ALPHA = [0.1, 0.2, 0.3, 0.5, 0.7]
HGB_DEPTH = [3, 5]
HGB_ITER = [100]
HGB_LR = [0.05]
# Reduced HGB grid: 2 configs (capacity comparator only)
TIE_EPS = 0.005

RDLogger.DisableLog("rdApp.*")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ═══════════════════════════════════════════════════════
# Phase 1: Static precomputation
# ═══════════════════════════════════════════════════════

def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), fp)
    return fp

def mol_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return np.array([mol.GetNumHeavyAtoms(), Descriptors.RingCount(mol),
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)], dtype=np.float32)

def load_all():
    """Load manifest, labels, compute all static features. Returns a state dict."""
    t0 = time.time()
    log("Loading manifest...")
    manifest = {}
    mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
    with mp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line); manifest[q["query_id"]] = q
    log(f"  {len(manifest)} queries, {len(set(q['old_fragment_smiles'] for q in manifest.values()))} OFs")

    # OF fingerprints
    all_ofs = sorted({q["old_fragment_smiles"] for q in manifest.values()})
    of_fps, of_props = {}, {}
    for s in all_ofs:
        fp = morgan_fp(s); props = mol_props(s)
        if fp is not None and props is not None: of_fps[s] = fp; of_props[s] = props
    valid_ofs = [s for s in all_ofs if s in of_fps]
    log(f"  {len(valid_ofs)} valid OFs")

    # OF-OF Morgan similarity matrix (static, used by C3F)
    log("Precomputing OF-OF similarity...")
    n_of = len(valid_ofs)
    of_fp_mat = np.array([of_fps[s] for s in valid_ofs], dtype=np.float32)
    of_inter = np.dot(of_fp_mat, of_fp_mat.T)
    of_sums = of_fp_mat.sum(axis=1)
    of_denom = of_sums[:, None] + of_sums[None, :] - of_inter
    of_denom[of_denom == 0] = 1.0
    of_sim_mat = of_inter / of_denom  # (n_of, n_of)
    of_to_idx = {s: i for i, s in enumerate(valid_ofs)}
    log(f"  OF sim matrix: {of_sim_mat.shape}")

    # OF→queries mapping
    of_queries = defaultdict(list)
    for q in manifest.values():
        if q["old_fragment_smiles"] in of_fps:
            of_queries[q["old_fragment_smiles"]].append(q["query_id"])

    # Load ALL shard labels
    log("Loading labels from ALL shards...")
    query_labels = defaultdict(dict)
    for split in ["train", "test"]:
        pattern = (
            "plan_results/routeA_chembl37k_d0d3_engineering_safe/"
            f"07_d4a0_matrix_freeze/matrices/{split}/{split}_features_shard_0*.jsonl"
        )
        shards = sorted(glob.glob(pattern))
        log(f"  {split}: {len(shards)} shards")
        for si, shard in enumerate(shards):
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    r = json.loads(line)
                    if r["query_id"] in manifest:
                        query_labels[r["query_id"]][r["candidate"]] = int(r["label"])
            if (si + 1) % 50 == 0: log(f"    {split} shard {si+1}/{len(shards)}")
    log(f"  {len(query_labels)} queries with labels")

    # Candidate fingerprints
    log("Computing candidate fingerprints...")
    cand_fps, cand_props = {}, {}
    for labels in query_labels.values():
        for c in labels:
            if c in cand_fps: continue
            fp = morgan_fp(c); props = mol_props(c)
            if fp is not None and props is not None: cand_fps[c] = fp; cand_props[c] = props
    cand_list = sorted(cand_fps.keys())
    cand_to_idx = {c: i for i, c in enumerate(cand_list)}
    log(f"  {len(cand_list)} unique candidates")

    # Identify labeled OFs
    labeled_ofs = sorted({
        manifest[qid]["old_fragment_smiles"]
        for qid, labels in query_labels.items()
        if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())
    })
    log(f"  {len(labeled_ofs)} labeled OFs")

    # ── Precompute static features ──
    log("Precomputing per-OF features...")
    # of_data[of_s] = {"X": np.array (n_rows, 7), "y": np.array, "queries": [(qid, n_cands, cand_indices)]}
    # We store features WITHOUT freq (7 cols) and fill freq per fold
    of_data = {}
    total_pairs = 0
    for of_s in labeled_ofs:
        ofp = of_fps[of_s]; op = of_props[of_s]
        ofp_norm = ofp / (ofp.sum() + 1e-10)
        X_list, y_list, query_info = [], [], []
        for qid in of_queries.get(of_s, []):
            labels = query_labels.get(qid, {})
            if not labels: continue
            candidates = list(labels.keys())
            n_c = len(candidates)
            feats = np.zeros((n_c, 7), dtype=np.float32)  # 7 static features, freq is col 7
            cand_idx_list = []
            for idx, c in enumerate(candidates):
                cfp = cand_fps.get(c); cp = cand_props.get(c)
                if cfp is None or cp is None: continue
                inter = float(cfp.dot(ofp))
                denom = float(cfp.sum() + ofp.sum() - inter)
                morgan = inter / max(denom, 0.001)
                cfp_n = cfp / (cfp.sum() + 1e-10)
                bit_corr = 0.0
                if cfp.sum() > 0 and ofp.sum() > 0:
                    corr = float(np.corrcoef(cfp_n, ofp_norm)[0, 1])
                    bit_corr = corr if np.isfinite(corr) else 0.0
                feats[idx] = [morgan, bit_corr,
                    abs(cp[0] - op[0]), abs(cp[1] - op[1]),
                    abs(cp[2] - op[2]) / max(op[2], 1),
                    abs(cp[3] - op[3]) / max(abs(op[3]) + 1, 1),
                    abs(cp[4] - op[4]) / max(op[4] + 1, 1)]
                cand_idx_list.append(cand_to_idx.get(c, -1))
            X_list.append(feats)
            y_list.append(np.array([labels[c] for c in candidates], dtype=np.int8))
            query_info.append((qid, n_c, cand_idx_list))
        if X_list:
            of_data[of_s] = {"X7": np.vstack(X_list), "y": np.concatenate(y_list),
                             "queries": query_info, "n_rows": sum(len(x) for x in X_list)}
            total_pairs += of_data[of_s]["n_rows"]
    log(f"  {total_pairs:,} total pairs")

    # ── Precompute CA components (Morgan sim and physchem per candidate-OF pair) ──
    log("Precomputing CA components...")
    # For CA: score = λ × morgan + (1-λ) × physchem
    # morgan is already in X7[:,0]. physchem = 1/(1 + sum(deltas))
    # Precompute physchem score per row per OF
    of_pc = {}
    for of_s in labeled_ofs:
        X7 = of_data[of_s]["X7"]
        # physchem deltas are in cols 2-6 (dHeavy, dRings, dMW, dLogP, dTPSA)
        # But dMW, dLogP, dTPSA are normalized. For physchem we want raw-like penalty.
        # Use the same formula as content_score: 1/(1 + d_heavy + d_rings + d_mw + d_logp)
        # d_mw and d_logp are already normalized; that's OK per original implementation
        d_sum = X7[:, 2] + X7[:, 3] + X7[:, 4] + X7[:, 5]  # dHeavy + dRings + dMW_norm + dLogP_norm
        pc = 1.0 / (1.0 + d_sum)
        of_pc[of_s] = pc

    log(f"Static precomputation done in {(time.time()-t0)/60:.1f} min")
    return {
        "manifest": manifest, "of_fps": of_fps, "of_props": of_props,
        "valid_ofs": valid_ofs, "of_to_idx": of_to_idx, "of_sim_mat": of_sim_mat,
        "of_queries": of_queries, "query_labels": query_labels,
        "cand_fps": cand_fps, "cand_props": cand_props,
        "cand_list": cand_list, "cand_to_idx": cand_to_idx,
        "labeled_ofs": labeled_ofs, "of_data": of_data, "of_pc": of_pc,
    }

# ═══════════════════════════════════════════════════════
# Phase 2: Per-seed evaluation
# ═══════════════════════════════════════════════════════

def hit10_vec(labels_arr, scores):
    """Vectorized Hit@10 for one query: labels_arr is binary, scores is float array."""
    order = np.argsort(-scores)
    ranked = labels_arr[order]
    pos = np.where(ranked == 1)[0]
    return int(len(pos) > 0 and pos[0] < 10)

def build_freq_from_ofs(train_ofs, st):
    """Build candidate frequency map from train OFs."""
    freq = {}
    for of_s in train_ofs:
        y = st["of_data"][of_s]["y"]
        queries = st["of_data"][of_s]["queries"]
        offset = 0
        for _, n_c, cand_idx_list in queries:
            for i in range(n_c):
                if y[offset + i] == 1:
                    c = st["cand_list"][cand_idx_list[i]]
                    freq[c] = freq.get(c, 0) + 1
            offset += n_c
    return freq

def build_c3f_M(train_ofs, st):
    """Build C3F support matrix: M[of_idx, cand_idx] = normalized positive count.
    Returns (n_train_ofs, n_candidates) float32 array + list of train OF indices."""
    n_c = len(st["cand_list"])
    train_idx = [st["of_to_idx"][s] for s in train_ofs]
    M = np.zeros((len(train_ofs), n_c), dtype=np.float32)
    for i, of_s in enumerate(train_ofs):
        y = st["of_data"][of_s]["y"]
        queries = st["of_data"][of_s]["queries"]
        offset = 0
        for _, n_cand, cand_idx_list in queries:
            for j in range(n_cand):
                if y[offset + j] == 1:
                    ci = cand_idx_list[j]
                    if ci >= 0: M[i, ci] += 1
            offset += n_cand
        # Normalize per OF
        row_sum = M[i].sum()
        if row_sum > 0: M[i] /= row_sum
    return M, train_idx

def c3f_score_vec(test_of_s, candidates, cand_indices, M, train_idx, freq, max_freq, st, K, alpha):
    """Vectorized C3F scoring for one query."""
    test_idx = st["of_to_idx"][test_of_s]
    # Get similarities from test OF to all train OFs
    sims = st["of_sim_mat"][test_idx, train_idx]  # (n_train_ofs,)
    # Top-K
    if K >= len(train_idx):
        top_k_idx = np.arange(len(train_idx))
    else:
        top_k_idx = np.argpartition(-sims, K)[:K]
    top_sims = sims[top_k_idx]
    ws = top_sims.sum()

    n_c = len(candidates)
    # Build sub-matrix for these candidates
    valid_mask = np.array([ci >= 0 for ci in cand_indices])
    collab = np.zeros(n_c, dtype=np.float32)
    if ws > 0.001 and valid_mask.any():
        M_sub = M[top_k_idx][:, [ci for ci in cand_indices if ci >= 0]]  # (K, n_valid)
        weighted = (top_sims[:, None] * M_sub).sum(axis=0)  # (n_valid,)
        valid_indices = np.where(valid_mask)[0]
        collab[valid_indices] = weighted

    freq_sub = np.array([freq.get(c, 0) / max(max_freq, 1) for c in candidates], dtype=np.float32)
    return (collab + alpha * freq_sub) / (ws + alpha)

def ca_score_vec(test_of_s, candidates, cand_indices, st, lam, of_row_offset):
    """Vectorized CA scoring for one query."""
    morgan = np.array([st["of_data"][test_of_s]["X7"][of_row_offset + i, 0] for i in range(len(candidates))], dtype=np.float32)
    pc = np.array([st["of_pc"][test_of_s][of_row_offset + i] for i in range(len(candidates))], dtype=np.float32)
    return lam * morgan + (1 - lam) * pc

def evaluate_one_of_h10(of_s, candidates, cand_indices, labels_dict, scores):
    """Compute Hit@10 for one query. labels_dict: {candidate: label}."""
    labels_arr = np.array([labels_dict[c] for c in candidates], dtype=np.int8)
    if labels_arr.sum() == 0: return False, 0
    return True, hit10_vec(labels_arr, scores)

def run_experiments(st):
    t0 = time.time()
    labeled_ofs = st["labeled_ofs"]

    # Cold-start audit
    log("Cold-start audit...")
    audit_rows = []
    for of_s in labeled_ofs:
        pos = int(st["of_data"][of_s]["y"].sum())
        audit_rows.append({"of": of_s, "positive_pairs": pos, "total_pairs": st["of_data"][of_s]["n_rows"]})
    pd.DataFrame(audit_rows).to_csv(OUT / "e4_coldstart_audit.csv", index=False)

    log("=" * 60)
    log("10-SEED EXPERIMENT (OPTIMIZED)")
    log("=" * 60)

    summary_rows, detail_rows, ablation_rows, tuning_rows, hgb_rows = [], [], [], [], []

    for seed_idx in range(10):
        seed_t0 = time.time()
        log(f"\n--- Seed {seed_idx} ---")

        # Split OFs
        rng = np.random.RandomState(SEED + seed_idx)
        shuffled = list(labeled_ofs)
        rng.shuffle(shuffled)
        n_train = int(len(shuffled) * 0.7)
        train_ofs = shuffled[:n_train]
        test_ofs = shuffled[n_train:]
        log(f"  Split: {len(train_ofs)} train / {len(test_ofs)} test")

        # Train-only frequency
        freq = build_freq_from_ofs(train_ofs, st)
        max_freq = max(freq.values()) if freq else 1

        # ── Inner 3-fold CV ──
        log("  Inner 3-fold CV...")
        inner_ofs_arr = np.array(train_ofs)
        inner_groups = np.arange(len(train_ofs))
        gkf = GroupKFold(n_splits=3)
        inner_results = []

        for fold_idx, (it_idx, iv_idx) in enumerate(gkf.split(inner_ofs_arr, groups=inner_groups)):
            ft0 = time.time()
            inner_train_ofs = list(inner_ofs_arr[it_idx])
            inner_val_ofs = list(inner_ofs_arr[iv_idx])

            # Fold-local C3F M matrix
            M, train_ofs_idx = build_c3f_M(inner_train_ofs, st)

            # Inner train freq
            inner_freq = build_freq_from_ofs(inner_train_ofs, st)
            inner_max_freq = max(inner_freq.values()) if inner_freq else 1

            # Build training matrix
            X_it = np.vstack([st["of_data"][o]["X7"] for o in inner_train_ofs])
            y_it = np.concatenate([st["of_data"][o]["y"] for o in inner_train_ofs])
            X_it_full = np.zeros((len(X_it), 8), dtype=np.float32)
            X_it_full[:, :7] = X_it
            offset = 0
            for o in inner_train_ofs:
                queries = st["of_data"][o]["queries"]
                for _, n_c, cand_idx_list in queries:
                    for j in range(n_c):
                        ci = cand_idx_list[j]
                        c = st["cand_list"][ci] if ci >= 0 else ""
                        X_it_full[offset + j, 7] = inner_freq.get(c, 0) / max(inner_max_freq, 1)
                    offset += n_c
            scaler = StandardScaler()
            X_it_s = scaler.fit_transform(X_it_full)

            # Train LR (once per fold)
            lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            lr.fit(X_it_s, y_it)

            # Train all HGB configs (once per config per fold, OUTSIDE the val loop)
            hgb_models = {}
            for depth in HGB_DEPTH:
                for n_iter in HGB_ITER:
                    for lr_h in HGB_LR:
                        cfg = f"HGB_d={depth}_i={n_iter}_lr={lr_h}"
                        hgb = HistGradientBoostingClassifier(max_depth=depth, max_iter=n_iter, learning_rate=lr_h, random_state=SEED)
                        hgb.fit(X_it_s, y_it)
                        hgb_models[cfg] = hgb

            # Evaluate all configs on inner val
            for of_s in inner_val_ofs:
                queries = st["of_data"][of_s]["queries"]
                y = st["of_data"][of_s]["y"]
                X7 = st["of_data"][of_s]["X7"]
                offset = 0

                for _, n_c, cand_idx_list in queries:
                    candidates = [st["cand_list"][ci] if ci >= 0 else "" for ci in cand_idx_list]
                    labels_dict = dict(zip(candidates, y[offset:offset+n_c].astype(int)))
                    if sum(labels_dict.values()) == 0:
                        offset += n_c; continue

                    # Pre-build query features once
                    X_q = np.zeros((n_c, 8), dtype=np.float32)
                    X_q[:, :7] = X7[offset:offset+n_c]
                    X_q[:, 7] = [inner_freq.get(c, 0) / max(inner_max_freq, 1) for c in candidates]
                    X_q_s = scaler.transform(X_q)

                    # LR
                    lr_scores = lr.predict_proba(X_q_s)[:, 1]
                    has_pos, h10_val = evaluate_one_of_h10(of_s, candidates, cand_idx_list, labels_dict, lr_scores)
                    if has_pos: inner_results.append({"config": "LBC-LR", "fold": fold_idx, "h10": h10_val, "complexity": 0})

                    # CA grid
                    for lam in CA_LAMBDAS:
                        ca_scores = ca_score_vec(of_s, candidates, cand_idx_list, st, lam, offset)
                        has_pos, h10_val = evaluate_one_of_h10(of_s, candidates, cand_idx_list, labels_dict, ca_scores)
                        if has_pos: inner_results.append({"config": f"CA_λ={lam}", "fold": fold_idx, "h10": h10_val, "complexity": 1})

                    # C3F grid
                    for k_val in C3F_K:
                        for alpha in C3F_ALPHA:
                            c3f_scores = c3f_score_vec(of_s, candidates, cand_idx_list, M, train_ofs_idx, inner_freq, inner_max_freq, st, k_val, alpha)
                            has_pos, h10_val = evaluate_one_of_h10(of_s, candidates, cand_idx_list, labels_dict, c3f_scores)
                            if has_pos: inner_results.append({"config": f"C3F_K={k_val}_α={alpha}", "fold": fold_idx, "h10": h10_val, "complexity": 2})

                    # HGB grid (pre-trained models, just predict)
                    for cfg, hgb in hgb_models.items():
                        hgb_scores = hgb.predict_proba(X_q_s)[:, 1]
                        has_pos, h10_val = evaluate_one_of_h10(of_s, candidates, cand_idx_list, labels_dict, hgb_scores)
                        if has_pos: inner_results.append({"config": cfg, "fold": fold_idx, "h10": h10_val, "complexity": 3})
                    offset += n_c
            log(f"    Fold {fold_idx} done in {(time.time()-ft0)/60:.1f} min")

        # ── Select best config per method ──
        inner_df = pd.DataFrame(inner_results)
        config_mean = inner_df.groupby(["config", "complexity"])["h10"].mean().reset_index()

        def select_best(df, prefix):
            subset = df[df["config"].str.startswith(prefix)].copy()
            if subset.empty: return None, 0.0
            idx_max = subset["h10"].idxmax()
            best_val = subset.loc[idx_max, "h10"]
            tied = subset[subset["h10"] >= best_val - TIE_EPS]
            if len(tied) > 1:
                if prefix.startswith("CA_"):
                    tied["_lam"] = tied["config"].str.extract(r"λ=([\d.]+)")[0].astype(float)
                    # prefer middle λ (closer to 0.5)
                    tied = tied.sort_values("_lam", key=lambda x: abs(x - 0.5))
                elif prefix.startswith("C3F"):
                    tied["_K"] = tied["config"].str.extract(r"K=(\d+)")[0].astype(int)
                    tied["_alpha"] = tied["config"].str.extract(r"α=([\d.]+)")[0].astype(float)
                    tied = tied.sort_values(["_alpha", "_K"])
                elif prefix.startswith("HGB"):
                    tied["_d"] = tied["config"].str.extract(r"d=(\d+)")[0].astype(int)
                    tied["_i"] = tied["config"].str.extract(r"i=(\d+)")[0].astype(int)
                    tied["_lr"] = tied["config"].str.extract(r"lr=([\d.]+)")[0].astype(float)
                    tied = tied.sort_values(["_lr", "_i", "_d"])
                idx_max = tied.index[0]
            return subset.loc[idx_max, "config"], subset.loc[idx_max, "h10"]

        best_ca, ca_best = select_best(config_mean, "CA_")
        best_c3f, c3f_best = select_best(config_mean, "C3F_")
        best_hgb_cfg, hgb_best = select_best(config_mean, "HGB_")

        for _, row in config_mean.iterrows():
            tuning_rows.append({"seed": seed_idx, "config": row["config"], "mean_inner_h10": row["h10"]})

        log(f"  Best CA: {best_ca} ({ca_best:.4f})")
        log(f"  Best C3F: {best_c3f} ({c3f_best:.4f})")
        log(f"  Best HGB: {best_hgb_cfg} ({hgb_best:.4f})")

        # Parse best params
        ca_lam = float(best_ca.split("λ=")[1]) if best_ca else 0.7
        c3f_k = int(best_c3f.split("K=")[1].split("_")[0]) if best_c3f else 5
        c3f_alpha = float(best_c3f.split("α=")[1]) if best_c3f else 0.3
        hgb_d = 3; hgb_i = 100; hgb_lr_val = 0.05
        if best_hgb_cfg:
            for p in best_hgb_cfg.split("_"):
                if p.startswith("d="): hgb_d = int(p[2:])
                if p.startswith("i="): hgb_i = int(p[2:])
                if p.startswith("lr="): hgb_lr_val = float(p[3:])

        # ── Retrain on full outer-train ──
        log("  Retraining on full outer-train...")
        M_full, train_idx_full = build_c3f_M(train_ofs, st)

        X_ot = np.vstack([st["of_data"][o]["X7"] for o in train_ofs])
        y_ot = np.concatenate([st["of_data"][o]["y"] for o in train_ofs])
        X_ot_full = np.zeros((len(X_ot), 8), dtype=np.float32)
        X_ot_full[:, :7] = X_ot
        offset = 0
        for o in train_ofs:
            queries = st["of_data"][o]["queries"]
            for _, n_c, cand_idx_list in queries:
                for j in range(n_c):
                    ci = cand_idx_list[j]
                    c = st["cand_list"][ci] if ci >= 0 else ""
                    X_ot_full[offset + j, 7] = freq.get(c, 0) / max(max_freq, 1)
                offset += n_c
        scaler_ot = StandardScaler()
        X_ot_s = scaler_ot.fit_transform(X_ot_full)

        lr_ot = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        lr_ot.fit(X_ot_s, y_ot)

        hgb_ot = HistGradientBoostingClassifier(max_depth=hgb_d, max_iter=hgb_i, learning_rate=hgb_lr_val, random_state=SEED)
        hgb_ot.fit(X_ot_s, y_ot)

        # ── Evaluate outer-test ──
        log("  Evaluating outer-test...")
        seed_detail = []
        seed_summary = defaultdict(float)
        n_test_q = 0

        for of_s in test_ofs:
            queries = st["of_data"][of_s]["queries"]
            y = st["of_data"][of_s]["y"]
            X7 = st["of_data"][of_s]["X7"]
            offset = 0
            of_counts = defaultdict(int)
            of_n_q = 0

            for _, n_c, cand_idx_list in queries:
                candidates = [st["cand_list"][ci] if ci >= 0 else "" for ci in cand_idx_list]
                labels_dict = dict(zip(candidates, y[offset:offset+n_c].astype(int)))
                if sum(labels_dict.values()) == 0:
                    offset += n_c; continue
                of_n_q += 1

                # Prepare features
                X_q = np.zeros((n_c, 8), dtype=np.float32)
                X_q[:, :7] = X7[offset:offset+n_c]
                X_q[:, 7] = [freq.get(c, 0) / max(max_freq, 1) for c in candidates]
                X_q_s = scaler_ot.transform(X_q)

                lr_sc = lr_ot.predict_proba(X_q_s)[:, 1]
                hgb_sc = hgb_ot.predict_proba(X_q_s)[:, 1]
                ca_sc = ca_score_vec(of_s, candidates, cand_idx_list, st, ca_lam, offset)
                freq_sc = np.array([freq.get(c, 0) / max(max_freq, 1) for c in candidates], dtype=np.float32)
                c3f_sc = c3f_score_vec(of_s, candidates, cand_idx_list, M_full, train_idx_full, freq, max_freq, st, c3f_k, c3f_alpha)

                of_counts["lbc"] += hit10_vec(np.array([labels_dict[c] for c in candidates], dtype=np.int8), lr_sc)
                of_counts["hgb"] += hit10_vec(np.array([labels_dict[c] for c in candidates], dtype=np.int8), hgb_sc)
                of_counts["ca"] += hit10_vec(np.array([labels_dict[c] for c in candidates], dtype=np.int8), ca_sc)
                of_counts["c3f"] += hit10_vec(np.array([labels_dict[c] for c in candidates], dtype=np.int8), c3f_sc)
                of_counts["freq"] += hit10_vec(np.array([labels_dict[c] for c in candidates], dtype=np.int8), freq_sc)
                offset += n_c

            if of_n_q > 0:
                row_d = {"seed": seed_idx, "of": of_s, "n_queries": of_n_q}
                for k in ["lbc", "hgb", "ca", "c3f", "freq"]:
                    v = of_counts[k] / of_n_q
                    row_d[k] = v; seed_summary[k] += v
                seed_summary["n_of"] += 1; n_test_q += of_n_q
                seed_detail.append(row_d)

        n_eval = int(seed_summary["n_of"])
        macro_summary = {
            "seed": seed_idx, "n_train_of": len(train_ofs), "n_test_of": n_eval, "n_queries": n_test_q,
            "lbc_macro": seed_summary["lbc"] / n_eval if n_eval else np.nan,
            "lbc_query": sum(d["lbc"] * d["n_queries"] for d in seed_detail) / max(n_test_q, 1) if n_test_q else np.nan,
            "hgb_macro": seed_summary["hgb"] / n_eval if n_eval else np.nan,
            "hgb_query": sum(d["hgb"] * d["n_queries"] for d in seed_detail) / max(n_test_q, 1) if n_test_q else np.nan,
            "ca_macro": seed_summary["ca"] / n_eval if n_eval else np.nan,
            "ca_query": sum(d["ca"] * d["n_queries"] for d in seed_detail) / max(n_test_q, 1) if n_test_q else np.nan,
            "c3f_macro": seed_summary["c3f"] / n_eval if n_eval else np.nan,
            "c3f_query": sum(d["c3f"] * d["n_queries"] for d in seed_detail) / max(n_test_q, 1) if n_test_q else np.nan,
            "freq_macro": seed_summary["freq"] / n_eval if n_eval else np.nan,
            "freq_query": sum(d["freq"] * d["n_queries"] for d in seed_detail) / max(n_test_q, 1) if n_test_q else np.nan,
        }

        baseline_macros = {"freq": macro_summary["freq_macro"], "ca": macro_summary["ca_macro"], "c3f": macro_summary["c3f_macro"]}
        best_base = max(baseline_macros, key=baseline_macros.get)
        macro_summary["best_non_lbc"] = best_base
        macro_summary["delta_macro"] = macro_summary["lbc_macro"] - baseline_macros[best_base]
        macro_summary["delta_query"] = macro_summary["lbc_query"] - macro_summary[f"{best_base}_query"]

        log(f"  LBC: {macro_summary['lbc_macro']:.4f} | HGB: {macro_summary['hgb_macro']:.4f} | CA: {macro_summary['ca_macro']:.4f} | C3F: {macro_summary['c3f_macro']:.4f} | Freq: {macro_summary['freq_macro']:.4f}")
        log(f"  Δ vs best ({best_base}): macro={macro_summary['delta_macro']:+.4f} query={macro_summary['delta_query']:+.4f}")

        summary_rows.append(macro_summary)
        detail_rows.extend(seed_detail)

        hgb_rows.append({"seed": seed_idx, "lr_macro": macro_summary["lbc_macro"], "hgb_macro": macro_summary["hgb_macro"],
                         "delta": macro_summary["hgb_macro"] - macro_summary["lbc_macro"]})

        # ── Ablation ──
        log("  Ablation...")
        full_macro = macro_summary["lbc_macro"]
        for drop_idx, drop_name in enumerate(FEATURE_NAMES):
            keep_cols = [i for i in range(8) if i != drop_idx]
            lr_drop = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
            lr_drop.fit(X_ot_s[:, keep_cols], y_ot)
            drop_h10_list = []
            for of_s in test_ofs:
                queries = st["of_data"][of_s]["queries"]
                y = st["of_data"][of_s]["y"]
                X7 = st["of_data"][of_s]["X7"]
                offset = 0
                for _, n_c, cand_idx_list in queries:
                    candidates = [st["cand_list"][ci] if ci >= 0 else "" for ci in cand_idx_list]
                    labels_dict = dict(zip(candidates, y[offset:offset+n_c].astype(int)))
                    if sum(labels_dict.values()) == 0:
                        offset += n_c; continue
                    X_q = np.zeros((n_c, 8), dtype=np.float32)
                    X_q[:, :7] = X7[offset:offset+n_c]
                    X_q[:, 7] = [freq.get(c, 0) / max(max_freq, 1) for c in candidates]
                    scores = lr_drop.predict_proba(scaler_ot.transform(X_q)[:, keep_cols])[:, 1]
                    has_pos, h10_v = evaluate_one_of_h10(of_s, candidates, cand_idx_list, labels_dict, scores)
                    if has_pos: drop_h10_list.append(h10_v)
                    offset += n_c
            drop_macro = float(np.mean(drop_h10_list)) if drop_h10_list else 0.0
            ablation_rows.append({"seed": seed_idx, "drop": drop_name,
                "full_h10_macro": full_macro, "drop_h10_macro": drop_macro, "delta_macro": drop_macro - full_macro})

        log(f"  Seed {seed_idx} done in {(time.time()-seed_t0)/60:.1f} min")

        # Incremental save (protect against crashes)
        pd.DataFrame(summary_rows).to_csv(OUT / "e1_main_10seed_summary.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(OUT / "e1_main_10seed_detail.csv", index=False)
        pd.DataFrame(ablation_rows).to_csv(OUT / "e2_ablation_full_detail.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(OUT / "e3_tuning_ledger.csv", index=False)
        pd.DataFrame(hgb_rows).to_csv(OUT / "e6_hgb_vs_lr.csv", index=False)

    # ═══════════════════════════════════════
    # Save outputs
    # ═══════════════════════════════════════
    log("\nSaving outputs...")
    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    tuning_df = pd.DataFrame(tuning_rows)
    hgb_df = pd.DataFrame(hgb_rows)

    summary_df.to_csv(OUT / "e1_main_10seed_summary.csv", index=False)
    detail_df.to_csv(OUT / "e1_main_10seed_detail.csv", index=False)
    ablation_df.to_csv(OUT / "e2_ablation_full_detail.csv", index=False)
    tuning_df.to_csv(OUT / "e3_tuning_ledger.csv", index=False)
    hgb_df.to_csv(OUT / "e6_hgb_vs_lr.csv", index=False)

    # Paired tests
    tests = []
    for col, label in [("ca_macro", "LBC_vs_CA"), ("c3f_macro", "LBC_vs_C3F"), ("freq_macro", "LBC_vs_Freq")]:
        if col not in summary_df.columns: continue
        diffs = (summary_df["lbc_macro"] - summary_df[col]).to_numpy(dtype=float)
        obs = float(np.mean(diffs))
        signed = []
        for signs in itertools.product([-1.0, 1.0], repeat=len(diffs)):
            signed.append(float(np.mean(diffs * np.array(signs))))
        signed = np.array(signed)
        tests.append({"comparison": label, "n": len(diffs), "mean_diff": obs,
            "all_diffs_positive": bool(np.all(diffs > 0)),
            "one_sided_p": float(np.mean(signed >= obs)),
            "two_sided_p": float(np.mean(np.abs(signed) >= abs(obs)))})
    pd.DataFrame(tests).to_csv(OUT / "e1_main_paired_tests.csv", index=False)

    # Ablation summary
    if not ablation_df.empty:
        abl_sum = ablation_df.groupby("drop")["delta_macro"].agg(["mean", "std"]).reset_index()
        abl_sum.columns = ["drop", "mean_delta_macro", "std_delta_macro"]
        abl_sum = abl_sum.sort_values("mean_delta_macro")
        abl_sum.to_csv(OUT / "e2_ablation_full_summary.csv", index=False)

    # Rank diagnostic
    rank_rows = []
    for d in detail_rows:
        rank_rows.append({"seed": d["seed"], "of": d["of"], "n_queries": d["n_queries"],
            "lbc_h10": d["lbc"], "ca_h10": d["ca"], "c3f_h10": d.get("c3f", np.nan), "freq_h10": d.get("freq", np.nan)})
    pd.DataFrame(rank_rows).to_csv(OUT / "e5_rank_diag.csv", index=False)

    # Final report
    log("\n" + "=" * 60)
    log("FINAL RESULTS")
    log("=" * 60)
    lbc_m = summary_df["lbc_macro"].mean(); lbc_s = summary_df["lbc_macro"].std()
    delta_m = summary_df["delta_macro"].mean(); delta_s = summary_df["delta_macro"].std()
    delta_q = summary_df["delta_query"].mean()
    wins = int((summary_df["delta_macro"] > 0).sum())

    log(f"LBC-Ranker:  {lbc_m:.4f} ± {lbc_s:.4f}")
    log(f"HGB:         {summary_df['hgb_macro'].mean():.4f}")
    log(f"CA (tuned):  {summary_df['ca_macro'].mean():.4f}")
    log(f"C3F (tuned): {summary_df['c3f_macro'].mean():.4f}")
    log(f"Frequency:   {summary_df['freq_macro'].mean():.4f}")
    log(f"Δ LBC vs best non-LBC: {delta_m:+.4f} ± {delta_s:.4f}")
    log(f"Δ query-weighted: {delta_q:+.4f}")
    log(f"LBC wins: {wins}/10 seeds")
    log(f"HGB vs LR delta: {hgb_df['delta'].mean():+.4f}")

    if not ablation_df.empty:
        abl_s = ablation_df.groupby("drop")["delta_macro"].mean().sort_values()
        log(f"Ablation (top 3 losses):")
        for drop_name in abl_s.index[:3]:
            log(f"  Drop {drop_name}: {abl_s[drop_name]:+.4f}")

    tier = "WEAK"
    if delta_m >= 0.15 and delta_q >= 0.05 and wins >= 9: tier = "STRONG"
    elif delta_m >= 0.10 and delta_q >= 0: tier = "MEDIUM"
    log(f"\nClaim tier: {tier}")
    log(f"Total time: {(time.time()-t0)/60:.1f} min")
    return tier


if __name__ == "__main__":
    st = load_all()
    run_experiments(st)
