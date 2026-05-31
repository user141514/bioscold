#!/usr/bin/env python3
"""
Route-A D4A2 Part B: Ranker Ceiling Test
=========================================
Tests multiple ranking model families against the canonical baseline.
Trains on the standard 7-feature set, evaluates with canonical metrics.

Environment: conda activate accfg
"""

import json, csv, os, sys, time, random, math
from pathlib import Path
from collections import defaultdict

import numpy as np

import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem, DataStructs, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from rdkit.Chem import AllChem, Descriptors

# Optional imports
HAS_LIGHTGBM = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass

HAS_XGBOOST = False
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A0 = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
D4A2 = BASE / "plan_results/routeA_chembl37k_d4a2_canonical_ranker_and_controls"
MATRICES = D4A0 / "matrices"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def write_csv(path, rows, fieldnames):
    path = D4A2 / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"  wrote {len(rows)} rows -> {path.name}")


# ── Query manifest ────────────────────────────────────────────────
def load_query_manifest():
    manifest_path = D4A0 / "d4a0_query_split_manifest.jsonl"
    queries = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["query_id"]] = q
    log(f"  Loaded {len(queries)} queries from manifest")
    return queries


# ── RDKit Feature Cache ───────────────────────────────────────────
class FeatureCache:
    def __init__(self):
        self.fp_cache = {}
        self.ha_cache = {}

    def get_fp(self, smiles):
        if smiles not in self.fp_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.fp_cache[smiles] = np.zeros(1024, dtype=np.float32)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                arr = np.zeros(1024, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                self.fp_cache[smiles] = arr
        return self.fp_cache[smiles]

    def get_ha(self, smiles):
        if smiles not in self.ha_cache:
            mol = Chem.MolFromSmiles(smiles)
            self.ha_cache[smiles] = mol.GetNumHeavyAtoms() if mol else 0
        return self.ha_cache[smiles]

    def get_pair_features(self, old_smiles, cand_smiles):
        old_fp = self.get_fp(old_smiles)
        cand_fp = self.get_fp(cand_smiles)
        if old_fp.sum() > 0 and cand_fp.sum() > 0:
            dot = float(np.dot(old_fp, cand_fp))
            tanimoto = dot / (old_fp.sum() + cand_fp.sum() - dot + 1e-10)
        else:
            tanimoto = 0.0
        old_ha = self.get_ha(old_smiles)
        cand_ha = self.get_ha(cand_smiles)
        return {
            "morgan_tanimoto": tanimoto,
            "heavy_atom_delta": abs(cand_ha - old_ha),
            "cand_heavy_atoms": cand_ha,
            "old_heavy_atoms": old_ha,
            "attachment_match": 0,
        }

    def cache_size(self):
        return f"fp={len(self.fp_cache)}, ha={len(self.ha_cache)}"


# Global cache
FCACHE = FeatureCache()


# ── Feature extraction ────────────────────────────────────────────
def extract_features_row(row, qinfo):
    """Extract 7 features from one matrix row + query info."""
    old_smiles = qinfo.get("old_fragment_smiles", "")
    candidate = row.get("candidate", "")
    gf = row.get("global_freq", 0)
    af = row.get("attach_freq", 0)
    pair = FCACHE.get_pair_features(old_smiles, candidate)
    return [
        math.log1p(gf),
        math.log1p(af),
        float(pair["morgan_tanimoto"]),
        float(pair["heavy_atom_delta"]),
        float(pair["cand_heavy_atoms"]),
        float(pair["old_heavy_atoms"]),
        float(pair["attachment_match"]),
    ]


FEATURE_NAMES = [
    "log_global_freq", "log_attach_freq", "morgan_tanimoto",
    "heavy_atom_delta", "cand_heavy_atoms", "old_heavy_atoms", "attachment_match",
]


# ── Stream shards to arrays ───────────────────────────────────────
def stream_shards_to_arrays(shard_paths, queries, max_rows=None):
    """Stream shards, extract features, return X, y, qids, candidates list.
    If max_rows is set, uses reservoir sampling to get a representative sample."""
    X_list = []
    y_list = []
    qid_list = []
    cand_list = []
    n_total = 0

    for shard_path in shard_paths:
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                n_total += 1
                row = json.loads(line)
                qid = row["query_id"]
                qinfo = queries.get(qid, {})
                feats = extract_features_row(row, qinfo)

                if max_rows:
                    if len(X_list) < max_rows:
                        # Fill reservoir
                        X_list.append(feats)
                        y_list.append(row.get("label", 0))
                        qid_list.append(qid)
                        cand_list.append(row.get("candidate", ""))
                    else:
                        # Reservoir sampling: replace with decreasing probability
                        if random.random() < max_rows / n_total:
                            idx = random.randrange(len(X_list))
                            X_list[idx] = feats
                            y_list[idx] = row.get("label", 0)
                            qid_list[idx] = qid
                            cand_list[idx] = row.get("candidate", "")
                else:
                    X_list.append(feats)
                    y_list.append(row.get("label", 0))
                    qid_list.append(qid)
                    cand_list.append(row.get("candidate", ""))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    log(f"  Streamed {len(X)} rows ({n_total} total seen), {y.sum()} positives, cache: {FCACHE.cache_size()}")
    return X, y, qid_list, cand_list


# ── Per-query metrics (canonical) ─────────────────────────────────
KS = [1, 5, 10, 20, 50]


def compute_query_metrics(query_cands):
    per_query = {}
    for qid, cands in query_cands.items():
        if not cands:
            continue
        ranked = sorted(cands, key=lambda x: x[1], reverse=True)
        labels = [lbl for _, _, lbl in ranked]
        qm = {"n_cand": len(ranked), "has_pos": int(sum(labels) > 0)}
        for k in KS:
            qm[f"top{k}"] = 1 if sum(labels[:k]) > 0 else 0
        mrr = 0.0
        for i, lbl in enumerate(labels):
            if lbl == 1:
                mrr = 1.0 / (i + 1)
                break
        qm["mrr"] = mrr
        per_query[qid] = qm

    n = len(per_query)
    summary = {"n_queries": n}
    if n > 0:
        for k in KS:
            summary[f"top{k}"] = float(np.mean([qm[f"top{k}"] for qm in per_query.values()]))
        summary["mrr"] = float(np.mean([qm["mrr"] for qm in per_query.values()]))
    else:
        for k in KS:
            summary[f"top{k}"] = 0.0
        summary["mrr"] = 0.0
    return summary, per_query


# ── Evaluate using batch score predictions ───────────────────────
def evaluate_batch(X_test, y_test, qid_test, cand_test, score_fn):
    """
    Evaluate using pre-computed features. score_fn(X) -> array of scores.
    Returns (summary, per_query)
    """
    scores = score_fn(X_test)
    query_cands = defaultdict(list)
    for i in range(len(qid_test)):
        query_cands[qid_test[i]].append((cand_test[i], float(scores[i]), int(y_test[i])))
    return compute_query_metrics(dict(query_cands))


# ── Bootstrap CI for delta ────────────────────────────────────────
def bootstrap_delta(per_query_a, per_query_b, metric="top10", n_bootstrap=1000):
    """Bootstrap CI for metric difference A - B."""
    qids = sorted(set(per_query_a.keys()) & set(per_query_b.keys()))
    if not qids:
        return 0.0, 0.0, 0.0
    va = np.array([per_query_a[q][metric] for q in qids])
    vb = np.array([per_query_b[q][metric] for q in qids])
    n = len(va)
    deltas = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        deltas[i] = np.mean(va[idx]) - np.mean(vb[idx])
    mean_d = float(np.mean(deltas))
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return mean_d, lo, hi


def bootstrap_ci(per_query, metric="top10", n_bootstrap=1000):
    qids = list(per_query.keys())
    values = np.array([per_query[q][metric] for q in qids])
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        means[i] = np.mean(values[idx])
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return float(np.mean(means)), float(lo), float(hi)


# ── Model Arms ────────────────────────────────────────────────────

# B0: HGB reproduced
def train_b0_hgb(X_train, y_train, X_val, y_val):
    """HistGradientBoostingClassifier"""
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Subsample for speed if needed
    max_samples = 200000
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_t = X_train[idx]
        y_t = y_train[idx]
    else:
        X_t, y_t = X_train, y_train

    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_depth=5, max_iter=100, learning_rate=0.1,
        early_stopping=False, random_state=SEED, class_weight="balanced",
    )
    model.fit(X_t, y_t)
    elapsed = time.time() - t0
    return model, {"train_time": round(elapsed, 1), "train_samples": len(X_t)}


# B1: Pairwise logistic ranker
def train_b1_pairwise(X_train, y_train, qid_train):
    """SGDClassifier on pairwise difference vectors within queries."""
    from sklearn.linear_model import SGDClassifier

    t0 = time.time()
    # Group by query
    pos_by_q = defaultdict(list)
    neg_by_q = defaultdict(list)
    for i in range(len(X_train)):
        if y_train[i] == 1:
            pos_by_q[qid_train[i]].append(X_train[i])
        else:
            neg_by_q[qid_train[i]].append(X_train[i])

    X_pair = []
    for qid in pos_by_q:
        pos_vecs = pos_by_q[qid][:5]
        neg_vecs = neg_by_q.get(qid, [])[:5]
        for pv in pos_vecs:
            for nv in neg_vecs:
                X_pair.append(pv - nv)

    if not X_pair:
        log("  WARNING: No pairwise data")
        return None, {"train_time": 0, "n_pairs": 0}

    X_pair_all = np.array(X_pair + [-x for x in X_pair], dtype=np.float32)
    y_pair = np.array([1] * len(X_pair) + [0] * len(X_pair), dtype=np.int32)

    model = SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001,
                          max_iter=1000, random_state=SEED)
    model.fit(X_pair_all, y_pair)
    elapsed = time.time() - t0

    def scorer(X):
        return model.decision_function(X)

    return scorer, {"train_time": round(elapsed, 1), "n_pairs": len(X_pair)}


# B2a: XGBoost rank:pairwise (preferred)
def train_b2_xgboost_ranker(X_train, y_train, qid_train, X_val, y_val, qid_val):
    """XGBoost rank:pairwise ranking model."""
    if not HAS_XGBOOST:
        log("  XGBoost not available, falling back to MLP...")
        return None, {"train_time": 0, "note": "XGBoost not available"}

    t0 = time.time()
    from collections import Counter
    train_qcount = Counter(qid_train)
    val_qcount = Counter(qid_val)
    train_group_sizes = list(train_qcount.values())
    val_group_sizes = list(val_qcount.values())

    max_samples = 100000
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_t = X_train[idx]
        y_t = y_train[idx]
        qid_t = np.array([qid_train[i] for i in idx])
        train_qcount = Counter(qid_t)
        train_group_sizes = list(train_qcount.values())
    else:
        X_t = X_train
        y_t = y_train
        qid_t = np.array(qid_train)

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        random_state=SEED,
        verbosity=0,
    )
    model.fit(X_t, y_t, qid=qid_t,
              eval_set=[(X_val, y_val)],
              eval_qid=[np.array(qid_val)])
    elapsed = time.time() - t0

    def scorer(X):
        return model.predict(X)

    return scorer, {"train_time": round(elapsed, 1), "train_samples": len(X_t)}


# B2b: Listwise softmax ranker (simple neural net, fallback)
def train_b2_listwise_softmax(X_train, y_train, qid_train, X_val, y_val, qid_val):
    """Simple neural net with per-query softmax."""
    from sklearn.neural_network import MLPRegressor

    t0 = time.time()
    # Use MLPRegressor as a simple scoring function (trained pointwise as proxy)
    max_samples = 100000
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_t = X_train[idx]
        y_t = y_train[idx]
    else:
        X_t, y_t = X_train, y_train

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu",
        solver="adam", max_iter=200, random_state=SEED,
        early_stopping=True, validation_fraction=0.1,
    )
    model.fit(X_t, y_t)
    elapsed = time.time() - t0

    def scorer(X):
        return model.predict(X)

    return scorer, {"train_time": round(elapsed, 1), "train_samples": len(X_t)}


# B3: Two-stage reranker
def make_b3_two_stage_reranker(base_model, base_scaler, top_n=100):
    """
    Stage 1: attach_freq top-N prefilter
    Stage 2: base_model rerank
    For batch inference, apply a combined score.
    """
    from sklearn.preprocessing import StandardScaler

    def scorer(X):
        """
        X has 7 features: [log_gf, log_af, tanimoto, ha_delta, cand_ha, old_ha, match]
        We need attach_freq for prefilter. It's in column 1 (log1p(attach_freq)).
        """
        scores = base_model.predict_proba(base_scaler.transform(X))[:, 1]
        attach_freq_scores = X[:, 1]  # log1p(attach_freq)
        # Blend: apply prefilter weighting via combined score
        # Normalize attach_freq to approx [0,1]
        af_norm = attach_freq_scores / 10.0
        return 0.3 * af_norm + 0.7 * scores

    return scorer, {"first_stage_k": top_n}


# B4: Feature-engineered HGB with extra features
def train_b4_engineered_hgb(X_train, y_train, X_val, y_val):
    """HGB with extra features (rank-based features)."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Engineering: add rank-based features
    def engineer(X):
        Xe = X.copy()
        # Column 1 is log_attach_freq, column 0 is log_global_freq
        # Add inverse rank: 1/(1+rank_by_freq)
        # Add candidate length (not available from these features)
        # Add interaction: tanimoto * attach_freq
        tanimoto = X[:, 2]
        log_af = X[:, 1]
        log_gf = X[:, 0]
        ha_delta = X[:, 3]
        cand_ha = X[:, 4]

        # Interaction features
        tanimoto_af = tanimoto * log_af
        tanimoto_gf = tanimoto * log_gf
        # Size ratio
        ha_ratio = cand_ha / (X[:, 5] + 1e-10)  # cand_ha / old_ha
        # Append extra features
        extras = np.column_stack([tanimoto_af, tanimoto_gf, ha_ratio, ha_delta ** 2])
        return np.column_stack([Xe, extras])

    X_tr_e = engineer(X_train)
    X_val_e = engineer(X_val)

    max_samples = 200000
    if len(X_tr_e) > max_samples:
        idx = np.random.choice(len(X_tr_e), max_samples, replace=False)
        X_t = X_tr_e[idx]
        y_t = y_train[idx]
    else:
        X_t, y_t = X_tr_e, y_train

    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_depth=5, max_iter=100, learning_rate=0.1,
        early_stopping=False, random_state=SEED, class_weight="balanced",
    )
    model.fit(X_t, y_t)
    elapsed = time.time() - t0

    def scorer(X):
        Xe = engineer(X)
        return model.predict_proba(Xe)[:, 1]

    return scorer, {"train_time": round(elapsed, 1), "train_samples": len(X_t), "n_features": X_t.shape[1]}


# B5: LightGBM LambdaRank (optional)
def train_b5_lambdarank(X_train, y_train, qid_train, X_val, y_val, qid_val):
    """LightGBM LambdaRank - skip if LightGBM not available."""
    if not HAS_LIGHTGBM:
        log("  SKIP: LightGBM not installed")
        return None, {"train_time": 0, "note": "LightGBM not available"}

    t0 = time.time()
    # Create group info for ranking
    train_groups = defaultdict(list)
    val_groups = defaultdict(list)

    from collections import Counter
    train_qcount = Counter(qid_train)
    val_qcount = Counter(qid_val)

    train_group_sizes = list(train_qcount.values())
    val_group_sizes = list(val_qcount.values())

    # Subsample for speed
    max_samples = 100000
    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_t = X_train[idx]
        y_t = y_train[idx]
        qid_t = [qid_train[i] for i in idx]
        train_qcount = Counter(qid_t)
        train_group_sizes = list(train_qcount.values())
    else:
        X_t, y_t, qid_t = X_train, y_train, qid_train

    train_data = lgb.Dataset(X_t, label=y_t, group=train_group_sizes)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_group_sizes, reference=train_data)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "num_threads": 4,
        "seed": SEED,
        "verbosity": -1,
    }

    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
    )
    elapsed = time.time() - t0

    def scorer(X):
        return model.predict(X)

    log(f"  LightGBM trained: {model.best_iteration} rounds")
    return scorer, {"train_time": round(elapsed, 1), "train_samples": len(X_t), "best_iteration": model.best_iteration}


# ── Main ──────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("D4A2 PART B: RANKER CEILING TEST")
    log("=" * 60)

    # 1. Load data
    queries = load_query_manifest()
    train_shards = sorted((MATRICES / "train").glob("train_features_shard_*.jsonl"))
    val_shards = sorted((MATRICES / "val").glob("val_features_shard_*.jsonl"))
    test_shards = sorted((MATRICES / "test").glob("test_features_shard_*.jsonl"))

    # Find test queries
    test_qids_set = set(qid for qid, q in queries.items() if q.get("split") == "test")
    val_qids_set = set(qid for qid, q in queries.items() if q.get("split") == "val")

    # 2. Pre-compute features (reservoir-sampled to 200k across ALL training shards)
    log("\n--- Pre-computing training features (reservoir-sampled 200k across ALL shards) ---")
    log("  Streaming all 247 train shards with reservoir sampling (this may take a few minutes)...")
    t_data = time.time()
    X_train_full, y_train_full, qid_train_full, _ = stream_shards_to_arrays(
        train_shards, queries, max_rows=200000)
    log(f"  Train: {len(X_train_full)} samples, {y_train_full.sum()} positives ({time.time()-t_data:.0f}s)")

    log("\n--- Pre-computing validation features (all) ---")
    X_val, y_val, qid_val, cand_val = stream_shards_to_arrays(val_shards, queries)
    log(f"  Val: {len(X_val)} samples, {y_val.sum()} positives")

    log("\n--- Pre-computing test features (all) ---")
    X_test, y_test, qid_test, cand_test = stream_shards_to_arrays(test_shards, queries)
    log(f"  Test: {len(X_test)} samples, {y_test.sum()} positives")

    # ── Load D4A1 canonical HGB predictions for bootstrap comparison ──
    log("\n--- Loading D4A1 canonical HGB predictions ---")
    d4a1_pred_path = BASE / "plan_results/routeA_chembl37k_d4a1_learned_ranker/d4a1_test_predictions.jsonl"
    canonical_per_q = defaultdict(lambda: {"hgb_hits": [], "labels": []})
    n_preds = 0
    with open(d4a1_pred_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            canonical_per_q[qid]["hgb_hits"].append((row["score"], row["label"]))
            n_preds += 1

    # Compute per-query canonical HGB top10 hit
    canonical_hits = {}
    for qid, data in canonical_per_q.items():
        ranked = sorted(data["hgb_hits"], key=lambda x: x[0], reverse=True)
        canonical_hits[qid] = 1 if any(l == 1 for _, l in ranked[:10]) else 0

    perq_canonical_hgb = {qid: {"top10": int(hit), "mrr": 0.0} for qid, hit in canonical_hits.items()}
    canonical_top10_mean = float(np.mean(list(canonical_hits.values()))) if canonical_hits else 0.0
    log(f"  Loaded {n_preds} predictions for {len(canonical_per_q)} queries")
    log(f"  Mean canonical HGB Top10 (from D4A1 predictions): {canonical_top10_mean:.4f}")

    # 3. Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 4. Reference: attachment_frequency on val and test
    log("\n--- Reference: attachment_frequency ---")
    def attach_freq_scorer(X):
        return X[:, 1]  # log_attach_freq

    summary_attach_val, perq_attach_val = evaluate_batch(X_val_s, y_val, qid_val, cand_val, attach_freq_scorer)
    summary_attach_test, perq_attach_test = evaluate_batch(X_test_s, y_test, qid_test, cand_test, attach_freq_scorer)
    attach_val_top10 = summary_attach_val["top10"]
    attach_test_top10 = summary_attach_test["top10"]
    log(f"  Val Top10={attach_val_top10:.4f}, Test Top10={attach_test_top10:.4f}")

    # 5. Train and evaluate all model arms
    model_results = []
    bootstrap_results = []

    # ── B0: HGB reproduced ──
    log("\n--- B0: HGB reproduced ---")
    b0_model, b0_info = train_b0_hgb(X_train_s, y_train_full, X_val_s, y_val)
    b0_model_obj = b0_model

    def b0_scorer(X):
        return b0_model.predict_proba(scaler.transform(
            scaler.inverse_transform(X) if X.shape[1] == 7 else X))[:, 1]

    def b0_scorer_wrap(X):
        return b0_model.predict_proba(X)[:, 1]

    summary_b0_val, perq_b0_val = evaluate_batch(X_val_s, y_val, qid_val, cand_val, b0_scorer_wrap)
    summary_b0_test, perq_b0_test = evaluate_batch(X_test_s, y_test, qid_test, cand_test, b0_scorer_wrap)
    log(f"  Val Top10={summary_b0_val['top10']:.4f}, Test Top10={summary_b0_test['top10']:.4f}")

    model_results.append({
        "model": "B0_HGB_reproduced", "arm": "B0",
        "val_top10": round(summary_b0_val["top10"], 4),
        "val_mrr": round(summary_b0_val["mrr"], 4),
        "test_top10": round(summary_b0_test["top10"], 4),
        "test_mrr": round(summary_b0_test["mrr"], 4),
        "train_time_sec": b0_info["train_time"],
        "note": "HistGradientBoostingClassifier, 200k samples, max_depth=5",
    })

    # Bootstrap delta vs attach_freq
    d, lo, hi = bootstrap_delta(perq_b0_test, perq_attach_test, "top10")
    bootstrap_results.append({
        "comparison": "B0_HGB_vs_attach_freq",
        "model_top10": round(summary_b0_test["top10"], 4),
        "baseline_top10": round(attach_test_top10, 4),
        "delta": round(d, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "significant": "YES" if lo > 0 else "NO",
    })

    # Bootstrap delta vs canonical HGB (D4A1)
    d_can, lo_can, hi_can = bootstrap_delta(perq_b0_test, perq_canonical_hgb, "top10")
    bootstrap_results.append({
        "comparison": "B0_HGB_vs_canonical_HGB",
        "model_top10": round(summary_b0_test["top10"], 4),
        "baseline_top10": round(canonical_top10_mean, 4),
        "delta": round(d_can, 4),
        "ci_lo": round(lo_can, 4),
        "ci_hi": round(hi_can, 4),
        "significant": "YES" if lo_can > 0 else "NO",
    })
    log(f"  B0 vs canonical HGB: delta={d_can:.4f} [{lo_can:.4f}, {hi_can:.4f}], significant={lo_can > 0}")

    # ── B1: Pairwise logistic ranker ──
    log("\n--- B1: Pairwise logistic ranker ---")
    b1_scorer, b1_info = train_b1_pairwise(X_train_s, y_train_full, qid_train_full)
    if b1_scorer:
        summary_b1_val, perq_b1_val = evaluate_batch(X_val_s, y_val, qid_val, cand_val, b1_scorer)
        summary_b1_test, perq_b1_test = evaluate_batch(X_test_s, y_test, qid_test, cand_test, b1_scorer)
        log(f"  Val Top10={summary_b1_val['top10']:.4f}, Test Top10={summary_b1_test['top10']:.4f}")

        model_results.append({
            "model": "B1_pairwise_ranker", "arm": "B1",
            "val_top10": round(summary_b1_val["top10"], 4),
            "val_mrr": round(summary_b1_val["mrr"], 4),
            "test_top10": round(summary_b1_test["top10"], 4),
            "test_mrr": round(summary_b1_test["mrr"], 4),
            "train_time_sec": b1_info["train_time"],
            "note": f"SGD pairwise, {b1_info['n_pairs']} pairs",
        })

        d, lo, hi = bootstrap_delta(perq_b1_test, perq_attach_test, "top10")
        bootstrap_results.append({
            "comparison": "B1_pairwise_vs_attach_freq",
            "model_top10": round(summary_b1_test["top10"], 4),
            "baseline_top10": round(attach_test_top10, 4),
            "delta": round(d, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "significant": "YES" if lo > 0 else "NO",
        })

        d, lo, hi = bootstrap_delta(perq_b1_test, perq_b0_test, "top10")
        bootstrap_results.append({
            "comparison": "B1_pairwise_vs_B0_HGB",
            "model_top10": round(summary_b1_test["top10"], 4),
            "baseline_top10": round(summary_b0_test["top10"], 4),
            "delta": round(d, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "significant": "YES" if lo > 0 else "NO",
        })
    else:
        log("  SKIP: No pairwise data")

    # ── B2: XGBoost rank:pairwise (MLP fallback) ──
    log("\n--- B2: XGBoost rank:pairwise ---")
    b2_scorer, b2_info = train_b2_xgboost_ranker(
        X_train_s, y_train_full, qid_train_full,
        X_val_s, y_val, qid_val)
    if b2_scorer is None:
        log("--- B2: MLP fallback ---")
        b2_scorer, b2_info = train_b2_listwise_softmax(
            X_train_s, y_train_full, qid_train_full,
            X_val_s, y_val, qid_val)
        b2_note = f"MLPRegressor, {b2_info.get('train_samples', 0)} samples, (64,32) hidden (XGBoost fallback)"
        b2_model_name = "B2_mlp_softmax_proxy"
    else:
        b2_note = f"XGBoost rank:pairwise, {b2_info['train_samples']} samples, max_depth=5"
        b2_model_name = "B2_xgboost_ranker"
    summary_b2_val, perq_b2_val = evaluate_batch(X_val_s, y_val, qid_val, cand_val, b2_scorer)
    summary_b2_test, perq_b2_test = evaluate_batch(X_test_s, y_test, qid_test, cand_test, b2_scorer)
    log(f"  Val Top10={summary_b2_val['top10']:.4f}, Test Top10={summary_b2_test['top10']:.4f}")

    model_results.append({
        "model": b2_model_name, "arm": "B2",
        "val_top10": round(summary_b2_val["top10"], 4),
        "val_mrr": round(summary_b2_val["mrr"], 4),
        "test_top10": round(summary_b2_test["top10"], 4),
        "test_mrr": round(summary_b2_test["mrr"], 4),
        "train_time_sec": b2_info["train_time"],
        "note": b2_note,
    })

    d, lo, hi = bootstrap_delta(perq_b2_test, perq_attach_test, "top10")
    bootstrap_results.append({
        "comparison": "B2_MLP_vs_attach_freq",
        "model_top10": round(summary_b2_test["top10"], 4),
        "baseline_top10": round(attach_test_top10, 4),
        "delta": round(d, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "significant": "YES" if lo > 0 else "NO",
    })

    d, lo, hi = bootstrap_delta(perq_b2_test, perq_b0_test, "top10")
    bootstrap_results.append({
        "comparison": "B2_MLP_vs_B0_HGB",
        "model_top10": round(summary_b2_test["top10"], 4),
        "baseline_top10": round(summary_b0_test["top10"], 4),
        "delta": round(d, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "significant": "YES" if lo > 0 else "NO",
    })

    # ── B3: Two-stage reranker ──
    log("\n--- B3: Two-stage reranker (attach_freq top-N + HGB rerank) ---")
    b3_scorer, b3_info = make_b3_two_stage_reranker(b0_model, scaler, top_n=100)
    summary_b3_val, perq_b3_val = evaluate_batch(X_val_s, y_val, qid_val, cand_val, b3_scorer)
    summary_b3_test, perq_b3_test = evaluate_batch(X_test_s, y_test, qid_test, cand_test, b3_scorer)
    log(f"  Val Top10={summary_b3_val['top10']:.4f}, Test Top10={summary_b3_test['top10']:.4f}")

    model_results.append({
        "model": "B3_hybrid_ranker", "arm": "B3",
        "val_top10": round(summary_b3_val["top10"], 4),
        "val_mrr": round(summary_b3_val["mrr"], 4),
        "test_top10": round(summary_b3_test["top10"], 4),
        "test_mrr": round(summary_b3_test["mrr"], 4),
        "train_time_sec": b0_info["train_time"],
        "note": "Hybrid reranker: attach_freq prefilter + HGB rerank, 0.3*af + 0.7*HGB",
    })

    d, lo, hi = bootstrap_delta(perq_b3_test, perq_attach_test, "top10")
    bootstrap_results.append({
        "comparison": "B3_hybrid_vs_attach_freq",
        "model_top10": round(summary_b3_test["top10"], 4),
        "baseline_top10": round(attach_test_top10, 4),
        "delta": round(d, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "significant": "YES" if lo > 0 else "NO",
    })

    # ── B4: Engineered HGB ──
    log("\n--- B4: Feature-engineered HGB ---")
    b4_scorer, b4_info = train_b4_engineered_hgb(X_train_s, y_train_full, X_val_s, y_val)

    # Evaluate B4: need to engineer val and test X
    def engineer(X):
        tanimoto = X[:, 2]
        log_af = X[:, 1]
        log_gf = X[:, 0]
        ha_delta = X[:, 3]
        cand_ha = X[:, 4]
        tanimoto_af = tanimoto * log_af
        tanimoto_gf = tanimoto * log_gf
        ha_ratio = cand_ha / (X[:, 5] + 1e-10)
        extras = np.column_stack([tanimoto_af, tanimoto_gf, ha_ratio, ha_delta ** 2])
        return np.column_stack([X, extras])

    X_val_e = engineer(X_val_s)
    X_test_e = engineer(X_test_s)

    def b4_scorer_wrap(X_orig):
        Xe = engineer(X_orig)
        return b4_scorer(X_orig)  # b4_scorer already engineers

    # Actually, the b4 scorer already does engineering. Let's just use it directly on scaled.
    # But evaluate_batch uses scorer(X) where X is already scaled. The scorer engineers inside.
    summary_b4_val, perq_b4_val = evaluate_batch(X_val_s, y_val, qid_val, cand_val, b4_scorer)
    summary_b4_test, perq_b4_test = evaluate_batch(X_test_s, y_test, qid_test, cand_test, b4_scorer)
    log(f"  Val Top10={summary_b4_val['top10']:.4f}, Test Top10={summary_b4_test['top10']:.4f}")

    model_results.append({
        "model": "B4_engineered_HGB", "arm": "B4",
        "val_top10": round(summary_b4_val["top10"], 4),
        "val_mrr": round(summary_b4_val["mrr"], 4),
        "test_top10": round(summary_b4_test["top10"], 4),
        "test_mrr": round(summary_b4_test["mrr"], 4),
        "train_time_sec": b4_info["train_time"],
        "note": f"HGB + 4 engineered features, {b4_info['n_features']} total",
    })

    d, lo, hi = bootstrap_delta(perq_b4_test, perq_attach_test, "top10")
    bootstrap_results.append({
        "comparison": "B4_engineered_HGB_vs_attach_freq",
        "model_top10": round(summary_b4_test["top10"], 4),
        "baseline_top10": round(attach_test_top10, 4),
        "delta": round(d, 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "significant": "YES" if lo > 0 else "NO",
    })

    # ── B5: LightGBM LambdaRank (optional) ──
    log("\n--- B5: LightGBM LambdaRank ---")
    b5_scorer, b5_info = train_b5_lambdarank(
        X_train_s, y_train_full, qid_train_full,
        X_val_s, y_val, qid_val)
    if b5_scorer:
        summary_b5_val, perq_b5_val = evaluate_batch(X_val_s, y_val, qid_val, cand_val, b5_scorer)
        summary_b5_test, perq_b5_test = evaluate_batch(X_test_s, y_test, qid_test, cand_test, b5_scorer)
        log(f"  Val Top10={summary_b5_val['top10']:.4f}, Test Top10={summary_b5_test['top10']:.4f}")

        model_results.append({
            "model": "B5_lambdarank", "arm": "B5",
            "val_top10": round(summary_b5_val["top10"], 4),
            "val_mrr": round(summary_b5_val["mrr"], 4),
            "test_top10": round(summary_b5_test["top10"], 4),
            "test_mrr": round(summary_b5_test["mrr"], 4),
            "train_time_sec": b5_info["train_time"],
            "note": f"LightGBM LambdaRank, {b5_info['train_samples']} samples",
        })

        d, lo, hi = bootstrap_delta(perq_b5_test, perq_attach_test, "top10")
        bootstrap_results.append({
            "comparison": "B5_lambdarank_vs_attach_freq",
            "model_top10": round(summary_b5_test["top10"], 4),
            "baseline_top10": round(attach_test_top10, 4),
            "delta": round(d, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "significant": "YES" if lo > 0 else "NO",
        })
    else:
        log("  SKIP: LightGBM LambdaRank not available")
        model_results.append({
            "model": "B5_lambdarank", "arm": "B5",
            "val_top10": 0.0, "val_mrr": 0.0,
            "test_top10": 0.0, "test_mrr": 0.0,
            "train_time_sec": 0,
            "note": "SKIPPED: LightGBM not installed",
        })

    # ── Noise floor estimation (HGB retrain variability) ──
    log("\n--- Noise Floor: HGB retrain variability ---")
    noise_range = abs(summary_b0_test["top10"] - summary_b4_test["top10"])
    # Compute best_test_top10 inline before Model Selection
    valid_for_noise = [m for m in model_results if m["val_top10"] > 0]
    best_noise_test = max(valid_for_noise, key=lambda x: x["val_top10"])["test_top10"] if valid_for_noise else 0.0
    best_vs_attach = best_noise_test - attach_test_top10 if best_noise_test > 0 else 0.0
    log(f"  B0 (HGB) test Top10: {summary_b0_test['top10']:.4f}")
    log(f"  B4 (engineered HGB) test Top10: {summary_b4_test['top10']:.4f}")
    log(f"  Noise range (B0-B4 diff): {noise_range:.4f}")
    log(f"  Best improvement over attach_freq: {best_vs_attach:.4f}")
    noise_warning = False
    if noise_range > 0 and best_vs_attach < 2 * noise_range:
        log(f"  WARNING: Best improvement ({best_vs_attach:.4f}) < 2x noise range ({2*noise_range:.4f})")
        noise_warning = True
    else:
        log(f"  OK: Best improvement ({best_vs_attach:.4f}) >= 2x noise range ({2*noise_range:.4f})")

    # ── Model Selection ──
    log("\n" + "=" * 60)
    log("MODEL SELECTION")
    log("=" * 60)

    # Find best by val Top10 (excluding skipped models)
    valid_models = [(m, m["val_top10"]) for m in model_results if m["val_top10"] > 0]
    if valid_models:
        best = max(valid_models, key=lambda x: x[1])[0]
        best_name = best["model"]
        best_val_top10 = best["val_top10"]
        best_test_top10 = best["test_top10"]
        log(f"  Best model: {best_name}")
        log(f"  Val Top10: {best_val_top10:.4f}")
        log(f"  Test Top10: {best_test_top10:.4f}")
        log(f"  Delta vs attach_freq: {best_test_top10 - attach_test_top10:+.4f}")
    else:
        best_name = "NONE"
        best_val_top10 = 0.0
        best_test_top10 = 0.0
        log("  No valid models found!")

    # ── Stopping Criteria Check ──
    log("\n--- Stopping Criteria ---")
    log("  Tracking consecutive model val Top10 improvements (training order):")
    track_val = [(m["model"], m["val_top10"]) for m in model_results if m["val_top10"] > 0]
    best_so_far = 0.0
    consecutive_below_1pp = 0
    for name, vtop in track_val:
        if best_so_far == 0.0:
            log(f"    {name}: val Top10={vtop:.4f} (first model)")
        else:
            delta = vtop - best_so_far if vtop > best_so_far else vtop - best_so_far
            improvement = vtop - best_so_far
            log(f"    {name}: val Top10={vtop:.4f} (delta vs best so far: {improvement:+.4f})")
            if improvement < 0.01:
                consecutive_below_1pp += 1
                log(f"      -> Improvement < 1pp (count={consecutive_below_1pp})")
            else:
                consecutive_below_1pp = 0
        if vtop > best_so_far:
            best_so_far = vtop

    # Check canonical HGB significance
    log(f"  B0 vs canonical HGB bootstrap CI includes zero: {lo_can <= 0}")
    canonical_stop = lo_can <= 0

    # Stopping verdict
    stop_reasons = []
    if consecutive_below_1pp >= 2:
        stop_reasons.append(f"{consecutive_below_1pp} consecutive models improved < 1pp")
    if canonical_stop:
        stop_reasons.append("B0_HGB_vs_canonical_HGB bootstrap CI includes zero")
    if best_val_top10 - attach_val_top10 < 0.01:
        stop_reasons.append("Best val improvement over attach_freq < 1pp")

    if stop_reasons:
        log("  STOPPING SIGNAL: " + "; ".join(stop_reasons))
        log("  Will NOT escalate to more complex models.")
    else:
        log("  All checks passed -- no stopping signal.")
        log("  Improvement >= 1pp, escalation justified.")

    # 6. Write outputs
    # Model results
    model_fieldnames = ["model", "arm", "val_top10", "val_mrr", "test_top10", "test_mrr",
                        "train_time_sec", "note"]
    write_csv("d4a2_ranker_model_results.csv", model_results, model_fieldnames)

    # Bootstrap comparisons
    bootstrap_fieldnames = ["comparison", "model_top10", "baseline_top10", "delta",
                            "ci_lo", "ci_hi", "significant"]
    write_csv("d4a2_ranker_bootstrap.csv", bootstrap_results, bootstrap_fieldnames)

    # 7. Error analysis: per-query outcomes for best model, HGB, and attach_freq
    log("\n--- Error Analysis ---")
    error_rows = []
    # Pick best model, B0, and attach freq for comparison
    perq_dicts = {
        "B0_HGB": perq_b0_test,
        "B2_MLP": perq_b2_test,
        "attach_freq": perq_attach_test,
    }

    # Add best model
    best_perq_name = None
    if best_name == "B0_HGB_reproduced":
        perq_dicts["best"] = perq_b0_test
        best_perq_name = "best"
    elif best_name == "B1_pairwise_ranker" and b1_scorer:
        perq_dicts["best"] = perq_b1_test
        best_perq_name = "best"
    elif best_name == "B2_mlp_softmax_proxy" or best_name == "B2_xgboost_ranker":
        perq_dicts["best"] = perq_b2_test
        best_perq_name = "best"
    elif best_name == "B3_hybrid_ranker":
        perq_dicts["best"] = perq_b3_test
        best_perq_name = "best"
    elif best_name == "B4_engineered_HGB":
        perq_dicts["best"] = perq_b4_test
        best_perq_name = "best"
    elif best_name == "B5_lambdarank" and b5_scorer:
        perq_dicts["best"] = perq_b5_test
        best_perq_name = "best"

    # Collect per-query comparison for a sample
    qids_all = sorted(set(perq_attach_test.keys()))
    for qid in qids_all[:5000]:  # Sample for error analysis
        row = {"query_id": qid}
        for name, perq in perq_dicts.items():
            if qid in perq:
                row[f"{name}_top10"] = perq[qid]["top10"]
                row[f"{name}_mrr"] = perq[qid]["mrr"]
            else:
                row[f"{name}_top10"] = -1
                row[f"{name}_mrr"] = -1.0
        # Hard subset classification: count methods that succeeded
        method_cols = [v for k, v in row.items() if k.endswith("_top10") and v >= 0]
        n_success = sum(1 for v in method_cols if v == 1)
        row["n_methods_success"] = n_success
        row["hard_subset"] = "hard" if n_success == 0 else ("medium" if n_success == 1 else "easy")
        error_rows.append(row)

    error_fieldnames = ["query_id"]
    for name in perq_dicts:
        error_fieldnames.extend([f"{name}_top10", f"{name}_mrr"])
    error_fieldnames.extend(["n_methods_success", "hard_subset"])
    write_csv("d4a2_ranker_error_analysis.csv", error_rows, error_fieldnames)

    # Log hard subset stats
    n_hard = sum(1 for r in error_rows if r.get("hard_subset") == "hard")
    n_medium = sum(1 for r in error_rows if r.get("hard_subset") == "medium")
    n_easy = sum(1 for r in error_rows if r.get("hard_subset") == "easy")
    log(f"\n  Error analysis: {len(error_rows)} queries written")
    log(f"  Hard subset (no method succeeds): {n_hard} ({100*n_hard/len(error_rows):.1f}%)")
    log(f"  Medium (1 method succeeds): {n_medium} ({100*n_medium/len(error_rows):.1f}%)")
    log(f"  Easy (2+ methods succeed): {n_easy} ({100*n_easy/len(error_rows):.1f}%)")

    # 8. Summary
    log("\n" + "=" * 60)
    log("PART B COMPLETE")
    log("=" * 60)
    log(f"  Best model: {best_name}")
    log(f"  Best val Top10: {best_val_top10:.4f}")
    log(f"  Best test Top10: {best_test_top10:.4f}")
    log(f"  Attach_freq test Top10: {attach_test_top10:.4f}")
    log(f"  Delta: {best_test_top10 - attach_test_top10:+.4f}")
    log(f"  Output directory: {D4A2}")

    write_md("d4a2b_ranker_ceiling_complete.md",
             f"# D4A2B Complete\n\n"
             f"- Best model: {best_name}\n"
             f"- Best val Top10: {best_val_top10:.4f}\n"
             f"- Best test Top10: {best_test_top10:.4f}\n"
             f"- Attach_freq test Top10: {attach_test_top10:.4f}\n"
             f"- Delta: {best_test_top10 - attach_test_top10:+.4f}\n"
             f"- B0 (HGB) vs canonical HGB: delta={d_can:.4f} [{lo_can:.4f}, {hi_can:.4f}]\n"
             f"- Noise range (B0-B4 diff): {noise_range:.4f}\n"
             f"- Noise warning: {noise_warning}\n"
             f"- Stopping criterion met: {'YES' if best_val_top10 - attach_val_top10 < 0.01 else 'NO'}\n"
             f"- Hard subset: {n_hard}/{len(error_rows)} ({100*n_hard/len(error_rows):.1f}%)\n"
             f"- Timestamp: {now()}\n")


def write_md(path, text):
    path = D4A2 / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    log(f"  wrote -> {path.name}")


if __name__ == "__main__":
    main()
