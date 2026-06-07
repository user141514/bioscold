#!/usr/bin/env python3
"""T4: Contrastive Ranking Loss — replace binary CE with pairwise margin ranking loss.

Implements margin-based ranking loss for LBC-Ranker using PyTorch.
All feature computation identical to baseline. Only training objective changes.

The margin ranking loss optimizes:
  loss = max(0, margin - (score_positive - score_negative))
for each (positive, negative) pair within a query.

A/B design: same 10 seeds, same split, same features. Only loss differs.
"""

import json, glob, time, itertools, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

SEED_BASE = 20260603
FEATURE_NAMES = ["morgan", "bit_corr", "dHeavy", "dRings", "dMW", "dLogP", "dTPSA", "freq"]

OUT = Path("technique_transfer/results")
OUT.mkdir(parents=True, exist_ok=True)

RDLogger.DisableLog("rdApp.*")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── static data loading (same as baseline) ──
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

def load_all_data():
    log("Loading data...")
    manifest = {}
    mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
    with mp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q

    all_ofs = sorted({q["old_fragment_smiles"] for q in manifest.values()})
    of_fps, of_props = {}, {}
    for s in all_ofs:
        fp = morgan_fp(s); props = mol_props(s)
        if fp is not None and props is not None: of_fps[s] = fp; of_props[s] = props

    of_queries = defaultdict(list)
    for q in manifest.values():
        if q["old_fragment_smiles"] in of_fps: of_queries[q["old_fragment_smiles"]].append(q["query_id"])

    query_labels = defaultdict(dict)
    for split in ["train", "test"]:
        pattern = f"plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/matrices/{split}/{split}_features_shard_0*.jsonl"
        for shard in sorted(glob.glob(pattern)):
            with open(shard, encoding="utf-8") as f:
                for l in f:
                    if not l.strip(): continue
                    r = json.loads(l)
                    if r["query_id"] in manifest: query_labels[r["query_id"]][r["candidate"]] = int(r["label"])

    cand_fps, cand_props = {}, {}
    for labels in query_labels.values():
        for c in labels:
            if c in cand_fps: continue
            fp = morgan_fp(c); props = mol_props(c)
            if fp is not None and props is not None: cand_fps[c] = fp; cand_props[c] = props
    cand_list = sorted(cand_fps.keys())

    labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in query_labels.items()
        if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})

    # Precompute features (7 static + freq filled per split)
    of_data = {}
    for of_s in labeled_ofs:
        ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
        Xl, yl, qinfo = [], [], []
        for qid in of_queries.get(of_s, []):
            labels = query_labels.get(qid, {})
            if not labels: continue
            cands = list(labels.keys()); n_c = len(cands)
            feats = np.zeros((n_c, 7), dtype=np.float32)
            cil = []
            for idx, c in enumerate(cands):
                cfp = cand_fps.get(c); cp = cand_props.get(c)
                if cfp is None or cp is None: continue
                inter = float(cfp.dot(ofp)); denom = float(cfp.sum() + ofp.sum() - inter)
                morgan_v = inter / max(denom, 0.001)
                cfp_n = cfp / (cfp.sum() + 1e-10); bit_c = 0.0
                if cfp.sum() > 0 and ofp.sum() > 0:
                    corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                    bit_c = corr if np.isfinite(corr) else 0.0
                feats[idx] = [morgan_v, bit_c, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                    abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1), abs(cp[4]-op[4])/max(op[4]+1,1)]
                cil.append(cand_list.index(c) if c in cand_list else -1)
            Xl.append(feats); yl.append(np.array([labels[c] for c in cands], dtype=np.int8)); qinfo.append((qid, n_c, cil))
        if Xl: of_data[of_s] = {"X7": np.vstack(Xl), "y": np.concatenate(yl), "queries": qinfo}

    return {
        "labeled_ofs": labeled_ofs, "of_queries": of_queries, "cand_list": cand_list,
        "of_data": of_data, "of_fps": of_fps, "of_props": of_props,
        "cand_fps": cand_fps, "cand_props": cand_props, "manifest": manifest, "query_labels": query_labels,
    }

# ── PyTorch LR with margin ranking loss ──
class LBCRanker(torch.nn.Module):
    """Linear ranker with margin ranking loss."""
    def __init__(self, n_features=8):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)

def pairwise_margin_loss(scores, labels, margin=0.1, max_pairs_per_query=500):
    """Margin ranking loss: push positives above negatives.

    For each query, pair each positive with sampled negatives.
    loss = max(0, margin - (s_pos - s_neg))

    scores: [n_candidates] predicted scores for one query
    labels: [n_candidates] binary labels
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, device=scores.device)

    # Sample pairs (limit to max_pairs_per_query)
    n_pairs = min(len(pos_idx) * len(neg_idx), max_pairs_per_query)
    if len(pos_idx) * len(neg_idx) > n_pairs:
        pos_sampled = pos_idx[torch.randint(0, len(pos_idx), (n_pairs,), device=scores.device)]
        neg_sampled = neg_idx[torch.randint(0, len(neg_idx), (n_pairs,), device=scores.device)]
    else:
        pos_sampled = pos_idx.repeat_interleave(len(neg_idx))
        neg_sampled = neg_idx.repeat(len(pos_idx))

    s_pos = scores[pos_sampled]
    s_neg = scores[neg_sampled]
    loss = torch.clamp(margin - (s_pos - s_neg), min=0).mean()
    return loss

# ── Training ──
def train_ranking_model(X_train, y_train, query_info_train, n_epochs=20, lr=0.01, margin=0.1, l2_weight=0.001):
    """Train LBC-Ranker with margin ranking loss.

    query_info_train: list of (qid, n_cands, cand_indices) — maintains query boundaries.
    """
    device = torch.device("cpu")  # small model, CPU fine
    model = LBCRanker(8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)

    # Build query boundaries
    offsets = [0]
    for _, n_c, _ in query_info_train:
        offsets.append(offsets[-1] + n_c)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_queries = 0

        for i in range(len(query_info_train)):
            start, end = offsets[i], offsets[i+1]
            X_q = X_t[start:end]
            y_q = y_t[start:end]

            if y_q.sum() == 0:  # no positives in this query
                continue

            scores = model(X_q)
            loss = pairwise_margin_loss(scores, y_q, margin=margin)
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_queries += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / max(n_queries, 1)
            # Not printing per-epoch to avoid noise

    model.eval()
    return model

def predict_scores(model, X, scaler):
    """Predict scores using trained model + scaler."""
    X_scaled = torch.FloatTensor(scaler.transform(X))
    with torch.no_grad():
        scores = model(X_scaled).numpy()
    return scores

# ── Evaluation (same as baseline) ──
def hit10_vec(labels_arr, scores):
    order = np.argsort(-scores)
    ranked = labels_arr[order]
    pos = np.where(ranked == 1)[0]
    return int(len(pos) > 0 and pos[0] < 10)

def run_baseline_comparison(st):
    """Run 3-seed comparison: baseline CE vs T4 margin ranking loss. Same features, same splits."""
    labeled_ofs = st["labeled_ofs"]
    results = []

    for seed_idx in range(3):  # 3 seeds for quick verification
        log(f"--- Seed {seed_idx} ---")
        rng = np.random.RandomState(SEED_BASE + seed_idx)
        shuffled = list(labeled_ofs); rng.shuffle(shuffled)
        n_tr = int(len(shuffled)*0.7)
        train_ofs, test_ofs = shuffled[:n_tr], shuffled[n_tr:]

        # Train frequency (train-only, same as baseline)
        freq_map = {}
        for o in train_ofs:
            y = st["of_data"][o]["y"]; queries = st["of_data"][o]["queries"]
            off = 0
            for _, nc, cil in queries:
                for j in range(nc):
                    if y[off+j] == 1:
                        ci = cil[j]
                        if ci >= 0: freq_map[st["cand_list"][ci]] = freq_map.get(st["cand_list"][ci], 0) + 1
                off += nc
        max_f = max(freq_map.values()) if freq_map else 1

        # Build full 8-dim feature matrix (same as baseline)
        X_tr_7 = np.vstack([st["of_data"][o]["X7"] for o in train_ofs])
        y_tr = np.concatenate([st["of_data"][o]["y"] for o in train_ofs])
        X_tr = np.zeros((len(X_tr_7), 8), dtype=np.float32)
        X_tr[:, :7] = X_tr_7

        # Build query info with freq
        query_info = []
        off = 0
        for o in train_ofs:
            for qid, nc, cil in st["of_data"][o]["queries"]:
                for j in range(nc):
                    ci = cil[j]; c = st["cand_list"][ci] if ci >= 0 else ""
                    X_tr[off+j, 7] = freq_map.get(c, 0) / max(max_f, 1)
                query_info.append((qid, nc, cil))
                off += nc

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        # A: Baseline CE (sklearn LR)
        from sklearn.linear_model import LogisticRegression
        lr_ce = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED_BASE)
        lr_ce.fit(X_tr_s, y_tr)

        # B: T4 Margin Ranking Loss (PyTorch LR)
        log("  Training T4 ranking model...")
        model_t4 = train_ranking_model(X_tr_s, y_tr, query_info, n_epochs=20, lr=0.01, margin=0.1, l2_weight=0.001)

        # Evaluate
        seed_results = {"seed": seed_idx, "n_train": len(train_ofs), "n_test": len(test_ofs)}
        ce_h10_list, t4_h10_list = [], []

        for o in test_ofs:
            queries = st["of_data"][o]["queries"]
            y_te = st["of_data"][o]["y"]
            X7 = st["of_data"][o]["X7"]
            off = 0
            for _, nc, cil in queries:
                cands = [st["cand_list"][ci] if ci >= 0 else "" for ci in cil]
                labels_dict = dict(zip(cands, y_te[off:off+nc].astype(int)))
                if sum(labels_dict.values()) == 0:
                    off += nc; continue

                X_q = np.zeros((nc, 8), dtype=np.float32)
                X_q[:, :7] = X7[off:off+nc]
                X_q[:, 7] = [freq_map.get(c, 0) / max(max_f, 1) for c in cands]
                X_q_s = scaler.transform(X_q)

                # CE scores
                ce_scores = lr_ce.predict_proba(X_q_s)[:, 1]
                # T4 scores
                t4_scores = predict_scores(model_t4, X_q, scaler)

                ce_h10_list.append(hit10_vec(np.array([labels_dict[c] for c in cands], dtype=np.int8), ce_scores))
                t4_h10_list.append(hit10_vec(np.array([labels_dict[c] for c in cands], dtype=np.int8), t4_scores))
                off += nc

        ce_macro = float(np.mean(ce_h10_list)) if ce_h10_list else 0.0
        t4_macro = float(np.mean(t4_h10_list)) if t4_h10_list else 0.0
        delta = t4_macro - ce_macro

        seed_results["ce_h10"] = ce_macro
        seed_results["t4_h10"] = t4_macro
        seed_results["delta"] = delta
        results.append(seed_results)
        log(f"  Baseline CE: {ce_macro:.4f}  |  T4 MarginRank: {t4_macro:.4f}  |  Δ = {delta:+.4f}")

    # Summary
    df = pd.DataFrame(results)
    df.to_csv(OUT / "t4_ranking_loss_vs_baseline.csv", index=False)
    log(f"\n=== T4 Results ===\n{df.to_string(index=False)}")
    log(f"CE mean: {df['ce_h10'].mean():.4f}  |  T4 mean: {df['t4_h10'].mean():.4f}  |  Δ mean: {df['delta'].mean():+.4f}")

    return df


if __name__ == "__main__":
    t0 = time.time()
    st = load_all_data()
    log(f"Data loaded: {len(st['labeled_ofs'])} OFs, {sum(len(st['of_data'][o]['X7']) for o in st['labeled_ofs']):,} pairs")
    run_baseline_comparison(st)
    log(f"Total: {(time.time()-t0)/60:.1f} min")
