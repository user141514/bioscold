#!/usr/bin/env python3
"""Quick verification: full-data LR vs HGB, + C3F baseline check."""
import json, glob, time, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

SEED = 20260603
FEATURE_NAMES = ["morgan", "bit_corr", "dHeavy", "dRings", "dMW", "dLogP", "dTPSA", "freq"]

RDLogger.DisableLog("rdApp.*")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── fingerprint / props ──
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

# ── data loading (no shard limit) ──
log("Loading manifest...")
manifest = {}
manifest_path = Path(
    "E:/zuhui/bioisosteric_diffusion/plan_results/routeA_chembl37k_d0d3_engineering_safe/"
    "07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl"
)
with manifest_path.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            manifest[q["query_id"]] = q

all_ofs = sorted({q["old_fragment_smiles"] for q in manifest.values()})
log(f"Manifest: {len(manifest)} queries, {len(all_ofs)} unique OFs")

# OF fingerprints
of_fps, of_props = {}, {}
for s in all_ofs:
    fp = morgan_fp(s); props = mol_props(s)
    if fp is not None and props is not None:
        of_fps[s] = fp; of_props[s] = props
log(f"Valid OFs: {len(of_fps)}")

of_queries = defaultdict(list)
for q in manifest.values():
    of_s = q["old_fragment_smiles"]
    if of_s in of_fps:
        of_queries[of_s].append(q["query_id"])

# Load ALL shards
log("Loading labels from ALL shards...")
query_labels = defaultdict(dict)
for split in ["train", "test"]:
    pattern = (
        f"E:/zuhui/bioisosteric_diffusion/plan_results/routeA_chembl37k_d0d3_engineering_safe/"
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
        if (si + 1) % 50 == 0:
            log(f"    {split} shard {si+1}/{len(shards)}")

log(f"Total queries with labels: {len(query_labels)}")

# Candidate fingerprints
log("Computing candidate fingerprints...")
cand_fps, cand_props = {}, {}
for labels in query_labels.values():
    for c in labels:
        if c in cand_fps: continue
        fp = morgan_fp(c); props = mol_props(c)
        if fp is not None and props is not None:
            cand_fps[c] = fp; cand_props[c] = props
log(f"Unique candidates: {len(cand_fps)}")

# Identify labeled OFs
labeled_ofs = sorted({
    manifest[qid]["old_fragment_smiles"]
    for qid, labels in query_labels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())
})
log(f"Labeled OFs: {len(labeled_ofs)}")

# ── feature building ──
log("Building features...")
of_X, of_y, of_qids, of_clists = {}, {}, {}, {}
for of_s in labeled_ofs:
    X_parts, y_parts, qids, clists = [], [], [], []
    ofp = of_fps[of_s]; op = of_props[of_s]
    ofp_norm = ofp / (ofp.sum() + 1e-10)
    for qid in of_queries.get(of_s, []):
        labels = query_labels.get(qid, {})
        if not labels: continue
        candidates = list(labels.keys())
        feats = np.zeros((len(candidates), 8), dtype=np.float32)
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
                abs(cp[4] - op[4]) / max(op[4] + 1, 1), 0.0]
        X_parts.append(feats); y_parts.append(np.array([labels[c] for c in candidates], dtype=np.int8))
        qids.append(qid); clists.append(candidates)
    if X_parts:
        of_X[of_s] = np.vstack(X_parts); of_y[of_s] = np.concatenate(y_parts)
        of_qids[of_s] = qids; of_clists[of_s] = clists

total_pairs = sum(len(of_X[o]) for o in labeled_ofs)
log(f"Total pairs: {total_pairs:,}")

# ── helpers ──
def hit10(labels, scores, candidates):
    order = np.argsort(-scores)
    ranked = np.array([labels[c] for c in [candidates[i] for i in order]])
    pos = np.where(ranked == 1)[0]
    return int(len(pos) > 0 and pos[0] < 10)

def get_freqs(train_ofs):
    freq = {}
    for of_s in train_ofs:
        for qid in of_queries.get(of_s, []):
            for c, lbl in query_labels.get(qid, {}).items():
                if lbl == 1: freq[c] = freq.get(c, 0) + 1
    return freq, max(freq.values()) if freq else 1

def with_freq(of_s, freq, max_freq):
    X = of_X[of_s].copy()
    clists = of_clists[of_s]
    offset = 0
    for candidates in clists:
        n = len(candidates)
        X[offset:offset+n, 7] = np.array([freq.get(c, 0)/max(max_freq,1) for c in candidates], dtype=np.float32)
        offset += n
    return X

def content_score(c, of_s):
    if c not in cand_fps: return 0.0
    cfp = cand_fps[c]; ofp = of_fps[of_s]
    inter = float(cfp.dot(ofp))
    denom = float(cfp.sum() + ofp.sum() - inter)
    morgan = inter / max(denom, 0.001)
    cp = cand_props.get(c); op = of_props[of_s]
    if cp is None: return morgan
    d_h = abs(cp[0] - op[0]); d_r = abs(cp[1] - op[1])
    d_mw = abs(cp[2] - op[2]) / max(op[2], 1)
    d_lp = abs(cp[3] - op[3]) / max(abs(op[3]) + 1, 1)
    pc = 1.0 / (1.0 + d_h + d_r + d_mw + d_lp)
    return 0.7 * morgan + 0.3 * pc

def random_score(c, of_s):
    return np.random.random()

# ── C3F baseline ──
def c3f_score(of_s, candidates, freq, alpha=0.3, K=5):
    """Collaborative filtering: K-nearest OFs by candidate frequency overlap."""
    scores = np.zeros(len(candidates))
    # Get OFs with positive labels
    other_ofs = [o for o in labeled_ofs if o != of_s]
    if not other_ofs:
        return np.full(len(candidates), alpha * np.mean([content_score(c, of_s) for c in candidates]))

    # Simple collaborative: for each candidate, sum freq across other OFs
    for i, c in enumerate(candidates):
        collab = freq.get(c, 0) / max(max_freq, 1)  # already from train OFs only
        ca = content_score(c, of_s)
        scores[i] = alpha * collab + (1 - alpha) * ca
    return scores

# ── main comparison ──
log("=" * 60)
log("VERIFICATION: LR vs HGB vs CA vs C3F (full 123+ OFs)")

# Single 70/30 split
rng = np.random.RandomState(SEED)
shuffled = list(labeled_ofs)
rng.shuffle(shuffled)
n_train = int(len(shuffled) * 0.7)
train_ofs = shuffled[:n_train]
test_ofs = shuffled[n_train:]
log(f"Split: {len(train_ofs)} train / {len(test_ofs)} test OFs")

freq, max_freq = get_freqs(train_ofs)

# ── Train LR ──
log("Training LR...")
X_train_lr = np.vstack([with_freq(o, freq, max_freq) for o in train_ofs])
y_train_lr = np.concatenate([of_y[o] for o in train_ofs])
scaler_lr = StandardScaler()
X_tr_s = scaler_lr.fit_transform(X_train_lr)
lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
lr.fit(X_tr_s, y_train_lr)

# ── Train HGB ──
log("Training HGB...")
X_train_hgb = X_train_lr.copy()  # same features
X_tr_hgb_s = scaler_lr.transform(X_train_hgb)  # same scaler
hgb = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=SEED)
hgb.fit(X_tr_hgb_s, y_train_lr)

# ── Evaluate ──
log("Evaluating...")
results_macro = defaultdict(float)
n_eval = 0
per_of_results = []

for of_s in test_ofs:
    X = with_freq(of_s, freq, max_freq)
    X_s = scaler_lr.transform(X)
    y = of_y[of_s]
    clists = of_clists[of_s]

    lr_proba = lr.predict_proba(X_s)[:, 1]
    hgb_proba = hgb.predict_proba(X_s)[:, 1]

    offset = 0
    counts = defaultdict(int)
    n_q = 0
    for candidates in clists:
        n_c = len(candidates)
        labels = dict(zip(candidates, y[offset:offset+n_c].astype(int)))
        if sum(labels.values()) == 0:
            offset += n_c; continue
        n_q += 1

        lr_sc = lr_proba[offset:offset+n_c]
        hgb_sc = hgb_proba[offset:offset+n_c]
        ca_sc = np.array([content_score(c, of_s) for c in candidates])
        freq_sc = np.array([freq.get(c, 0)/max(max_freq,1) for c in candidates])
        c3f_sc = c3f_score(of_s, candidates, freq)

        counts["lr"] += hit10(labels, lr_sc, candidates)
        counts["hgb"] += hit10(labels, hgb_sc, candidates)
        counts["ca"] += hit10(labels, ca_sc, candidates)
        counts["freq"] += hit10(labels, freq_sc, candidates)
        counts["c3f"] += hit10(labels, c3f_sc, candidates)
        offset += n_c

    if n_q > 0:
        row = {"of": of_s, "n_queries": n_q}
        for k in ["lr", "hgb", "ca", "freq", "c3f"]:
            v = counts[k] / n_q
            row[k] = v
            results_macro[k] += v
        n_eval += 1
        per_of_results.append(row)
total_q = sum(r['n_queries'] for r in per_of_results)
log(f"Evaluated on {n_eval} OFs ({total_q} queries)")

print()
print("=" * 60)
print("QUICK VERIFICATION RESULTS (single 70/30 split)")
print("=" * 60)
print(f"Train OFs: {len(train_ofs)}, Test OFs: {n_eval}/{len(test_ofs)} with queries, Total labeled: {len(labeled_ofs)}")
header = f"{'Method':<20} {'Macro Hit@10':>14} {'vs CA delta':>12}"
print(header)
print("-" * len(header))
ca_macro = results_macro["ca"] / n_eval if n_eval > 0 else 0
for name, label in [("lr", "LBC-Kernel (LR)"), ("hgb", "HGB"), ("ca", "Content-Aware"), ("freq", "Frequency"), ("c3f", "C3F (K=5)")]:
    val = results_macro[name] / n_eval if n_eval > 0 else 0
    delta = val - ca_macro
    print(f"{label:<20} {val:>14.4f} {delta:>+12.4f}")
print()
log("Done.")
