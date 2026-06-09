#!/usr/bin/env python3
"""Pseudo-labeling: LBC discovers candidates from 2M pool (batched, no OOM)."""
import json, glob, time, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 20260603; OUT = Path("technique_transfer/results")
OUT.mkdir(parents=True, exist_ok=True); RDLogger.DisableLog("rdApp.*")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Load labeled data + train LBC (same as before) ──
log("Loading labeled data...")
manifest = {}
mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
with mp.open(encoding="utf-8") as f:
    for line in f:
        if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q

def mfp(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    fp = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), fp)
    return fp
def mprops(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    return np.array([m.GetNumHeavyAtoms(), Descriptors.RingCount(m),
        Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m)], dtype=np.float32)

all_ofs = sorted({q["old_fragment_smiles"] for q in manifest.values()})
of_fps, of_props = {}, {}
for s in all_ofs:
    fp = mfp(s); props = mprops(s)
    if fp is not None and props is not None: of_fps[s] = fp; of_props[s] = props

qlabels = defaultdict(dict)
for split in ["train", "test"]:
    pattern = f"plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/matrices/{split}/{split}_features_shard_0*.jsonl"
    for shard in sorted(glob.glob(pattern)):
        with open(shard, encoding="utf-8") as f:
            for l in f:
                if not l.strip(): continue
                r = json.loads(l)
                if r["query_id"] in manifest: qlabels[r["query_id"]][r["candidate"]] = int(r["label"])

cand_fps, cand_props = {}, {}
for labels in qlabels.values():
    for c in labels:
        if c in cand_fps: continue
        fp = mfp(c); props = mprops(c)
        if fp is not None and props is not None: cand_fps[c] = fp; cand_props[c] = props
cand_list = sorted(cand_fps.keys())
log(f"  {len(cand_list)} labeled candidates")

labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in qlabels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})
zero_ofs = sorted(set(all_ofs) - set(labeled_ofs))
log(f"  {len(labeled_ofs)} labeled OFs, {len(zero_ofs)} zero-positive OFs")

# Train freq
freq_map = {}
for o in labeled_ofs:
    for qid in [q for q in manifest if manifest[q]["old_fragment_smiles"] == o]:
        for c, lbl in qlabels.get(qid, {}).items():
            if lbl == 1: freq_map[c] = freq_map.get(c, 0) + 1
max_f = max(freq_map.values()) if freq_map else 1

# Build training matrix
X_tr, y_tr = [], []
for o in labeled_ofs:
    ofp = of_fps[o]; op = of_props[o]; ofp_n = ofp / (ofp.sum() + 1e-10)
    for qid in [q for q in manifest if manifest[q]["old_fragment_smiles"] == o]:
        labels = qlabels.get(qid, {})
        if not labels: continue
        cands = list(labels.keys()); n_c = len(cands)
        feats = np.zeros((n_c, 8), dtype=np.float32)
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
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                abs(cp[4]-op[4])/max(op[4]+1,1), freq_map.get(c, 0)/max(max_f, 1)]
        X_tr.append(feats); y_tr.extend([labels[c] for c in cands])
X_tr = np.vstack(X_tr); y_tr = np.array(y_tr, dtype=np.int8)
scaler = StandardScaler(); X_s = scaler.fit_transform(X_tr)
lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
lr.fit(X_s, y_tr)
log(f"  Trained on {len(X_tr):,} pairs")

# ── Batched pool scan: for each OF, batch through pool files ──
pool_files = sorted(glob.glob("chembl_candidate_pool/batch_*.parquet"))
N_TOP = 2000
OF_TARGETS = [
    ("*c1ccccc1Cl", "labeled high-support"),
    ("*c1ccc(OC)cc1", "labeled high-support"),
    ("*CCCC", "labeled low-support"),
    ("*c1ccc(Cl)cc1", "zero-positive"),
    ("*c1cccc(OC)c1", "zero-positive"),
    ("*C(=O)c1cccs1", "zero-positive"),
]

log(f"\n=== TOP-20 LBC CANDIDATES (batched, N={N_TOP}) ===")

for of_s, label in OF_TARGETS:
    if of_s not in of_fps: continue
    ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
    of_sum = ofp.sum()

    # Accumulate top candidates across batches
    top_sims = np.zeros(N_TOP)  # similarity values
    top_smis = [""] * N_TOP
    top_props = [None] * N_TOP

    for pf in pool_files:
        df = pd.read_parquet(pf)
        batch_fps = np.array([np.frombuffer(b, dtype=np.float32) for b in df["fp"].values])
        batch_smis = df["smiles"].values
        batch_sums = batch_fps.sum(axis=1)

        # Tanimoto to OF
        inter = np.dot(batch_fps, ofp.astype(np.float32))
        denom = batch_sums + of_sum - inter; denom[denom == 0] = 1.0
        sims = inter / denom

        # Merge with top-N
        all_sims = np.concatenate([top_sims, sims])
        all_smis = top_smis + list(batch_smis)
        all_props = top_props + [
            np.array([r["heavy_atoms"], r["rings"], r["mw"], r["logp"], r["tpsa"]], dtype=np.float32)
            for _, r in df.iterrows()
        ]
        best_idx = np.argsort(-all_sims)[:N_TOP]
        top_sims = all_sims[best_idx]
        top_smis = [all_smis[i] for i in best_idx]
        top_props = [all_props[i] for i in best_idx]

    # Compute LBC features for top-N
    feats = np.zeros((N_TOP, 8), dtype=np.float32)
    for i, (smi, cp) in enumerate(zip(top_smis, top_props)):
        inter_v = float(np.dot(cand_fps.get(smi, np.zeros(2048, dtype=np.float32)), ofp)) if smi in cand_fps else top_sims[i] * (of_sum + cp[2]) / 2
        morgan_v = top_sims[i]
        # For bit_corr, approximate or compute
        cfp_full = mfp(smi)
        bit_c = 0.0
        if cfp_full is not None and cfp_full.sum() > 0:
            cfp_n = cfp_full / (cfp_full.sum() + 1e-10)
            if ofp.sum() > 0:
                corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                bit_c = corr if np.isfinite(corr) else 0.0
        feats[i] = [morgan_v, bit_c, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
            abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
            abs(cp[4]-op[4])/max(op[4]+1,1), freq_map.get(smi, 0)/max(max_f, 1)]

    scores = lr.predict_proba(scaler.transform(feats))[:, 1]
    best = np.argsort(-scores)[:20]

    print(f"\n{of_s} ({label}):")
    for rank, bidx in enumerate(best):
        smi = top_smis[bidx]; sc = scores[bidx]
        in_labeled = "★" if smi in cand_fps else " "
        m = top_sims[bidx]
        print(f"  {rank+1:2d}. {smi:30s}  morgan={m:.3f}  score={sc:.4f}  {in_labeled}")

log("\n★ = already in labeled candidate set")
log("Done")
