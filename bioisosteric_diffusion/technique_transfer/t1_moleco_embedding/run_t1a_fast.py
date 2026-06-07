#!/usr/bin/env python3
"""T1a light: add AtomPair Dice similarity to baseline features. Uses cached AP sim."""

import json, glob, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 20260603
OUT = Path("technique_transfer/results")
SEEDS_N = 3

RDLogger.DisableLog("rdApp.*")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Load AP cache ──
log("Loading AP cache...")
ap_cache = pd.read_parquet("technique_transfer/t1_moleco_embedding/ap_similarity_cache.parquet")
ap_dict = {}
for _, r in ap_cache.iterrows():
    ap_dict[(r["of"], r["candidate"])] = r["ap_dice"]
log(f"  {len(ap_dict):,} entries")

# ── Load baseline data ──
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
log(f"  {len(of_fps)} valid OFs")

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
log(f"  {len(cand_list)} candidates")

labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in query_labels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})
log(f"  {len(labeled_ofs)} labeled OFs")

# Precompute 8 static features (baseline) + AP = 9 static
log("Building features...")
of_data = {}
for of_s in labeled_ofs:
    ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
    Xl, yl, qinfo = [], [], []
    for qid in of_queries.get(of_s, []):
        labels = query_labels.get(qid, {})
        if not labels: continue
        cands = list(labels.keys()); n_c = len(cands)
        feats = np.zeros((n_c, 8), dtype=np.float32)  # 7 baseline + AP static
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
            ap_v = ap_dict.get((of_s, c), 0.0)
            feats[idx] = [morgan_v, bit_c, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                abs(cp[4]-op[4])/max(op[4]+1,1), ap_v]
            cil.append(cand_list.index(c) if c in cand_list else -1)
        Xl.append(feats); yl.append(np.array([labels[c] for c in cands], dtype=np.int8)); qinfo.append((qid, n_c, cil))
    if Xl: of_data[of_s] = {"X": np.vstack(Xl), "y": np.concatenate(yl), "queries": qinfo}

total_pairs = sum(len(of_data[o]["X"]) for o in labeled_ofs)
log(f"  {total_pairs:,} total pairs")

def hit10(labels_dict, scores, candidates):
    order = np.argsort(-scores)
    ranked = np.array([labels_dict[c] for c in [candidates[i] for i in order]])
    pos = np.where(ranked == 1)[0]
    return int(len(pos) > 0 and pos[0] < 10)

# ── 3-seed comparison ──
log("\n=== T1a: Baseline (8F) vs +AtomPair (9F) ===")
results = []

for seed_idx in range(SEEDS_N):
    rng = np.random.RandomState(SEED + seed_idx)
    shuffled = list(labeled_ofs); rng.shuffle(shuffled)
    n_tr = int(len(shuffled)*0.7)
    train_ofs, test_ofs = shuffled[:n_tr], shuffled[n_tr:]

    freq_map = {}
    for o in train_ofs:
        y = of_data[o]["y"]; queries = of_data[o]["queries"]
        off = 0
        for _, nc, cil in queries:
            for j in range(nc):
                if y[off+j] == 1:
                    ci = cil[j]
                    if ci >= 0: freq_map[cand_list[ci]] = freq_map.get(cand_list[ci], 0) + 1
            off += nc
    max_f = max(freq_map.values()) if freq_map else 1

    # Build training matrices
    X_tr_static = np.vstack([of_data[o]["X"] for o in train_ofs])
    y_tr = np.concatenate([of_data[o]["y"] for o in train_ofs])
    n_rows = len(X_tr_static)

    # Baseline: 7 static + freq
    Xb = np.zeros((n_rows, 8), dtype=np.float32)
    Xb[:, :7] = X_tr_static[:, :7]
    # T1a: 8 static + freq (AP is col 7 in X)
    Xt = np.zeros((n_rows, 9), dtype=np.float32)
    Xt[:, :8] = X_tr_static[:, :8]

    off = 0
    for o in train_ofs:
        for _, nc, cil in of_data[o]["queries"]:
            for j in range(nc):
                ci = cil[j]; c = cand_list[ci] if ci >= 0 else ""
                fv = freq_map.get(c, 0) / max(max_f, 1)
                Xb[off+j, 7] = fv
                Xt[off+j, 8] = fv
            off += nc

    scaler_b = StandardScaler(); scaler_t = StandardScaler()
    Xb_s = scaler_b.fit_transform(Xb); Xt_s = scaler_t.fit_transform(Xt)

    lr_b = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr_b.fit(Xb_s, y_tr)
    lr_t = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr_t.fit(Xt_s, y_tr)

    # Evaluate
    b_h10, t_h10 = [], []
    for o in test_ofs:
        queries = of_data[o]["queries"]; y_te = of_data[o]["y"]; X_static = of_data[o]["X"]
        off = 0
        for _, nc, cil in queries:
            cands = [cand_list[ci] if ci >= 0 else "" for ci in cil]
            labels_dict = dict(zip(cands, y_te[off:off+nc].astype(int)))
            if sum(labels_dict.values()) == 0:
                off += nc; continue
            Xqb = np.zeros((nc, 8), dtype=np.float32)
            Xqt = np.zeros((nc, 9), dtype=np.float32)
            for j in range(nc):
                ci = cil[j]; c = cand_list[ci] if ci >= 0 else ""
                fv = freq_map.get(c, 0) / max(max_f, 1)
                Xqb[j, :7] = X_static[off+j, :7]; Xqb[j, 7] = fv
                Xqt[j, :8] = X_static[off+j, :8]; Xqt[j, 8] = fv
            bs = lr_b.predict_proba(scaler_b.transform(Xqb))[:, 1]
            ts = lr_t.predict_proba(scaler_t.transform(Xqt))[:, 1]
            b_h10.append(hit10(labels_dict, bs, cands))
            t_h10.append(hit10(labels_dict, ts, cands))
            off += nc

    bm = float(np.mean(b_h10)); tm = float(np.mean(t_h10))
    results.append({"seed": seed_idx, "base_h10": bm, "t1a_h10": tm, "delta": tm - bm})
    log(f"  Seed {seed_idx}: Baseline={bm:.4f}  T1a(+AP)={tm:.4f}  Δ={tm-bm:+.4f}")

df = pd.DataFrame(results)
df.to_csv(OUT / "t1a_atompair_vs_baseline.csv", index=False)
log(f"\n{df.to_string(index=False)}")
log(f"Base mean: {df['base_h10'].mean():.4f}  T1a mean: {df['t1a_h10'].mean():.4f}  Δ mean: {df['delta'].mean():+.4f}")
