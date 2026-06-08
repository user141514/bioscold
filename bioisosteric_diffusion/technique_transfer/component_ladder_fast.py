#!/usr/bin/env python3
"""Component ladder — fast version: Freq vs CA vs Freq+CA blend vs LBC (3 seeds)."""
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
OUT.mkdir(parents=True, exist_ok=True)
RDLogger.DisableLog("rdApp.*")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Load
log("Loading data...")
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

of_queries = defaultdict(list)
for q in manifest.values():
    if q["old_fragment_smiles"] in of_fps: of_queries[q["old_fragment_smiles"]].append(q["query_id"])

qlabels = defaultdict(dict)
for split in ["train", "test"]:
    pattern = (
        "plan_results/routeA_chembl37k_d0d3_engineering_safe/"
        f"07_d4a0_matrix_freeze/matrices/{split}/{split}_features_shard_0*.jsonl"
    )
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

labeled_ofs = sorted({
    manifest[qid]["old_fragment_smiles"] for qid, labels in qlabels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())
})
log(f"  {len(labeled_ofs)} OFs, {len(cand_list)} candidates")

# Precompute LBC features + CA scores (lam=0.75, known optimal)
of_data = {}
for of_s in labeled_ofs:
    ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
    Xl, yl, qinfo = [], [], []
    for qid in of_queries.get(of_s, []):
        labels = qlabels.get(qid, {})
        if not labels: continue
        cands = list(labels.keys()); n_c = len(cands)
        feats = np.zeros((n_c, 7), dtype=np.float32)
        ca_pre = np.zeros(n_c, dtype=np.float32)
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
            feats[idx] = [
                morgan_v, bit_c, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                abs(cp[4]-op[4])/max(op[4]+1,1)
            ]
            pc = 1.0/(1.0 + abs(cp[0]-op[0]) + abs(cp[1]-op[1]) +
                     abs(cp[2]-op[2])/max(op[2],1) + abs(cp[3]-op[3])/max(abs(op[3])+1,1))
            ca_pre[idx] = 0.75*morgan_v + 0.25*pc
            cil.append(cand_list.index(c) if c in cand_list else -1)
        Xl.append(feats); yl.append(np.array([labels[c] for c in cands], dtype=np.int8))
        qinfo.append((qid, n_c, cil, ca_pre))
    if Xl:
        of_data[of_s] = {"X7": np.vstack(Xl), "y": np.concatenate(yl), "queries": qinfo}
log(f"  {sum(len(v['X7']) for v in of_data.values()):,} pairs")

def hit10(labels_dict, scores, candidates):
    order = np.argsort(-scores)
    ranked = np.array([labels_dict[c] for c in [candidates[i] for i in order]])
    pos = np.where(ranked == 1)[0]
    return int(len(pos) > 0 and pos[0] < 10)

# 3-seed evaluation
log("=== COMPONENT LADDER (3 seeds) ===")
results = []

for seed_idx in range(3):
    rng = np.random.RandomState(SEED + seed_idx)
    shuffled = list(labeled_ofs); rng.shuffle(shuffled)
    n_tr = int(len(shuffled) * 0.7)
    train_ofs, test_ofs = shuffled[:n_tr], shuffled[n_tr:]

    # Freq from train
    freq_map = {}
    for o in train_ofs:
        y = of_data[o]["y"]; queries = of_data[o]["queries"]
        off = 0
        for _, nc, cil, _ in queries:
            for j in range(nc):
                if y[off+j] == 1:
                    ci = cil[j]
                    if ci >= 0: freq_map[cand_list[ci]] = freq_map.get(cand_list[ci], 0) + 1
            off += nc
    max_f = max(freq_map.values()) if freq_map else 1

    # Train LBC
    X7_tr = np.vstack([of_data[o]["X7"] for o in train_ofs])
    y_tr = np.concatenate([of_data[o]["y"] for o in train_ofs])
    X_tr = np.zeros((len(X7_tr), 8), dtype=np.float32); X_tr[:, :7] = X7_tr
    off = 0
    for o in train_ofs:
        for _, nc, cil, _ in of_data[o]["queries"]:
            for j in range(nc):
                ci = cil[j]; c = cand_list[ci] if ci >= 0 else ""
                X_tr[off+j, 7] = freq_map.get(c, 0) / max(max_f, 1)
            off += nc
    scaler = StandardScaler(); X_s = scaler.fit_transform(X_tr)
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr.fit(X_s, y_tr)

    # Evaluate on test OFs
    sh10 = {"Freq": [], "CA": [], "Freq+CA": [], "LBC": []}
    for o in test_ofs:
        queries = of_data[o]["queries"]; yt = of_data[o]["y"]; X7 = of_data[o]["X7"]
        off = 0
        for _, nc, cil, ca_pre in queries:
            cands = [cand_list[ci] if ci >= 0 else "" for ci in cil]
            ld = dict(zip(cands, yt[off:off+nc].astype(int)))
            if sum(ld.values()) == 0:
                off += nc; continue
            fs = np.array([freq_map.get(c, 0) / max(max_f, 1) for c in cands])
            sh10["Freq"].append(hit10(ld, fs, cands))
            sh10["CA"].append(hit10(ld, ca_pre, cands))
            sh10["Freq+CA"].append(hit10(ld, 0.5*fs + 0.5*ca_pre, cands))
            Xq = np.zeros((nc, 8), dtype=np.float32)
            Xq[:, :7] = X7[off:off+nc]; Xq[:, 7] = fs
            sh10["LBC"].append(hit10(ld, lr.predict_proba(scaler.transform(Xq))[:, 1], cands))
            off += nc

    row = {"seed": seed_idx}
    for k in ["Freq", "CA", "Freq+CA", "LBC"]:
        row[k] = float(np.mean(sh10[k])) if sh10[k] else 0.0
    row["Delta_LBC_vs_Blend"] = row["LBC"] - row["Freq+CA"]
    row["Delta_Blend_vs_CA"] = row["Freq+CA"] - row["CA"]
    results.append(row)
    log(f"  S{seed_idx}: Freq={row['Freq']:.4f} CA={row['CA']:.4f} "
        f"Blend={row['Freq+CA']:.4f} LBC={row['LBC']:.4f} "
        f"LBC-Blend={row['Delta_LBC_vs_Blend']:+.4f} Blend-CA={row['Delta_Blend_vs_CA']:+.4f}")

df = pd.DataFrame(results)
print(f"\n=== RESULTS ===")
print(df.to_string(index=False))
print(f"\nLBC vs Blend mean Delta = {df['Delta_LBC_vs_Blend'].mean():+.4f}")
print(f"Blend vs CA mean Delta = {df['Delta_Blend_vs_CA'].mean():+.4f}")
df.to_csv(OUT / "component_ladder.csv", index=False)
log("Done")
