#!/usr/bin/env python3
"""PCA-based feature transformation: Raw vs PC scores + freq."""
import json, glob, numpy as np
from collections import defaultdict
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

RDLogger.DisableLog("rdApp.*")
OUT = Path("technique_transfer/results"); OUT.mkdir(parents=True, exist_ok=True)

def log(msg): print(msg, flush=True)

manifest = {}
mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
with mp.open(encoding="utf-8") as f:
    for line in f:
        if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q
all_ofs = sorted({q["old_fragment_smiles"] for q in manifest.values()})
def fp(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    f = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), f); return f
def pr(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    return np.array([m.GetNumHeavyAtoms(), Descriptors.RingCount(m),
        Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m)], dtype=np.float32)
of_fps, of_props = {}, {}
for s in all_ofs:
    fp_ = fp(s); pr_ = pr(s)
    if fp_ is not None and pr_ is not None: of_fps[s] = fp_; of_props[s] = pr_
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
        fp_ = fp(c); pr_ = pr(c)
        if fp_ is not None and pr_ is not None: cand_fps[c] = fp_; cand_props[c] = pr_
labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in qlabels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})
log(f"  {len(labeled_ofs)} OFs")

of_queries = defaultdict(list)
for q in manifest.values():
    if q["old_fragment_smiles"] in of_fps: of_queries[q["old_fragment_smiles"]].append(q["query_id"])

# Fit PCA on all content pairs (sampled)
X_pca = []
for o in labeled_ofs:
    ofp = of_fps[o]; op = of_props[o]; ofp_n = ofp / (ofp.sum() + 1e-10)
    for qid in of_queries.get(o, [])[:20]:
        labels = qlabels.get(qid, {})
        for c in list(labels.keys())[:10]:
            cfp = cand_fps.get(c); cp = cand_props.get(c)
            if cfp is None: continue
            inter = float(cfp.dot(ofp)); denom = float(cfp.sum() + ofp.sum() - inter)
            m = inter / max(denom, 0.001)
            cfp_n = cfp / (cfp.sum() + 1e-10); bc = 0.0
            if cfp.sum() > 0 and ofp.sum() > 0:
                corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                bc = corr if np.isfinite(corr) else 0.0
            X_pca.append([m, bc, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                abs(cp[2]-op[2])/max(op[2], 1), abs(cp[3]-op[3])/max(abs(op[3])+1, 1),
                abs(cp[4]-op[4])/max(op[4]+1, 1)])
X_pca = np.array(X_pca, dtype=np.float32)
sc_pca = StandardScaler(); pca_model = PCA(); pca_model.fit(sc_pca.fit_transform(X_pca))
log(f"  PCA on {len(X_pca):,} pairs")
for n in [1, 2, 3, 5, 7]:
    log(f"    PC1-{n}: {pca_model.explained_variance_ratio_[:n].sum():.4f}")

# ── 3-seed comparison ──
def hit10(ld, s, c):
    o = np.argsort(-s); r = np.array([ld[c[i]] for i in o])
    p = np.where(r == 1)[0]; return int(len(p) > 0 and p[0] < 10)

results = []
for si in range(3):
    rng = np.random.RandomState(20260603 + si)
    sh = list(labeled_ofs); rng.shuffle(sh)
    n_tr = int(len(sh) * 0.7); tr_ofs, te_ofs = sh[:n_tr], sh[n_tr:]

    freq_map = {}
    for o in tr_ofs:
        for qid in of_queries.get(o, []):
            for c, lbl in qlabels.get(qid, {}).items():
                if lbl == 1: freq_map[c] = freq_map.get(c, 0) + 1
    mf = max(freq_map.values()) if freq_map else 1

    def build(ofs_list):
        X, y = [], []
        for o in ofs_list:
            ofp = of_fps[o]; op = of_props[o]; ofp_n = ofp / (ofp.sum() + 1e-10)
            for qid in of_queries.get(o, []):
                labels = qlabels.get(qid, {})
                if not labels: continue
                for c, lbl in labels.items():
                    cfp = cand_fps.get(c); cp = cand_props.get(c)
                    if cfp is None: continue
                    inter = float(cfp.dot(ofp)); denom = float(cfp.sum() + ofp.sum() - inter)
                    m = inter / max(denom, 0.001)
                    cfp_n = cfp / (cfp.sum() + 1e-10); bc = 0.0
                    if cfp.sum() > 0 and ofp.sum() > 0:
                        corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                        bc = corr if np.isfinite(corr) else 0.0
                    X.append([m, bc, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                        abs(cp[2]-op[2])/max(op[2], 1), abs(cp[3]-op[3])/max(abs(op[3])+1, 1),
                        abs(cp[4]-op[4])/max(op[4]+1, 1), freq_map.get(c, 0)/max(mf, 1)])
                    y.append(lbl)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)

    Xtr, ytr = build(tr_ofs)
    # PCA on train content
    sc = StandardScaler()
    Xtr_c = sc.fit_transform(Xtr[:, :7])
    pca = PCA(); pca.fit(Xtr_c)

    row = {"seed": si}
    for name, n_pc in [("Raw8F", 8), ("PC1+freq", 1), ("PC1-2+freq", 2), ("PC1-3+freq", 3)]:
        if name == "Raw8F":
            Xtr_use = Xtr; X_c = Xtr_c
        else:
            Xtr_pc = np.zeros((len(Xtr), n_pc + 1), dtype=np.float32)
            Xtr_pc[:, :n_pc] = pca.transform(Xtr_c)[:, :n_pc]
            Xtr_pc[:, n_pc] = Xtr[:, 7]
            Xtr_use = Xtr_pc; X_c = Xtr_pc[:, :-1]

        sc2 = StandardScaler()
        lr = LogisticRegression(C=1.0, max_iter=2000, random_state=20260603)
        lr.fit(sc2.fit_transform(Xtr_use), ytr)

        # Evaluate on test
        h10_list = []
        for o in te_ofs:
            ofp = of_fps[o]; op = of_props[o]; ofp_n = ofp / (ofp.sum() + 1e-10)
            for qid in of_queries.get(o, []):
                labels = qlabels.get(qid, {})
                if not labels or sum(labels.values()) == 0: continue
                cands = list(labels.keys()); nc = len(cands)
                Xq = np.zeros((nc, 8), dtype=np.float32)
                for idx, c in enumerate(cands):
                    cfp = cand_fps.get(c); cp = cand_props.get(c)
                    if cfp is None: continue
                    inter = float(cfp.dot(ofp)); denom = float(cfp.sum() + ofp.sum() - inter)
                    m = inter / max(denom, 0.001)
                    cfp_n = cfp / (cfp.sum() + 1e-10); bc = 0.0
                    if cfp.sum() > 0 and ofp.sum() > 0:
                        corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                        bc = corr if np.isfinite(corr) else 0.0
                    Xq[idx] = [m, bc, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                        abs(cp[2]-op[2])/max(op[2], 1), abs(cp[3]-op[3])/max(abs(op[3])+1, 1),
                        abs(cp[4]-op[4])/max(op[4]+1, 1), freq_map.get(c, 0)/max(mf, 1)]
                if name == "Raw8F":
                    Xq_use = Xq
                else:
                    Xq_c = sc.transform(Xq[:, :7])
                    Xq_pc = np.zeros((nc, n_pc + 1), dtype=np.float32)
                    Xq_pc[:, :n_pc] = pca.transform(Xq_c)[:, :n_pc]
                    Xq_pc[:, n_pc] = Xq[:, 7]
                    Xq_use = Xq_pc
                scores = lr.predict_proba(sc2.transform(Xq_use))[:, 1]
                ld = dict(zip(cands, [labels[c] for c in cands]))
                h10_list.append(hit10(ld, scores, cands))
        h10_val = float(np.mean(h10_list)) if h10_list else 0.0
        row[name] = h10_val

    results.append(row)
    log(f"  S{si}: Raw={row['Raw8F']:.4f} PC1+freq={row['PC1+freq']:.4f} PC1-2+freq={row['PC1-2+freq']:.4f} PC1-3+freq={row['PC1-3+freq']:.4f}")

import pandas as pd
df = pd.DataFrame(results)
print(f"\n=== PCA Feature Transform (3 seeds) ===")
print(df.to_string(index=False))
for c in ["Raw8F", "PC1+freq", "PC1-2+freq", "PC1-3+freq"]:
    print(f"  {c}: {df[c].mean():.4f}")
print("Done")
