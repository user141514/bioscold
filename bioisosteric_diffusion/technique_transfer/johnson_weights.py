#!/usr/bin/env python3
"""Johnson Relative Weights: decompose R² into fair per-feature contributions."""
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

FEAT_NAMES = ["Morgan", "bit_corr", "dHeavy", "dRings", "dMW", "dLogP", "dTPSA", "freq"]

log("Loading...")
manifest = {}
mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
with mp.open(encoding="utf-8") as f:
    for line in f:
        if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q
def mfp(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    fp = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), fp); return fp
def mprops(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    return np.array([m.GetNumHeavyAtoms(), Descriptors.RingCount(m),
        Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m)], dtype=np.float32)
all_ofs = sorted({q["old_fragment_smiles"] for q in manifest.values()})
of_fps, of_props = {}, {}
for s in all_ofs:
    fp_ = mfp(s); pr_ = mprops(s)
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
        fp_ = mfp(c); pr_ = mprops(c)
        if fp_ is not None and pr_ is not None: cand_fps[c] = fp_; cand_props[c] = pr_
cand_list = sorted(cand_fps.keys())
of_queries = defaultdict(list)
for q in manifest.values():
    if q["old_fragment_smiles"] in of_fps: of_queries[q["old_fragment_smiles"]].append(q["query_id"])
labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in qlabels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})
log(f"  {len(labeled_ofs)} OFs")

# ── Per-seed Johnson weights ──
jh_rows = []
for si in range(10):
    rng = np.random.RandomState(SEED + si)
    sh = list(labeled_ofs); rng.shuffle(sh)
    n_tr = int(len(sh)*0.7); tr_ofs = sh[:n_tr]

    freq_map = {}
    for o in tr_ofs:
        for qid in of_queries.get(o, []):
            for c, lbl in qlabels.get(qid, {}).items():
                if lbl == 1: freq_map[c] = freq_map.get(c, 0) + 1
    mf = max(freq_map.values()) if freq_map else 1

    # Build training matrix
    X, y = [], []
    for o in tr_ofs:
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
                    abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                    abs(cp[4]-op[4])/max(op[4]+1,1), freq_map.get(c, 0)/max(mf, 1)])
                y.append(lbl)
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int8)
    sc = StandardScaler(); X_s = sc.fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED); lr.fit(X_s, y)

    # Johnson relative weights: RW_j = Σ_k β_k * corr(X_j, X_k)
    # where β are standardized coefficients and corr is the predictor correlation
    beta_std = lr.coef_[0]  # standardized (features are z-scored)
    R = np.corrcoef(X_s[:, :8].T)  # 8x8 correlation matrix
    R2_pseudo = 1 - np.mean((y - lr.predict_proba(X_s)[:, 1])**2) / np.var(y)

    johnson = np.zeros(8)
    for j in range(8):
        johnson[j] = sum(beta_std[k] * R[j, k] for k in range(8))
    # Normalize to sum to R²
    johnson = johnson / johnson.sum() * R2_pseudo

    row = {"seed": si, "R2": R2_pseudo}
    for j, name in enumerate(FEAT_NAMES):
        row[name] = johnson[j]
    jh_rows.append(row)

df = pd.DataFrame(jh_rows)
print(f"\n=== Johnson Relative Weights (10 seeds) ===")
print(f"Mean R2 = {df['R2'].mean():.4f}")
print(f"\n{'Feature':12s} {'Mean RW':>8s} {'Std':>8s} {'Ablation Δ':>10s}")
for j, name in enumerate(FEAT_NAMES):
    abl_delta = {"Morgan": -0.109, "bit_corr": -0.155, "dHeavy": -0.070, "dRings": -0.079,
                 "dMW": -0.077, "dLogP": -0.091, "dTPSA": -0.085, "freq": -0.193}[name]
    print(f"{name:12s} {df[name].mean():8.4f} {df[name].std():8.4f} {abl_delta:10.3f}")

# Sum check
physchem_sum = sum(df[n].mean() for n in ["dHeavy", "dRings", "dMW", "dLogP", "dTPSA"])
print(f"\nPhyschem combined: {physchem_sum:.4f}")
print(f"Sum check: {sum(df[n].mean() for n in FEAT_NAMES):.4f} (should ~ R2={df['R2'].mean():.4f})")

df.to_csv(OUT / "johnson_weights.csv", index=False)
log("Done")
