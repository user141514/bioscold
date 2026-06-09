#!/usr/bin/env python3
"""Freq engineering: baseline vs log-freq vs tfidf-freq (3 seeds)."""
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

log("Loading...")
manifest = {}
mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
with mp.open(encoding="utf-8") as f:
    for line in f:
        if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q
def mfp(s):
    m = Chem.MolFromSmiles(s);
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
log(f"  {len(labeled_ofs)} OFs, {len(cand_list)} candidates")

# Precompute 7 static features
of_data = {}
for of_s in labeled_ofs:
    ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
    Xl, yl, qinfo = [], [], []
    for qid in of_queries.get(of_s, []):
        labels = qlabels.get(qid, {})
        if not labels: continue
        cands = list(labels.keys()); n_c = len(cands)
        feats = np.zeros((n_c, 7), dtype=np.float32)
        cil = []
        for idx, c in enumerate(cands):
            cfp = cand_fps.get(c); cp = cand_props.get(c)
            if cfp is None or cp is None: continue
            inter = float(cfp.dot(ofp)); denom = float(cfp.sum() + ofp.sum() - inter)
            feats[idx, 0] = inter / max(denom, 0.001)
            cfp_n = cfp / (cfp.sum() + 1e-10)
            if cfp.sum() > 0 and ofp.sum() > 0:
                corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                feats[idx, 1] = corr if np.isfinite(corr) else 0.0
            feats[idx, 2:] = [abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                abs(cp[4]-op[4])/max(op[4]+1,1)]
            cil.append(cand_list.index(c) if c in cand_list else -1)
        Xl.append(feats); yl.append(np.array([labels[c] for c in cands], dtype=np.int8)); qinfo.append((qid, n_c, cil))
    if Xl: of_data[of_s] = {"X7": np.vstack(Xl), "y": np.concatenate(yl), "queries": qinfo}
log(f"  {sum(len(v['X7']) for v in of_data.values()):,} pairs")

def hit10(ld, s, c):
    o = np.argsort(-s); r = np.array([ld[c[i]] for i in o])
    p = np.where(r == 1)[0]; return int(len(p) > 0 and p[0] < 10)

# ── 3-seed comparison ──
log("=== FREQ ENGINEERING (3 seeds) ===")
results = []

for si in range(3):
    rng = np.random.RandomState(SEED + si); sh = list(labeled_ofs); rng.shuffle(sh)
    n_tr = int(len(sh)*0.7); tr_ofs, te_ofs = sh[:n_tr], sh[n_tr:]

    # Raw frequency counts
    freq_raw = {}
    for o in tr_ofs:
        y = of_data[o]["y"]; queries = of_data[o]["queries"]; off = 0
        for _, nc, cil in queries:
            for j in range(nc):
                if y[off+j] == 1:
                    ci = cil[j]
                    if ci >= 0: freq_raw[cand_list[ci]] = freq_raw.get(cand_list[ci], 0) + 1
            off += nc
    max_raw = max(freq_raw.values()) if freq_raw else 1
    n_of = len(tr_ofs)

    # Compute how many training OFs each candidate appears in
    cand_of_count = defaultdict(int)
    for o in tr_ofs:
        seen = set()
        y = of_data[o]["y"]; queries = of_data[o]["queries"]; off = 0
        for _, nc, cil in queries:
            for j in range(nc):
                if y[off+j] == 1:
                    ci = cil[j]
                    if ci >= 0 and cand_list[ci] not in seen:
                        cand_of_count[cand_list[ci]] += 1
                        seen.add(cand_list[ci])
            off += nc

    # Three freq variants
    def freq_baseline(c):
        return freq_raw.get(c, 0) / max(max_raw, 1)
    def freq_log(c):
        return np.log(1 + freq_raw.get(c, 0)) / np.log(1 + max_raw)
    def freq_tfidf(c):
        raw = freq_raw.get(c, 0)
        n_docs = cand_of_count.get(c, 1)
        return (raw / max(max_raw, 1)) * np.log(n_of / (1 + n_docs))

    # Build training matrix
    X7_tr = np.vstack([of_data[o]["X7"] for o in tr_ofs])
    y_tr = np.concatenate([of_data[o]["y"] for o in tr_ofs])
    n_rows = len(y_tr)

    # Pre-build candidate lists for each row
    tr_all_cands = []
    off = 0
    for o in tr_ofs:
        for _, nc, cil in of_data[o]["queries"]:
            for j in range(nc):
                ci = cil[j]; c = cand_list[ci] if ci >= 0 else ""
                tr_all_cands.append(c)
            off += nc

    def build_X(freq_fn):
        X = np.zeros((n_rows, 8), dtype=np.float32)
        X[:, :7] = X7_tr
        for j, c in enumerate(tr_all_cands):
            X[j, 7] = freq_fn(c)
        return X

    sc_base = StandardScaler(); sc_log = StandardScaler(); sc_tfidf = StandardScaler()
    lr_base = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr_log = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr_tfidf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)

    lr_base.fit(sc_base.fit_transform(build_X(freq_baseline)), y_tr)
    lr_log.fit(sc_log.fit_transform(build_X(freq_log)), y_tr)
    lr_tfidf.fit(sc_tfidf.fit_transform(build_X(freq_tfidf)), y_tr)

    # Evaluate
    sh10 = {"base": [], "log": [], "tfidf": []}
    for o in te_ofs:
        queries = of_data[o]["queries"]; yt = of_data[o]["y"]; X7 = of_data[o]["X7"]
        off = 0
        for _, nc, cil in queries:
            cands = [cand_list[ci] if ci >= 0 else "" for ci in cil]
            ld = dict(zip(cands, yt[off:off+nc].astype(int)))
            if sum(ld.values()) == 0: off += nc; continue
            Xq = np.zeros((nc, 8), dtype=np.float32); Xq[:, :7] = X7[off:off+nc]
            Xq_b = Xq.copy(); Xq_l = Xq.copy(); Xq_t = Xq.copy()
            Xq_b[:, 7] = [freq_baseline(c) for c in cands]
            Xq_l[:, 7] = [freq_log(c) for c in cands]
            Xq_t[:, 7] = [freq_tfidf(c) for c in cands]
            sh10["base"].append(hit10(ld, lr_base.predict_proba(sc_base.transform(Xq_b))[:, 1], cands))
            sh10["log"].append(hit10(ld, lr_log.predict_proba(sc_log.transform(Xq_l))[:, 1], cands))
            sh10["tfidf"].append(hit10(ld, lr_tfidf.predict_proba(sc_tfidf.transform(Xq_t))[:, 1], cands))
            off += nc

    row = {"seed": si}
    for k in ["base", "log", "tfidf"]:
        row[k] = float(np.mean(sh10[k])) if sh10[k] else 0.0
    row["delta_log"] = row["log"] - row["base"]
    row["delta_tfidf"] = row["tfidf"] - row["base"]
    results.append(row)
    log(f"  S{si}: base={row['base']:.4f} log={row['log']:.4f} (Δ={row['delta_log']:+.4f}) tfidf={row['tfidf']:.4f} (Δ={row['delta_tfidf']:+.4f})")

df = pd.DataFrame(results)
print(f"\n=== FREQ ENGINEERING (3 seeds, query-weighted) ===")
print(df.to_string(index=False))
print(f"base={df['base'].mean():.4f} log={df['log'].mean():.4f} (Δ={df['delta_log'].mean():+.4f}) tfidf={df['tfidf'].mean():.4f} (Δ={df['delta_tfidf'].mean():+.4f})")
df.to_csv(OUT / "freq_engineering.csv", index=False)
log("Done")
