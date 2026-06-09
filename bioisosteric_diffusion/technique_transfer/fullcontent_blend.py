#!/usr/bin/env python3
"""Full-content Blend: Morgan+bit_corr+5 physchem → scalar, then blend with freq."""
import json, glob, time, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

SEED = 20260603; OUT = Path("technique_transfer/results")
OUT.mkdir(parents=True, exist_ok=True); RDLogger.DisableLog("rdApp.*")
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Data loading ──
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
    fp = mfp(s); props = mprops(s)
    if fp is not None and props is not None: of_fps[s] = fp; of_props[s] = props
of_queries = defaultdict(list)
for q in manifest.values():
    if q["old_fragment_smiles"] in of_fps: of_queries[q["old_fragment_smiles"]].append(q["query_id"])
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
labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in qlabels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})
log(f"  {len(labeled_ofs)} OFs, {len(cand_list)} candidates")

# Precompute 7 content features per pair (Morgan + bit_corr + 5 physchem)
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
            morgan_v = inter / max(denom, 0.001)
            cfp_n = cfp / (cfp.sum() + 1e-10); bit_c = 0.0
            if cfp.sum() > 0 and ofp.sum() > 0:
                corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                bit_c = corr if np.isfinite(corr) else 0.0
            feats[idx] = [morgan_v, bit_c, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                abs(cp[4]-op[4])/max(op[4]+1,1)]
            cil.append(cand_list.index(c) if c in cand_list else -1)
        Xl.append(feats); yl.append(np.array([labels[c] for c in cands], dtype=np.int8)); qinfo.append((qid, n_c, cil))
    if Xl: of_data[of_s] = {"X7c": np.vstack(Xl), "y": np.concatenate(yl), "queries": qinfo}
log(f"  {sum(len(v['X7c']) for v in of_data.values()):,} pairs")

def hit10(ld, s, c):
    o = np.argsort(-s); r = np.array([ld[c[i]] for i in o])
    p = np.where(r == 1)[0]; return int(len(p) > 0 and p[0] < 10)

# ── 10-seed FullBlend vs LBC ──
log("=== FullBlend vs LBC (10-seed) ===")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
results = []

for si in range(10):
    rng = np.random.RandomState(SEED + si)
    sh = list(labeled_ofs); rng.shuffle(sh)
    n_tr = int(len(sh)*0.7); tr_ofs, te_ofs = sh[:n_tr], sh[n_tr:]

    # Train frequency
    freq = {}
    for o in tr_ofs:
        y = of_data[o]["y"]; queries = of_data[o]["queries"]; off = 0
        for _, nc, cil in queries:
            for j in range(nc):
                if y[off+j] == 1:
                    ci = cil[j]
                    if ci >= 0: freq[cand_list[ci]] = freq.get(cand_list[ci], 0) + 1
            off += nc
    mf = max(freq.values()) if freq else 1

    # Train LBC (8 features)
    X7_tr = np.vstack([of_data[o]["X7c"] for o in tr_ofs])
    y_tr = np.concatenate([of_data[o]["y"] for o in tr_ofs])
    X_lbc = np.zeros((len(X7_tr), 8), dtype=np.float32); X_lbc[:, :7] = X7_tr
    off = 0
    for o in tr_ofs:
        for _, nc, cil in of_data[o]["queries"]:
            for j in range(nc):
                ci = cil[j]; c = cand_list[ci] if ci >= 0 else ""
                X_lbc[off+j, 7] = freq.get(c, 0) / max(mf, 1)
            off += nc
    sc_lbc = StandardScaler(); lr_lbc = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr_lbc.fit(sc_lbc.fit_transform(X_lbc), y_tr)

    # Inner 3-fold CV: train FullContent (Ridge) on 7 content features + tune blend α
    inner_ofs = np.array(tr_ofs); gkf = GroupKFold(n_splits=3)
    blend_scores = {b: [] for b in ALPHAS}

    for it_i, iv_i in gkf.split(inner_ofs, groups=np.arange(len(tr_ofs))):
        it_o = list(inner_ofs[it_i]); iv_o = list(inner_ofs[iv_i])
        # Inner freq
        ifreq = {}
        for o in it_o:
            yt = of_data[o]["y"]; queries = of_data[o]["queries"]; off2 = 0
            for _, nc, cil in queries:
                for j in range(nc):
                    if yt[off2+j] == 1:
                        ci = cil[j]
                        if ci >= 0: ifreq[cand_list[ci]] = ifreq.get(cand_list[ci], 0) + 1
                off2 += nc
        imf = max(ifreq.values()) if ifreq else 1

        # Train FullContent scalar (Ridge regression on 7 content features)
        X7c_it = np.vstack([of_data[o]["X7c"] for o in it_o])
        y_it = np.concatenate([of_data[o]["y"] for o in it_o])
        sc_content = StandardScaler()
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(sc_content.fit_transform(X7c_it), y_it)

        # Evaluate on inner val
        for o in iv_o:
            queries = of_data[o]["queries"]; yt = of_data[o]["y"]; X7c = of_data[o]["X7c"]
            off2 = 0
            for _, nc, cil in queries:
                cands = [cand_list[ci] if ci >= 0 else "" for ci in cil]
                ld = dict(zip(cands, yt[off2:off2+nc].astype(int)))
                if sum(ld.values()) == 0: off2 += nc; continue
                content_s = ridge.predict(sc_content.transform(X7c[off2:off2+nc]))
                fs = np.array([ifreq.get(c, 0) / max(imf, 1) for c in cands])
                z_content = (content_s - content_s.mean()) / (content_s.std() + 1e-10)
                z_freq = (fs - fs.mean()) / (fs.std() + 1e-10)
                for b in ALPHAS:
                    blend_s = b * z_freq + (1 - b) * z_content
                    blend_scores[b].append(hit10(ld, blend_s, cands))
                off2 += nc

    best_alpha = max(ALPHAS, key=lambda b: np.mean(blend_scores[b]) if blend_scores[b] else 0)

    # Retrain FullContent on full train
    X7c_tr = np.vstack([of_data[o]["X7c"] for o in tr_ofs])
    sc_full = StandardScaler()
    ridge_full = Ridge(alpha=1.0, random_state=SEED)
    ridge_full.fit(sc_full.fit_transform(X7c_tr), y_tr)

    # Evaluate on test — compute BOTH macro and query-weighted
    per_of = {"LBC": defaultdict(list), "FullBlend": defaultdict(list)}
    for o in te_ofs:
        queries = of_data[o]["queries"]; yt = of_data[o]["y"]; X7c = of_data[o]["X7c"]
        off = 0
        for _, nc, cil in queries:
            cands = [cand_list[ci] if ci >= 0 else "" for ci in cil]
            ld = dict(zip(cands, yt[off:off+nc].astype(int)))
            if sum(ld.values()) == 0: off += nc; continue
            fs = np.array([freq.get(c, 0) / max(mf, 1) for c in cands])
            Xq = np.zeros((nc, 8), dtype=np.float32); Xq[:, :7] = X7c[off:off+nc]; Xq[:, 7] = fs
            per_of["LBC"][o].append(hit10(ld, lr_lbc.predict_proba(sc_lbc.transform(Xq))[:, 1], cands))
            content_s = ridge_full.predict(sc_full.transform(X7c[off:off+nc]))
            zc = (content_s - content_s.mean()) / (content_s.std() + 1e-10)
            zf = (fs - fs.mean()) / (fs.std() + 1e-10)
            per_of["FullBlend"][o].append(hit10(ld, best_alpha * zf + (1 - best_alpha) * zc, cands))
            off += nc

    row = {"seed": si, "best_alpha": best_alpha}
    for k in ["LBC", "FullBlend"]:
        # MACRO: mean of per-OF means
        of_means = [float(np.mean(v)) for v in per_of[k].values() if v]
        row[f"{k}_macro"] = float(np.mean(of_means)) if of_means else 0.0
        # QUERY: flat average
        all_q = [x for v in per_of[k].values() for x in v]
        row[f"{k}_query"] = float(np.mean(all_q)) if all_q else 0.0
    row["delta_macro"] = row["LBC_macro"] - row["FullBlend_macro"]
    row["delta_query"] = row["LBC_query"] - row["FullBlend_query"]
    results.append(row)
    log(f"  S{si}: mac LBC={row['LBC_macro']:.4f} FB={row['FullBlend_macro']:.4f} Δ={row['delta_macro']:+.4f} | qry LBC={row['LBC_query']:.4f} FB={row['FullBlend_query']:.4f}")

df = pd.DataFrame(results)
print(f"\n=== FullBlend vs LBC (10-seed) ===")
macro_cols = [c for c in df.columns if "macro" in c or c in ["seed", "best_alpha"]]
print("MACRO:")
print(df[macro_cols].to_string(index=False))
q_cols = [c for c in df.columns if "query" in c or c in ["seed"]]
print("\nQUERY:")
print(df[q_cols].to_string(index=False))
print(f"\nMACRO LBC={df['LBC_macro'].mean():.4f} FullBlend={df['FullBlend_macro'].mean():.4f} Δ={df['delta_macro'].mean():+.4f} ± {df['delta_macro'].std():.4f}")
print(f"QUERY LBC={df['LBC_query'].mean():.4f} FullBlend={df['FullBlend_query'].mean():.4f} Δ={df['delta_query'].mean():+.4f} ± {df['delta_query'].std():.4f}")
df.to_csv(OUT / "fullblend_vs_lbc.csv", index=False)
log("Done")
