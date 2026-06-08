#!/usr/bin/env python3
"""Component Ladder: 10-seed with inner 3-fold CV for CA(lam) and Blend(alpha)."""
import json, glob, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

SEED = 20260603
OUT = Path("technique_transfer/results")
OUT.mkdir(parents=True, exist_ok=True)
RDLogger.DisableLog("rdApp.*")

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Load data ──
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

def hit10(ld, s, c):
    o = np.argsort(-s); r = np.array([ld[c[i]] for i in o])
    p = np.where(r == 1)[0]
    return int(len(p) > 0 and p[0] < 10)

log("Loading...")
manifest = {}
mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
with mp.open(encoding="utf-8") as f:
    for line in f:
        if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q

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

# Precompute LBC features
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
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1), abs(cp[4]-op[4])/max(op[4]+1,1)]
            cil.append(cand_list.index(c) if c in cand_list else -1)
        Xl.append(feats); yl.append(np.array([labels[c] for c in cands], dtype=np.int8)); qinfo.append((qid, n_c, cil))
    if Xl: of_data[of_s] = {"X7": np.vstack(Xl), "y": np.concatenate(yl), "queries": qinfo}
log(f"  {sum(len(v['X7']) for v in of_data.values()):,} pairs")

# ── 10-seed component ladder ──
log("=== 10-SEED COMPONENT LADDER ===")
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
results = []

for seed_idx in range(10):
    seed_t0 = time.time()
    rng = np.random.RandomState(SEED + seed_idx)
    shuffled = list(labeled_ofs); rng.shuffle(shuffled)
    n_tr = int(len(shuffled) * 0.7)
    train_ofs, test_ofs = shuffled[:n_tr], shuffled[n_tr:]

    # Train-frequency
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

    # Inner 3-fold CV to tune CA(lam) and Blend(alpha)
    inner_ofs = np.array(train_ofs)
    gkf = GroupKFold(n_splits=3)
    lam_scores = {lam: [] for lam in LAMS}
    blend_scores = {}  # (lam, alpha) -> [h10]

    for it_i, iv_i in gkf.split(inner_ofs, groups=np.arange(len(train_ofs))):
        it_o = list(inner_ofs[it_i]); iv_o = list(inner_ofs[iv_i])
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

        for o in iv_o:
            queries = of_data[o]["queries"]; yt = of_data[o]["y"]; X7 = of_data[o]["X7"]
            off2 = 0
            for _, nc, cil in queries:
                cands = [cand_list[ci] if ci >= 0 else "" for ci in cil]
                ld = dict(zip(cands, yt[off2:off2+nc].astype(int)))
                if sum(ld.values()) == 0:
                    off2 += nc; continue

                fs = np.array([ifreq.get(c, 0) / max(imf, 1) for c in cands])
                morgan_vec = X7[off2:off2+nc, 0]
                # Physchem score: 1/(1+dHeavy+dRings+dMW_norm+dLogP_norm)
                pc_vec = 1.0 / (1.0 + X7[off2:off2+nc, 2] + X7[off2:off2+nc, 3] +
                                X7[off2:off2+nc, 4] + X7[off2:off2+nc, 5])

                for lam in LAMS:
                    ca_s = lam * morgan_vec + (1 - lam) * pc_vec
                    lam_scores[lam].append(hit10(ld, ca_s, cands))
                    for alpha in ALPHAS:
                        blend_s = alpha * fs + (1 - alpha) * ca_s
                        key = (lam, alpha)
                        if key not in blend_scores: blend_scores[key] = []
                        blend_scores[key].append(hit10(ld, blend_s, cands))
                off2 += nc

    # Select best lam and (lam, alpha)
    best_lam = max(LAMS, key=lambda x: np.mean(lam_scores[x]) if lam_scores[x] else 0)
    best_blend_key = max(blend_scores, key=lambda k: np.mean(blend_scores[k]) if blend_scores[k] else 0)
    best_blend_lam, best_blend_alpha = best_blend_key

    # Train LBC (same as baseline)
    X7_tr = np.vstack([of_data[o]["X7"] for o in train_ofs])
    y_tr = np.concatenate([of_data[o]["y"] for o in train_ofs])
    X_tr = np.zeros((len(X7_tr), 8), dtype=np.float32); X_tr[:, :7] = X7_tr
    off = 0
    for o in train_ofs:
        for _, nc, cil in of_data[o]["queries"]:
            for j in range(nc):
                ci = cil[j]; c = cand_list[ci] if ci >= 0 else ""
                X_tr[off+j, 7] = freq_map.get(c, 0) / max(max_f, 1)
            off += nc
    scaler = StandardScaler(); X_s = scaler.fit_transform(X_tr)
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    lr.fit(X_s, y_tr)

    # Evaluate on test — compute BOTH macro and query-weighted
    per_of_h10 = {k: defaultdict(list) for k in ["Freq", "CA", "Freq+CA", "LBC"]}
    for o in test_ofs:
        queries = of_data[o]["queries"]; yt = of_data[o]["y"]; X7 = of_data[o]["X7"]
        off = 0
        for _, nc, cil in queries:
            cands = [cand_list[ci] if ci >= 0 else "" for ci in cil]
            ld = dict(zip(cands, yt[off:off+nc].astype(int)))
            if sum(ld.values()) == 0:
                off += nc; continue
            fs = np.array([freq_map.get(c, 0) / max(max_f, 1) for c in cands])
            per_of_h10["Freq"][o].append(hit10(ld, fs, cands))
            ca_s = best_lam * X7[off:off+nc, 0] + (1 - best_lam) * (1.0 / (1.0 + X7[off:off+nc, 2] + X7[off:off+nc, 3] + X7[off:off+nc, 4] + X7[off:off+nc, 5]))
            per_of_h10["CA"][o].append(hit10(ld, ca_s, cands))
            blend_s = best_blend_alpha * fs + (1 - best_blend_alpha) * ca_s
            per_of_h10["Freq+CA"][o].append(hit10(ld, blend_s, cands))
            Xq = np.zeros((nc, 8), dtype=np.float32); Xq[:, :7] = X7[off:off+nc]; Xq[:, 7] = fs
            per_of_h10["LBC"][o].append(hit10(ld, lr.predict_proba(scaler.transform(Xq))[:, 1], cands))
            off += nc

    row = {
        "seed": seed_idx, "best_lam": best_lam, "best_blend_alpha": best_blend_alpha,
        "best_blend_lam": best_blend_lam,
    }
    for k in ["Freq", "CA", "Freq+CA", "LBC"]:
        # OF-macro: mean of per-OF means
        of_means = [float(np.mean(v)) for v in per_of_h10[k].values() if v]
        row[f"{k}_macro"] = float(np.mean(of_means)) if of_means else 0.0
        # Query-weighted: flat average across all queries
        all_q = [x for v in per_of_h10[k].values() for x in v]
        row[f"{k}_query"] = float(np.mean(all_q)) if all_q else 0.0
    row["Delta_Blend_vs_CA_macro"] = row["Freq+CA_macro"] - row["CA_macro"]
    row["Delta_LBC_vs_Blend_macro"] = row["LBC_macro"] - row["Freq+CA_macro"]
    row["Delta_Blend_vs_CA_query"] = row["Freq+CA_query"] - row["CA_query"]
    row["Delta_LBC_vs_Blend_query"] = row["LBC_query"] - row["Freq+CA_query"]
    results.append(row)
    log(f"  S{seed_idx}: macro Freq={row['Freq_macro']:.4f} CA={row['CA_macro']:.4f} Blend={row['Freq+CA_macro']:.4f} LBC={row['LBC_macro']:.4f} Bl-CA={row['Delta_Blend_vs_CA_macro']:+.4f} LBC-Bl={row['Delta_LBC_vs_Blend_macro']:+.4f} ({time.time()-seed_t0:.0f}s)")

df = pd.DataFrame(results)
print(f"\n=== 10-SEED COMPONENT LADDER (MACRO) ===")
macro_cols = [c for c in df.columns if c.endswith("_macro") or c in ["seed", "best_lam", "best_blend_alpha"]]
print(df[macro_cols].to_string(index=False))
print(f"\nMACRO: Freq={df['Freq_macro'].mean():.4f} CA={df['CA_macro'].mean():.4f} Blend={df['Freq+CA_macro'].mean():.4f} LBC={df['LBC_macro'].mean():.4f}")
print(f"MACRO Blend-CA={df['Delta_Blend_vs_CA_macro'].mean():+.4f} LBC-Blend={df['Delta_LBC_vs_Blend_macro'].mean():+.4f}")
print(f"\nQUERY-WT: Freq={df['Freq_query'].mean():.4f} CA={df['CA_query'].mean():.4f} Blend={df['Freq+CA_query'].mean():.4f} LBC={df['LBC_query'].mean():.4f}")
print(f"QUERY Blend-CA={df['Delta_Blend_vs_CA_query'].mean():+.4f} LBC-Blend={df['Delta_LBC_vs_Blend_query'].mean():+.4f}")
df.to_csv(OUT / "component_ladder_10seed.csv", index=False)
log("Done")
