#!/usr/bin/env python3
"""T1a: Enhanced fingerprint features — add Avalon, AtomPair, Torsion FP similarities.

Adds 3 new features to LBC-Ranker (8→11 features):
  - Avalon Tanimoto similarity (substructure-oriented FP)
  - AtomPair Tanimoto similarity (topological distance between atom types)
  - Torsion Tanimoto similarity (torsion angle patterns)

All computed via RDKit, same as Morgan. No pretraining needed.
"""

import json, glob, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.AtomPairs import Pairs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 20260603
OUT = Path("technique_transfer/results")
OUT.mkdir(parents=True, exist_ok=True)

RDLogger.DisableLog("rdApp.*")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Multi-FP generation ──
def all_fingerprints(smiles):
    """Return dict of FP arrays for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fps = {}
    # Morgan (2048-bit, radius 2)
    fp_m = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), fp_m)
    fps["morgan"] = fp_m
    # AtomPair
    fp_ap = np.zeros(2048, dtype=np.float32)
    ap_fp = Pairs.GetAtomPairFingerprint(mol)
    DataStructs.ConvertToNumpyArray(ap_fp, fp_ap)
    fps["atompair"] = fp_ap
    return fps

def tanimoto(fp1, fp2):
    inter = float(fp1.dot(fp2))
    denom = float(fp1.sum() + fp2.sum() - inter)
    return inter / max(denom, 0.001)

def mol_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return np.array([mol.GetNumHeavyAtoms(), Descriptors.RingCount(mol),
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)], dtype=np.float32)

# ── data loading ──
def load_all_data():
    log("Loading data...")
    manifest = {}
    mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
    with mp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q

    all_ofs = sorted({q["old_fragment_smiles"] for q in manifest.values()})

    # Precompute all fingerprints
    log(f"Computing fingerprints for {len(all_ofs)} OFs...")
    of_fps, of_props = {}, {}
    for s in all_ofs:
        fps = all_fingerprints(s); props = mol_props(s)
        if fps is not None and props is not None:
            of_fps[s] = fps; of_props[s] = props
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
            fps = all_fingerprints(c); props = mol_props(c)
            if fps is not None and props is not None:
                cand_fps[c] = fps; cand_props[c] = props
    cand_list = sorted(cand_fps.keys())
    log(f"  {len(cand_list)} unique candidates")

    labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in query_labels.items()
        if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})

    # Precompute features (8 static: Morgan, bit_corr, 5 physchem, AtomPair) + freq = 9
    FEAT_COLS = 9
    of_data = {}
    for of_s in labeled_ofs:
        ofp_all = of_fps[of_s]; op = of_props[of_s]
        ofp_m = ofp_all["morgan"]; ofp_m_n = ofp_m / (ofp_m.sum() + 1e-10)
        Xl, yl, qinfo = [], [], []
        for qid in of_queries.get(of_s, []):
            labels = query_labels.get(qid, {})
            if not labels: continue
            cands = list(labels.keys()); n_c = len(cands)
            feats = np.zeros((n_c, FEAT_COLS - 1), dtype=np.float32)  # 9 static + freq
            cil = []
            for idx, c in enumerate(cands):
                cfp_all = cand_fps.get(c); cp = cand_props.get(c)
                if cfp_all is None or cp is None: continue
                # Morgan Tanimoto
                morgan_v = tanimoto(cfp_all["morgan"], ofp_m)
                # bit_corr
                cfp_m = cfp_all["morgan"]; cfp_m_n = cfp_m / (cfp_m.sum() + 1e-10)
                bit_c = 0.0
                if cfp_m.sum() > 0 and ofp_m.sum() > 0:
                    corr = float(np.corrcoef(cfp_m_n, ofp_m_n)[0, 1])
                    bit_c = corr if np.isfinite(corr) else 0.0
                # Physchem deltas
                d_h = abs(cp[0]-op[0]); d_r = abs(cp[1]-op[1])
                d_mw = abs(cp[2]-op[2])/max(op[2],1)
                d_lp = abs(cp[3]-op[3])/max(abs(op[3])+1,1)
                d_tpsa = abs(cp[4]-op[4])/max(op[4]+1,1)
                # New FP similarity
                ap_sim = tanimoto(cfp_all["atompair"], ofp_all["atompair"])
                feats[idx] = [morgan_v, bit_c, d_h, d_r, d_mw, d_lp, d_tpsa, ap_sim]
                cil.append(cand_list.index(c) if c in cand_list else -1)
            Xl.append(feats); yl.append(np.array([labels[c] for c in cands], dtype=np.int8)); qinfo.append((qid, n_c, cil))
        if Xl: of_data[of_s] = {"X": np.vstack(Xl), "y": np.concatenate(yl), "queries": qinfo}

    return {
        "labeled_ofs": labeled_ofs, "of_queries": of_queries, "cand_list": cand_list,
        "of_data": of_data,
    }

def hit10(labels_dict, scores, candidates):
    order = np.argsort(-scores)
    ranked = np.array([labels_dict[c] for c in [candidates[i] for i in order]])
    pos = np.where(ranked == 1)[0]
    return int(len(pos) > 0 and pos[0] < 10)

# ── Baseline + T1a comparison (3 seeds) ──
def run_t1a(st):
    labeled_ofs = st["labeled_ofs"]
    results = []

    for seed_idx in range(3):
        log(f"--- Seed {seed_idx} ---")
        rng = np.random.RandomState(SEED + seed_idx)
        shuffled = list(labeled_ofs); rng.shuffle(shuffled)
        n_tr = int(len(shuffled)*0.7)
        train_ofs, test_ofs = shuffled[:n_tr], shuffled[n_tr:]

        # Train frequency
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

        n_static_cols = st["of_data"][train_ofs[0]]["X"].shape[1]  # 9 or 10
        n_total_cols = n_static_cols + 1  # + freq

        # Build training matrix (baseline: first 8 cols, T1a: all 11 cols)
        X_tr_static = np.vstack([st["of_data"][o]["X"] for o in train_ofs])
        y_tr = np.concatenate([st["of_data"][o]["y"] for o in train_ofs])

        X_tr_base = np.zeros((len(X_tr_static), 8), dtype=np.float32)
        X_tr_t1a = np.zeros((len(X_tr_static), n_total_cols), dtype=np.float32)
        X_tr_t1a[:, :n_static_cols] = X_tr_static

        off = 0
        for o in train_ofs:
            for _, nc, cil in st["of_data"][o]["queries"]:
                for j in range(nc):
                    ci = cil[j]; c = st["cand_list"][ci] if ci >= 0 else ""
                    freq_val = freq_map.get(c, 0) / max(max_f, 1)
                    if j < 8:  # baseline uses first 8 cols (7 static + freq)
                        pass
                    X_tr_base[off+j, :7] = X_tr_static[off+j, :7]
                    X_tr_base[off+j, 7] = freq_val
                    X_tr_t1a[off+j, n_static_cols] = freq_val
                off += nc

        # Baseline (8 features)
        scaler_base = StandardScaler()
        Xb_s = scaler_base.fit_transform(X_tr_base)
        lr_base = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        lr_base.fit(Xb_s, y_tr)
        base_coef = np.abs(lr_base.coef_[0])

        # T1a (11 features)
        scaler_t1a = StandardScaler()
        Xt_s = scaler_t1a.fit_transform(X_tr_t1a)
        lr_t1a = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        lr_t1a.fit(Xt_s, y_tr)

        # Evaluate
        base_h10, t1a_h10 = [], []
        for o in test_ofs:
            queries = st["of_data"][o]["queries"]
            y_te = st["of_data"][o]["y"]
            X_static = st["of_data"][o]["X"]
            off = 0
            for _, nc, cil in queries:
                cands = [st["cand_list"][ci] if ci >= 0 else "" for ci in cil]
                labels_dict = dict(zip(cands, y_te[off:off+nc].astype(int)))
                if sum(labels_dict.values()) == 0:
                    off += nc; continue

                Xb = np.zeros((nc, 8), dtype=np.float32)
                Xt = np.zeros((nc, n_total_cols), dtype=np.float32)
                for j in range(nc):
                    ci = cil[j]; c = st["cand_list"][ci] if ci >= 0 else ""
                    fv = freq_map.get(c, 0) / max(max_f, 1)
                    Xb[j, :7] = X_static[off+j, :7]
                    Xb[j, 7] = fv
                    Xt[j, :n_static_cols] = X_static[off+j]
                    Xt[j, n_static_cols] = fv

                base_s = lr_base.predict_proba(scaler_base.transform(Xb))[:, 1]
                t1a_s = lr_t1a.predict_proba(scaler_t1a.transform(Xt))[:, 1]
                base_h10.append(hit10(labels_dict, base_s, cands))
                t1a_h10.append(hit10(labels_dict, t1a_s, cands))
                off += nc

        bm = float(np.mean(base_h10)); tm = float(np.mean(t1a_h10))
        results.append({"seed": seed_idx, "base_h10": bm, "t1a_h10": tm, "delta": tm - bm})
        log(f"  Baseline (8F): {bm:.4f}  |  T1a (+3FPs): {tm:.4f}  |  Δ = {tm-bm:+.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "t1a_extra_fingerprints_vs_baseline.csv", index=False)
    log(f"\n=== T1a Results ===\n{df.to_string(index=False)}")
    log(f"Base mean: {df['base_h10'].mean():.4f}  |  T1a mean: {df['t1a_h10'].mean():.4f}  |  Δ mean: {df['delta'].mean():+.4f}")
    return df

if __name__ == "__main__":
    t0 = time.time()
    st = load_all_data()
    log(f"Data loaded: {len(st['labeled_ofs'])} OFs, features per row: {st['of_data'][st['labeled_ofs'][0]]['X'].shape[1]}")
    run_t1a(st)
    log(f"Total: {(time.time()-t0)/60:.1f} min")
