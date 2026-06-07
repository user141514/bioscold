#!/usr/bin/env python3
"""T5: Reliable Negative Sampling — filter negatives by Morgan distance from positives.

Strategy: for each query, keep all positives + sample negatives from candidates
that are Morgan-dissimilar to the query's positive candidates (hard negatives are
those that are structurally close but not replacements — we keep those too).
Actually: we keep ALL negatives but add a sampling weight based on Morgan distance.
Hard negatives (Morgan-similar but not replacements) get higher weight.
"""

import json, glob, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 20260603
OUT = Path("technique_transfer/results")
OUT.mkdir(parents=True, exist_ok=True)

RDLogger.DisableLog("rdApp.*")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── data loading (same as baseline) ──
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

def load_all_data():
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

    labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in query_labels.items()
        if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})

    # Precompute features
    of_data = {}
    for of_s in labeled_ofs:
        ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
        Xl, yl, qinfo = [], [], []
        for qid in of_queries.get(of_s, []):
            labels = query_labels.get(qid, {})
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

    return {
        "labeled_ofs": labeled_ofs, "of_queries": of_queries, "cand_list": cand_list,
        "of_data": of_data, "cand_fps": cand_fps, "of_fps": of_fps,
    }

def hit10(labels_dict, scores, candidates):
    order = np.argsort(-scores)
    ranked = np.array([labels_dict[c] for c in [candidates[i] for i in order]])
    pos = np.where(ranked == 1)[0]
    return int(len(pos) > 0 and pos[0] < 10)

def run_t5(st):
    """Compare baseline (all negatives) vs T5 (hard negative re-weighting).

    T5 strategy: for each query, compute Morgan similarity between each negative candidate
    and the query's positive candidates. Negatives that are Morgan-similar to positives
    but NOT labeled as replacements are "hard negatives" — these get higher sample weight.

    We implement this as sample_weight in sklearn LogisticRegression.
    """
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

        # Build training matrix
        X_tr_7 = np.vstack([st["of_data"][o]["X7"] for o in train_ofs])
        y_tr = np.concatenate([st["of_data"][o]["y"] for o in train_ofs])
        X_tr = np.zeros((len(X_tr_7), 8), dtype=np.float32)
        X_tr[:, :7] = X_tr_7
        sample_weights = np.ones(len(y_tr), dtype=np.float32)

        # Compute per-query hard negative weight
        off = 0
        for o in train_ofs:
            ofp = st["of_fps"][o]
            for _, nc, cil in st["of_data"][o]["queries"]:
                cands = [st["cand_list"][ci] if ci >= 0 else "" for ci in cil]
                labels = y_tr[off:off+nc]

                # Fill freq column
                for j in range(nc):
                    ci = cil[j]; c = st["cand_list"][ci] if ci >= 0 else ""
                    X_tr[off+j, 7] = freq_map.get(c, 0) / max(max_f, 1)

                # Find positive candidates in this query
                pos_idx = [j for j in range(nc) if labels[j] == 1]
                neg_idx = [j for j in range(nc) if labels[j] == 0]

                if pos_idx and neg_idx:
                    # Compute Morgan similarity of each negative to the closest positive
                    for nj in neg_idx:
                        cn = cands[nj]
                        cfp_n = st["cand_fps"].get(cn)
                        if cfp_n is None: continue
                        max_sim = 0.0
                        for pj in pos_idx:
                            cp = cands[pj]
                            cfp_p = st["cand_fps"].get(cp)
                            if cfp_p is None: continue
                            inter = float(cfp_n.dot(cfp_p))
                            denom = float(cfp_n.sum() + cfp_p.sum() - inter)
                            sim = inter / max(denom, 0.001)
                            max_sim = max(max_sim, sim)
                        # Hard negative if Morgan-similar to a positive (sim > 0.3) but NOT a replacement
                        if max_sim > 0.3:
                            sample_weights[off + nj] = 2.0  # double weight for hard negatives

                off += nc

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        # Baseline: uniform weights
        from sklearn.linear_model import LogisticRegression
        lr_base = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        lr_base.fit(X_tr_s, y_tr)

        # T5: re-weighted
        lr_t5 = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        lr_t5.fit(X_tr_s, y_tr, sample_weight=sample_weights)

        # Evaluate
        base_h10, t5_h10 = [], []
        for o in test_ofs:
            queries = st["of_data"][o]["queries"]
            y_te = st["of_data"][o]["y"]
            X7 = st["of_data"][o]["X7"]
            off = 0
            for _, nc, cil in queries:
                cands = [st["cand_list"][ci] if ci >= 0 else "" for ci in cil]
                labels_dict = dict(zip(cands, y_te[off:off+nc].astype(int)))
                if sum(labels_dict.values()) == 0:
                    off += nc; continue
                X_q = np.zeros((nc, 8), dtype=np.float32)
                X_q[:, :7] = X7[off:off+nc]
                X_q[:, 7] = [freq_map.get(c, 0) / max(max_f, 1) for c in cands]
                X_q_s = scaler.transform(X_q)
                base_s = lr_base.predict_proba(X_q_s)[:, 1]
                t5_s = lr_t5.predict_proba(X_q_s)[:, 1]
                base_h10.append(hit10(labels_dict, base_s, cands))
                t5_h10.append(hit10(labels_dict, t5_s, cands))
                off += nc

        bm = float(np.mean(base_h10)); tm = float(np.mean(t5_h10))
        results.append({"seed": seed_idx, "base_h10": bm, "t5_h10": tm, "delta": tm - bm})
        log(f"  Baseline: {bm:.4f}  |  T5 HardNeg: {tm:.4f}  |  Δ = {tm-bm:+.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "t5_negative_sampling_vs_baseline.csv", index=False)
    log(f"\n=== T5 Results ===\n{df.to_string(index=False)}")
    log(f"Base mean: {df['base_h10'].mean():.4f}  |  T5 mean: {df['t5_h10'].mean():.4f}  |  Δ mean: {df['delta'].mean():+.4f}")
    return df

if __name__ == "__main__":
    st = load_all_data()
    log(f"Data loaded: {len(st['labeled_ofs'])} OFs")
    run_t5(st)
