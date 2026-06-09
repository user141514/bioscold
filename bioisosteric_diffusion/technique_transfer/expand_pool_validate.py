#!/usr/bin/env python3
"""Pool expansion: LBC vs CA vs Morgan comparison + novelty filtering + pass-rate."""
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

# ── Data loading (same as before) ──
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
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), fp)
    return fp
def mprops(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    return np.array([m.GetNumHeavyAtoms(), Descriptors.RingCount(m),
        Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m)], dtype=np.float32)
def tanimoto(fp1, fp2):
    inter = float(fp1.dot(fp2)); denom = float(fp1.sum() + fp2.sum() - inter)
    return inter / max(denom, 0.001)
def physchem_score(ofp, cfp, op, cp):
    return 1.0 / (1.0 + abs(cp[0]-op[0]) + abs(cp[1]-op[1]) +
                  abs(cp[2]-op[2])/max(op[2],1) + abs(cp[3]-op[3])/max(abs(op[3])+1,1))

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
labeled_ofs = sorted({manifest[qid]["old_fragment_smiles"] for qid, labels in qlabels.items()
    if qid in manifest and manifest[qid]["old_fragment_smiles"] in of_fps and any(labels.values())})
zero_ofs = sorted(set(all_ofs) - set(labeled_ofs))
log(f"  {len(labeled_ofs)} labeled, {len(zero_ofs)} zero-positive OFs")

# Frequency
freq_map = {}
for o in labeled_ofs:
    for qid in [q for q in manifest if manifest[q]["old_fragment_smiles"] == o]:
        for c, lbl in qlabels.get(qid, {}).items():
            if lbl == 1: freq_map[c] = freq_map.get(c, 0) + 1
max_f = max(freq_map.values()) if freq_map else 1

# Train LBC
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
lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED); lr.fit(X_s, y_tr)
log("  LBC trained")

# ── Batched pool scan for 20 OFs ──
pool_files = sorted(glob.glob("chembl_candidate_pool/batch_*.parquet"))
N_TOP = 2000; K_DISPLAY = 15
rng = np.random.RandomState(SEED)
of_sample = list(rng.choice(labeled_ofs, 12, replace=False)) + list(rng.choice(zero_ofs, min(8, len(zero_ofs)), replace=False))
log(f"  Evaluating {len(of_sample)} OFs")

def get_pool_top(of_s, n=N_TOP):
    """Get top-N Morgan-similar pool candidates for an OF."""
    ofp = of_fps[of_s]; of_sum = ofp.sum()
    top_sims = np.zeros(n); top_smis = [""] * n; top_props = [None] * n
    for pf in pool_files:
        df = pd.read_parquet(pf)
        batch_fps = np.array([np.frombuffer(b, dtype=np.float32) for b in df["fp"].values])
        sims = np.dot(batch_fps, ofp.astype(np.float32)) / (batch_fps.sum(axis=1) + of_sum - np.dot(batch_fps, ofp.astype(np.float32)) + 1e-10)
        all_sims = np.concatenate([top_sims, sims])
        all_smis = top_smis + list(df["smiles"].values)
        all_props = top_props + [np.array([r["heavy_atoms"], r["rings"], r["mw"], r["logp"], r["tpsa"]], dtype=np.float32) for _, r in df.iterrows()]
        best = np.argsort(-all_sims)[:n]
        top_sims = all_sims[best]; top_smis = [all_smis[i] for i in best]; top_props = [all_props[i] for i in best]
    return top_sims, top_smis, top_props

def compute_scores(of_s, smis, props_list, sims):
    """Compute LBC, CA, Morgan scores for candidates."""
    ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
    n_c = len(smis)
    feats = np.zeros((n_c, 8), dtype=np.float32)
    ca_scores = np.zeros(n_c); morgan_scores = np.zeros(n_c)
    for i, (smi, cp, m) in enumerate(zip(smis, props_list, sims)):
        pc = physchem_score(ofp, None, op, cp)  # physchem component
        ca_scores[i] = 0.75 * m + 0.25 * pc  # CA with tuned lam=0.75
        morgan_scores[i] = m
        # LBC features (bit_corr approximated, other FP needed)
        cfp = mfp(smi)
        bit_c = 0.0
        if cfp is not None and cfp.sum() > 0:
            cfp_n = cfp / (cfp.sum() + 1e-10)
            if ofp.sum() > 0:
                corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
                bit_c = corr if np.isfinite(corr) else 0.0
            feats[i] = [m, bit_c, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                abs(cp[4]-op[4])/max(op[4]+1,1), freq_map.get(smi, 0)/max(max_f, 1)]
    lbc_scores = lr.predict_proba(scaler.transform(feats))[:, 1]
    return lbc_scores, ca_scores, morgan_scores

def murcko(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return smi
    try:
        core = Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(m)
        return core if core else smi
    except: return smi

def filter_novelty(smis, props_list, sims, of_s, top_k=5):
    """Filter top-k per method, remove identity/near-dup, return novel picks."""
    of_murcko = murcko(of_s)
    results = {"lbc": [], "ca": [], "morgan": []}
    for method in results:
        seen_smi = set(); seen_murcko = set()
        for idx in range(len(smis)):
            if len(results[method]) >= top_k: break
            smi = smis[idx]
            if smi == of_s: continue  # skip identity
            if smi in seen_smi: continue
            m_smi = murcko(smi)
            if m_smi == of_murcko: continue  # same Murcko scaffold
            seen_smi.add(smi); seen_murcko.add(m_smi)
            results[method].append(idx)
    return results

# ── Main evaluation ──
all_rows = []

for of_s in of_sample:
    sims, smis, props_list = get_pool_top(of_s, N_TOP)
    lbc, ca, morgan = compute_scores(of_s, smis, props_list, sims)

    # Rank by each method
    idx_lbc = np.argsort(-lbc)
    idx_ca = np.argsort(-ca)
    idx_morgan = np.argsort(-morgan)

    # Novelty-filtered top-5
    novel = filter_novelty(smis, props_list, sims, of_s, top_k=5)

    for rank in range(5):
        for method, idx_list in [("LBC", novel["lbc"]), ("CA", novel["ca"]), ("Morgan", novel["morgan"])]:
            i = idx_list[rank] if rank < len(idx_list) else -1
            if i < 0: continue
            smi = smis[i]
            ofp = of_fps[of_s]; cfp = mfp(smi)
            morgan_val = sims[i]
            # Plausibility heuristics
            if smi == of_s: plaus = "identity"
            elif morgan_val > 0.95: plaus = "near-identical"
            elif cfp is not None and cfp.sum() > 0:
                # Check for reactive groups
                has_halogen = any(atom in smi for atom in ["Cl", "Br", "I", "F"])
                has_carbonyl = "C(=O)" in smi or "C=O" in smi
                has_nitro = "[N+](=O)[O-]" in smi
                if has_nitro: plaus = "potentially-reactive"
                elif has_halogen and has_carbonyl: plaus = "functionalized-analog"
                elif has_halogen: plaus = "halogen-analog"
                elif abs(float(Descriptors.MolWt(Chem.MolFromSmiles(smi)) or 0) - float(of_props[of_s][2])) < 20:
                    plaus = "property-conserving"
                else: plaus = "structural-analog"
            else: plaus = "structural-analog"

            all_rows.append({
                "of": of_s, "of_type": "labeled" if of_s in labeled_ofs else "zero-pos",
                "method": method, "rank": rank + 1,
                "candidate": smi, "morgan": round(morgan_val, 3),
                "score": round(lbc[i] if method == "LBC" else ca[i] if method == "CA" else morgan[i], 4),
                "plausibility": plaus, "filtered": "novelty",
            })

    if len(all_rows) % 100 == 0:
        log(f"  {len(all_rows)} entries so far...")

df = pd.DataFrame(all_rows)
df.to_csv(OUT / "pool_expansion_validation.csv", index=False)

# ── Pass-rate analysis ──
log("\n=== PASS-RATE ANALYSIS ===")
plausibility_counts = df.groupby(["method", "plausibility"]).size().unstack(fill_value=0)
print(plausibility_counts.to_string())

# Summary by method
for method in ["LBC", "CA", "Morgan"]:
    sub = df[df["method"] == method]
    n_total = len(sub)
    n_implausible = len(sub[sub["plausibility"].str.contains("identity|near-identical|reactive")])
    n_good = n_total - n_implausible
    log(f"  {method}: {n_good}/{n_total} good ({100*n_good/max(n_total,1):.0f}%)")

# Overlap analysis
log("\n=== METHOD OVERLAP (top-5 novelty-filtered) ===")
overlap_rows = []
for of_s in of_sample:
    for method in ["LBC", "CA", "Morgan"]:
        sub = df[(df["of"] == of_s) & (df["method"] == method)]
        top5_smis = set(sub["candidate"].values[:5])
        overlap_rows.append({"of": of_s, "method": method, "n_unique": len(top5_smis)})
    # Pairwise overlap
    lbc_set = set(df[(df["of"] == of_s) & (df["method"] == "LBC")]["candidate"].values[:5])
    ca_set = set(df[(df["of"] == of_s) & (df["method"] == "CA")]["candidate"].values[:5])
    morgan_set = set(df[(df["of"] == of_s) & (df["method"] == "Morgan")]["candidate"].values[:5])
    log(f"  {of_s[:25]}: LBC-CA overlap={len(lbc_set & ca_set)}/{len(lbc_set | ca_set)}, LBC-Morgan={len(lbc_set & morgan_set)}/{len(lbc_set | morgan_set)}")

log(f"\nSaved to {OUT / 'pool_expansion_validation.csv'}")
log("Done")
