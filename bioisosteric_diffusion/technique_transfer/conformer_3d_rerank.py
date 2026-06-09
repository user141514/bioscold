#!/usr/bin/env python3
"""3D rerank POC: LBC top-50 → conformer generation → 3D similarity rerank."""
import json, glob, time, numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 20260603; OUT = Path("technique_transfer/results")
OUT.mkdir(parents=True, exist_ok=True); RDLogger.DisableLog("rdApp.*")
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Electrostatic similarity: Coulomb overlap on a grid ──
def electrostatic_similarity(mol1, mol2, grid_spacing=0.5, padding=2.0):
    """Compute electrostatic similarity via grid-based Coulomb overlap."""
    # Get atom positions and charges
    conf1 = mol1.GetConformer(); conf2 = mol2.GetConformer()
    pos1 = np.array([list(conf1.GetAtomPosition(i)) for i in range(mol1.GetNumAtoms())])
    pos2 = np.array([list(conf2.GetAtomPosition(i)) for i in range(mol2.GetNumAtoms())])
    charges1 = np.array([float(a.GetDoubleProp('_GasteigerCharge')) for a in mol1.GetAtoms()])
    charges2 = np.array([float(a.GetDoubleProp('_GasteigerCharge')) for a in mol2.GetAtoms()])

    # Build grid
    all_pos = np.vstack([pos1, pos2])
    x_min, y_min, z_min = all_pos.min(axis=0) - padding
    x_max, y_max, z_max = all_pos.max(axis=0) + padding
    x = np.arange(x_min, x_max, grid_spacing)
    y = np.arange(y_min, y_max, grid_spacing)
    z = np.arange(z_min, z_max, grid_spacing)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid = np.stack([X, Y, Z], axis=-1)  # (nx, ny, nz, 3)

    def potential(grid, pos, charges):
        pot = np.zeros(grid.shape[:3])
        for i, (p, q) in enumerate(zip(pos, charges)):
            r = np.linalg.norm(grid - p, axis=-1) + 1e-8
            pot += q / r
        return pot

    pot1 = potential(grid, pos1, charges1)
    pot2 = potential(grid, pos2, charges2)
    # Normalized overlap (cosine similarity of potentials)
    sim = (pot1 * pot2).sum() / (np.sqrt((pot1**2).sum()) * np.sqrt((pot2**2).sum()) + 1e-10)
    return float(sim)

def generate_conformer(smi):
    """Generate a single low-energy conformer."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=42)
    if status != 0: return None
    AllChem.MMFFOptimizeMolecule(mol)
    AllChem.ComputeGasteigerCharges(mol)
    return mol

# ── Load baseline ──
log("Loading baseline data...")
manifest = {}
mp = Path("plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl")
with mp.open(encoding="utf-8") as f:
    for line in f:
        if line.strip(): q = json.loads(line); manifest[q["query_id"]] = q
def mfp(s):
    m = Chem.MolFromSmiles(s);
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
log(f"  {len(labeled_ofs)} labeled OFs, {len(cand_list)} candidates")

# Train LBC
freq_map = {}
for o in labeled_ofs:
    for qid in [q for q in manifest if manifest[q]["old_fragment_smiles"] == o]:
        for c, lbl in qlabels.get(qid, {}).items():
            if lbl == 1: freq_map[c] = freq_map.get(c, 0) + 1
max_f = max(freq_map.values()) if freq_map else 1
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
            feats[idx] = [inter/max(denom,0.001), 0, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
                abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
                abs(cp[4]-op[4])/max(op[4]+1,1), freq_map.get(c,0)/max(max_f,1)]
            cfp_n = cfp/(cfp.sum()+1e-10);
            if cfp.sum() > 0 and ofp.sum() > 0:
                corr = float(np.corrcoef(cfp_n, ofp_n)[0,1])
                feats[idx,1] = corr if np.isfinite(corr) else 0.0
        X_tr.append(feats); y_tr.extend([labels[c] for c in cands])
X_tr = np.vstack(X_tr); y_tr = np.array(y_tr, dtype=np.int8)
scaler = StandardScaler(); lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
lr.fit(scaler.fit_transform(X_tr), y_tr)
log("  LBC trained")

# ── 3D rerank on selected OFs ──
of_sample = list(np.random.RandomState(SEED).choice(labeled_ofs, 5, replace=False))
K_RERANK = 50

log(f"\n=== 3D RERANK POC ({len(of_sample)} OFs, top-{K_RERANK} candidates) ===")

for of_s in of_sample:
    ofp = of_fps[of_s]; op = of_props[of_s]; ofp_n = ofp / (ofp.sum() + 1e-10)
    # Get LBC scores for all 152 candidates
    feats_all = np.zeros((len(cand_list), 8), dtype=np.float32)
    for idx, c in enumerate(cand_list):
        cfp = cand_fps.get(c); cp = cand_props.get(c)
        if cfp is None: continue
        inter = float(cfp.dot(ofp)); denom = float(cfp.sum() + ofp.sum() - inter)
        feats_all[idx] = [inter/max(denom,0.001), 0, abs(cp[0]-op[0]), abs(cp[1]-op[1]),
            abs(cp[2]-op[2])/max(op[2],1), abs(cp[3]-op[3])/max(abs(op[3])+1,1),
            abs(cp[4]-op[4])/max(op[4]+1,1), freq_map.get(c,0)/max(max_f,1)]
        cfp_n = cfp/(cfp.sum()+1e-10)
        if cfp.sum() > 0 and ofp.sum() > 0:
            corr = float(np.corrcoef(cfp_n, ofp_n)[0,1])
            feats_all[idx,1] = corr if np.isfinite(corr) else 0.0
    scores = lr.predict_proba(scaler.transform(feats_all))[:, 1]
    top_idx = np.argsort(-scores)[:K_RERANK]
    log(f"  {of_s}: LBC top-{K_RERANK} selected")

    # Generate 3D conformers
    of_mol = generate_conformer(of_s)
    if of_mol is None:
        log(f"    SKIP: conformer generation failed for OF")
        continue

    top_cands = [cand_list[i] for i in top_idx]
    top_scores = scores[top_idx]
    esp_sims = []
    success = 0
    for i, c in enumerate(top_cands):
        cand_mol = generate_conformer(c)
        if cand_mol is None:
            esp_sims.append(0.0)
            continue
        try:
            esp = electrostatic_similarity(of_mol, cand_mol)
            esp_sims.append(esp)
            success += 1
        except:
            esp_sims.append(0.0)

    log(f"    Conformers: {success}/{K_RERANK} successful, ESP sim mean={np.mean(esp_sims):.4f}")

    # Rerank: combine LBC score + ESP similarity
    alpha = 0.3  # weight for 3D
    combined = (1 - alpha) * top_scores + alpha * np.array(esp_sims)
    rerank_idx = np.argsort(-combined)

    log(f"    Top-5 LBC:    {[top_cands[i][:20] for i in np.argsort(-top_scores)[:5]]}")
    log(f"    Top-5 Rerank: {[top_cands[i][:20] for i in rerank_idx[:5]]}")
    log(f"    Rank correlation (LBC vs Rerank): {np.corrcoef(top_scores, combined)[0,1]:.3f}")

log("\nDone")
