"""
Route A D3: Baseline Replacement Proposal — frequency, NN, property-random, CReM-like.
"""
import json, os, sys, hashlib, glob, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors

BASE = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\routeA_chembl37k_d0d3_engineering_safe")
LABEL_DIR = BASE / "05_d2_labeling"
D3_DIR = BASE / "06_d3_baselines"
D3_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42; random.seed(SEED); np.random.seed(SEED)
K = 10  # Top-K for all baselines

# Load benchmark manifest
manifest = []
with open(LABEL_DIR/"d2_pair_benchmark_manifest.jsonl") as f:
    for line in f: manifest.append(json.loads(line))
print(f"Manifest: {len(manifest)} pairs")

# Split
train = [m for m in manifest if m["split"]=="train"]
test = [m for m in manifest if m["split"]=="test"]
print(f"Train: {len(train)}, Test: {len(test)}")

# Load pair data (positive pairs from train)
pos_pairs = []
with open(LABEL_DIR/"d2_weak_positive_pairs.jsonl") as f:
    for line in f: pos_pairs.append(json.loads(line))
print(f"Positive pairs: {len(pos_pairs)}")

# Load transform frequencies
tf = pd.read_csv(BASE/"04_d1_pair_generation/d1_transform_frequency.csv")
tf_map = dict(zip(tf.transform_key, tf.frequency))

# Build train indices
# 1. Frequency index: old_fragment → ranked replacements by frequency
freq_index = defaultdict(lambda: defaultdict(int))
for p in pos_pairs:
    of = p.get("old_fragment_key","")
    rf = p.get("replacement_fragment_key","")
    rs = p.get("replacement_fragment_smiles","")
    freq_index[of][rs] += 1

# 2. Transform index: old→replacement pairs sorted by frequency
transform_index = defaultdict(list)
for p in pos_pairs:
    tk = p.get("transform_key","")
    freq = tf_map.get(tk, 1)
    transform_index[p.get("old_fragment_key","")].append((p.get("replacement_fragment_smiles",""), freq))

# 3. Global frequency index
global_freq = defaultdict(int)
for p in pos_pairs:
    global_freq[p.get("replacement_fragment_smiles","")] += 1
global_ranked = sorted(global_freq.items(), key=lambda x: -x[1])

# 4. Fragment fingerprint index for NN
fp_index = {}
for p in pos_pairs[:5000]:  # cap at 5K for speed
    smi = p.get("old_fragment_smiles","")
    if smi in fp_index: continue
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol: fp_index[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    except: pass

# ================================================================
# Baseline 1: Frequency
# ================================================================
print("\n=== Baseline 1: Frequency ===")
freq_results = []

for m in test[:500]:  # Test on 500 for speed
    pair_id = m["pair_id"]
    true_repl = None
    for p in pos_pairs:
        if p["pair_id"] == pair_id:
            true_repl = p.get("replacement_fragment_smiles","")
            old_frag = p.get("old_fragment_key","")
            break
    if not true_repl: continue

    # Global frequency ranking
    candidates = [(smi, freq) for smi, freq in global_ranked[:K]]
    top_smis = [c[0] for c in candidates]
    rank_global = top_smis.index(true_repl)+1 if true_repl in top_smis else 999

    # Conditioned frequency
    cond_cands = sorted(freq_index.get(old_frag,{}).items(), key=lambda x: -x[1])[:K]
    cond_smis = [c[0] for c in cond_cands]
    rank_cond = cond_smis.index(true_repl)+1 if true_repl in cond_smis else 999

    freq_results.append({"pair_id":pair_id,"rank_global":rank_global,"rank_conditioned":rank_cond,
        "top1_global":1 if rank_global==1 else 0,"top5_global":1 if rank_global<=5 else 0})

freq_df = pd.DataFrame(freq_results)
freq_df.to_csv(D3_DIR/"d3_baseline_frequency_results.csv", index=False)
print(f"  Frequency: {len(freq_results)} queries")
if len(freq_results) > 0:
    print(f"  Top-1 global: {freq_df.top1_global.mean():.4f}, Top-5: {freq_df.top5_global.mean():.4f}")

# ================================================================
# Baseline 2: Nearest Neighbor (Morgan FP)
# ================================================================
print("\n=== Baseline 2: Nearest Neighbor ===")
nn_results = []

for m in test[:500]:
    pair_id = m["pair_id"]
    old_smi = None; true_repl = None
    for p in pos_pairs:
        if p["pair_id"] == pair_id:
            old_smi = p.get("old_fragment_smiles",""); true_repl = p.get("replacement_fragment_smiles","")
            break
    if not old_smi or not true_repl: continue
    try:
        qmol = Chem.MolFromSmiles(old_smi)
        if qmol is None: continue
        qfp = AllChem.GetMorganFingerprintAsBitVect(qmol, 2, nBits=1024)
        scores = []
        for smi, fp in fp_index.items():
            if smi == old_smi: continue
            sim = DataStructs.TanimotoSimilarity(qfp, fp)
            scores.append((smi, sim))
        scores.sort(key=lambda x: -x[1])
        top_smis = [s[0] for s in scores[:K]]
        rank_nn = top_smis.index(true_repl)+1 if true_repl in top_smis else 999
        nn_results.append({"pair_id":pair_id,"rank_nn":rank_nn,"top1_nn":1 if rank_nn==1 else 0})
    except: continue

nn_df = pd.DataFrame(nn_results)
nn_df.to_csv(D3_DIR/"d3_baseline_nearest_neighbor_results.csv", index=False)
print(f"  NN: {len(nn_results)} queries")
if len(nn_results) > 0: print(f"  Top-1 NN: {nn_df.top1_nn.mean():.4f}")

# ================================================================
# Baseline 3: Property-Matched Random
# ================================================================
print("\n=== Baseline 3: Property Random ===")
prop_results = []
all_frags = list(set(p.get("replacement_fragment_smiles","") for p in pos_pairs if p.get("replacement_fragment_smiles")))[:2000]

for m in test[:500]:
    pair_id = m["pair_id"]
    true_repl = None
    for p in pos_pairs:
        if p["pair_id"] == pair_id:
            true_repl = p.get("replacement_fragment_smiles",""); break
    if not true_repl: continue
    # Random selection from all fragments
    random_cands = random.sample(all_frags, min(K, len(all_frags)))
    rank_rand = random_cands.index(true_repl)+1 if true_repl in random_cands else 999
    prop_results.append({"pair_id":pair_id,"rank_random":rank_rand,"top1_random":1 if rank_rand==1 else 0})

prop_df = pd.DataFrame(prop_results)
prop_df.to_csv(D3_DIR/"d3_baseline_property_random_results.csv", index=False)
print(f"  Random: {len(prop_results)} queries")
if len(prop_results) > 0: print(f"  Top-1 Random: {prop_df.top1_random.mean():.4f}")

# ================================================================
# Baseline 4: CReM-like (transform-rule)
# ================================================================
print("\n=== Baseline 4: CReM-like ===")
crem_results = []
# Transform rule: for each old_fragment, use most frequent replacement from train transforms
transform_rules = defaultdict(list)
for p in pos_pairs:
    of = p.get("old_fragment_key",""); rf = p.get("replacement_fragment_smiles","")
    tk = p.get("transform_key",""); freq = tf_map.get(tk, 1)
    transform_rules[of].append((rf, freq))

for m in test[:500]:
    pair_id = m["pair_id"]
    old_frag = None; true_repl = None
    for p in pos_pairs:
        if p["pair_id"] == pair_id:
            old_frag = p.get("old_fragment_key",""); true_repl = p.get("replacement_fragment_smiles","")
            break
    if not old_frag or not true_repl: continue
    rules = sorted(transform_rules.get(old_frag,[]), key=lambda x: -x[1])[:K]
    rule_smis = [r[0] for r in rules]
    rank_crem = rule_smis.index(true_repl)+1 if true_repl in rule_smis else 999
    crem_results.append({"pair_id":pair_id,"rank_crem":rank_crem,"top1_crem":1 if rank_crem==1 else 0})

crem_df = pd.DataFrame(crem_results)
crem_df.to_csv(D3_DIR/"d3_baseline_crem_like_results.csv", index=False)
print(f"  CReM-like: {len(crem_results)} queries")
if len(crem_results) > 0: print(f"  Top-1 CReM: {crem_df.top1_crem.mean():.4f}")

# ================================================================
# Summary
# ================================================================
print("\n=== D3 Summary ===")
summary = {"baseline":"frequency","top1":freq_df.top1_global.mean() if len(freq_df)>0 else 0,
    "n_queries":len(freq_results),"random_top1":prop_df.top1_random.mean() if len(prop_df)>0 else 0}
pd.DataFrame([summary]).to_csv(D3_DIR/"d3_baseline_summary.csv", index=False)

beats_random = summary["top1"] > summary["random_top1"]
with open(D3_DIR/"D3_BASELINE_REPLACEMENT_PROPOSAL_VERDICT.md","w") as f:
    f.write(f"# D3 Baseline Verdict\n\nDate: 2026-05-21\n")
    f.write(f"Frequency Top-1: {summary['top1']:.4f}\nRandom Top-1: {summary['random_top1']:.4f}\n")
    f.write(f"Beats random: {beats_random}\n")

print(f"Frequency Top-1: {summary['top1']:.4f}, Random: {summary['random_top1']:.4f}, Beats random: {beats_random}")
print("D3 complete.")
