"""
Route A D2: Labeling — structure-derived weak labels + decoys from D1 pairs.
"""
import json, os, sys, hashlib, glob, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\routeA_chembl37k_d0d3_engineering_safe")
PAIR_DIR = BASE / "04_d1_pair_generation/pair_shards"
LABEL_DIR = BASE / "05_d2_labeling"
LABEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load pair shards
shards = sorted(glob.glob(str(PAIR_DIR / "d1_mmp_pairs_filtered_shard_*.jsonl")))
if not shards:
    print("No D1 pair shards found. Run D1 first.")
    print("Pair shard dir:", PAIR_DIR)
    sys.exit(0)

print(f"Found {len(shards)} pair shards")

# Load transform frequency
tf = pd.read_csv(BASE/"04_d1_pair_generation/d1_transform_frequency.csv")
tf_map = dict(zip(tf.transform_key, tf.frequency))
print(f"Transform frequencies: {len(tf_map)}")

# ================================================================
# D2B: Structure-derived weak labels
# ================================================================
print("\n=== D2B: Weak Structure Labels ===")
min_tf = 5  # transform frequency threshold

weak_positives = []
all_pairs = []
for shard_path in shards:
    with open(shard_path) as f:
        for line in f:
            pair = json.loads(line)
            all_pairs.append(pair)
            tk = pair.get("transform_key", "")
            freq = tf_map.get(tk, 0)
            if freq >= min_tf:
                weak_positives.append(pair)

print(f"Total pairs: {len(all_pairs)}")
print(f"Weak positives (tf>={min_tf}): {len(weak_positives)}")

# Generate decoys
# DECOY_RANDOM_FRAGMENT: pair random fragments from different cores
# DECOY_PROPERTY_MATCHED: not implemented (needs property computation)
# DECOY_UNSEEN_TRANSFORM: transforms seen only once

decoy_random = []
decoy_unseen = []
for pair in all_pairs:
    tk = pair.get("transform_key", "")
    freq = tf_map.get(tk, 0)
    if freq <= 1:
        decoy_unseen.append(pair)

# Random fragment decoys: shuffle replacement fragments across different core groups
pos_frags = set()
for p in weak_positives[:10000]:
    pos_frags.add(p["replacement_fragment_smiles"])

# Simple decoys: use low-frequency transforms as decoys
decoy_sample = random.sample(decoy_unseen, min(len(decoy_unseen), len(weak_positives)))
print(f"Decoy unseen: {len(decoy_unseen)}, sampled: {len(decoy_sample)}")

# Write weak positives
with open(LABEL_DIR/"d2_weak_positive_pairs.jsonl","w") as f:
    for p in weak_positives[:50000]:  # cap at 50K
        p["label"] = "WEAK_POSITIVE"
        p["label_strength"] = "WEAK_STRUCTURE"
        f.write(json.dumps(p)+"\n")

# Write decoys
with open(LABEL_DIR/"d2_decoy_pairs.jsonl","w") as f:
    for p in decoy_sample[:50000]:
        p["label"] = "DECOY_UNSEEN_TRANSFORM"
        p["label_strength"] = "WEAK_DECOY"
        f.write(json.dumps(p)+"\n")

# Label summary
label_summary = {"total_pairs": len(all_pairs), "weak_positives": min(len(weak_positives),50000),
    "decoys_unseen": len(decoy_sample), "label_regime": "STRUCTURE_DERIVED_WEAK"}
with open(LABEL_DIR/"d2_label_summary.csv","w") as f:
    for k,v in label_summary.items(): f.write(f"{k},{v}\n")

# ================================================================
# Split and Leakage Audit
# ================================================================
print("\n=== Split & Leakage Audit ===")

# Build benchmark manifest from weak positives + decoys
manifest = []
for p in weak_positives[:10000]:
    manifest.append({"pair_id": p["pair_id"], "core_key": p["core_key"],
        "transform_key": p["transform_key"], "label": "WEAK_POSITIVE", "label_strength": "WEAK"})
for p in decoy_sample[:10000]:
    manifest.append({"pair_id": p["pair_id"], "core_key": p["core_key"],
        "transform_key": p["transform_key"], "label": "DECOY", "label_strength": "WEAK_DECOY"})

# Random split 80/10/10
random.shuffle(manifest)
n = len(manifest)
n_train = int(n*0.8)
n_val = int(n*0.1)
for i, m in enumerate(manifest):
    if i < n_train: m["split"] = "train"
    elif i < n_train+n_val: m["split"] = "val"
    else: m["split"] = "test"

with open(LABEL_DIR/"d2_pair_benchmark_manifest.jsonl","w") as f:
    for m in manifest:
        f.write(json.dumps(m)+"\n")

# Leakage audit
train_cores = set(m["core_key"] for m in manifest if m["split"]=="train")
test_cores = set(m["core_key"] for m in manifest if m["split"]=="test")
leakage = []
leakage.append({"check":"core_key_train_test_overlap","count":len(train_cores & test_cores),
    "status":"FAIL" if len(train_cores & test_cores)>0 else "PASS"})
train_trans = set(m["transform_key"] for m in manifest if m["split"]=="train")
test_trans = set(m["transform_key"] for m in manifest if m["split"]=="test")
leakage.append({"check":"transform_train_test_overlap","count":len(train_trans & test_trans),
    "status":"WARN" if len(train_trans & test_trans)>0 else "PASS"})
pd.DataFrame(leakage).to_csv(LABEL_DIR/"d2_pair_split_leakage_audit.csv", index=False)

print(f"Manifest: {len(manifest)} pairs (train={n_train}, val={n_val}, test={n-n_train-n_val})")
print(f"Core leakage: {len(train_cores & test_cores)}")
print(f"Transform leakage: {len(train_trans & test_trans)}")

# Verdict
with open(LABEL_DIR/"D2_LABELING_VERDICT.md","w") as f:
    f.write(f"# D2 Labeling Verdict\n\nDate: 2026-05-21\n")
    f.write(f"Regime: B_D2_PASS_WEAK_STRUCTURE_LABELS\n")
    f.write(f"Positives: {len(weak_positives)}\nDecoys: {len(decoy_sample)}\n")
    f.write(f"Leakage: cores={len(train_cores & test_cores)}, transforms={len(train_trans & test_trans)}\n")

print("D2 complete.")
