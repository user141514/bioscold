#!/usr/bin/env python3
"""
D4A2G Part A+B: Embedding Inventory + Delta Dataset + Gate A (Delta Structure Test)
=====================================================================================

Step 1: Morgan FP (r=2, nBits=2048) for all 166 unique fragments
Step 2: Delta dataset (old -> replacement vector differences)
Step 3: Gate A — delta structure test (within/between transform similarity, kNN purity, etc.)
"""

import json, csv, os, sys, time, random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.logger().setLevel(RDLogger.ERROR)

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path("E:/zuhui/bioisosteric_diffusion")
D4A0 = BASE / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
OUT = BASE / "plan_results/routeA_chembl37k_d4a2g_transform_vector_gates"
MANIFEST = D4A0 / "d4a0_query_split_manifest.jsonl"
VOCAB = D4A0 / "d4a0_train_replacement_vocabulary.csv"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def write_csv(path, rows, fieldnames):
    fpath = OUT / path
    os.makedirs(fpath.parent, exist_ok=True)
    with open(fpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"  wrote {len(rows)} rows -> {path}")


def write_json(path, obj):
    fpath = OUT / path
    os.makedirs(fpath.parent, exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=2)
    log(f"  wrote -> {path}")


def write_jsonl(path, rows):
    fpath = OUT / path
    os.makedirs(fpath.parent, exist_ok=True)
    with open(fpath, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"  wrote {len(rows)} lines -> {path}")


# ── Morgan FP helper ──────────────────────────────────────────────
def smiles_to_morgan_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ====================================================================
# STEP 1: Embedding Inventory
# ====================================================================
log("=" * 60)
log("STEP 1: Fragment Embedding Inventory")
log("=" * 60)

# Collect all unique fragments
manifest_fragments = set()
with open(MANIFEST, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        manifest_fragments.add(d["old_fragment_smiles"])
        for r in d["positive_replacement_set"]:
            manifest_fragments.add(r)

# Also add vocab fragments (should be subset)
vocab_fragments = set()
with open(VOCAB, encoding="utf-8") as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split(",")
        vocab_fragments.add(parts[0])

all_fragments = sorted(manifest_fragments | vocab_fragments)
log(f"Unique fragments: {len(all_fragments)} (manifest: {len(manifest_fragments)}, vocab: {len(vocab_fragments)})")

# Compute Morgan FP for all
embedding_rows = []
for smi in all_fragments:
    fp = smiles_to_morgan_fp(smi, radius=2, nbits=2048)
    embedding_rows.append({
        "fragment_smiles": smi,
        "embedding_dim": 2048,
        "embedding_source": "MORGAN_FALLBACK",
        "radius": 2,
        "in_vocab": int(smi in vocab_fragments),
        "in_manifest": int(smi in manifest_fragments),
    })

write_csv("d4a2g_fragment_embedding_inventory.csv", embedding_rows,
           ["fragment_smiles", "embedding_dim", "embedding_source", "radius", "in_vocab", "in_manifest"])

embedding_config = {
    "n_unique_fragments": len(all_fragments),
    "n_vocab_fragments": len(vocab_fragments),
    "n_manifest_fragments": len(manifest_fragments),
    "embedding_dim": 2048,
    "embedding_source": "MORGAN_FALLBACK",
    "radius": 2,
    "nBits": 2048,
    "date": now(),
}
write_json("d4a2g_embedding_config.json", embedding_config)

# Build fragment -> Morgan FP map
fragment_fp_cache = {}
for smi in all_fragments:
    fragment_fp_cache[smi] = smiles_to_morgan_fp(smi, radius=2, nbits=2048)

# Load vocab freq features
vocab_freq = {}
with open(VOCAB, encoding="utf-8") as f:
    header = f.readline().strip().split(",")
    for line in f:
        parts = line.strip().split(",")
        vocab_freq[parts[0]] = {
            "global_freq": int(parts[2]),
            "attach_freq": int(parts[3]),
        }

log(f"Fragment FP cache built: {len(fragment_fp_cache)} entries")


# ====================================================================
# STEP 2: Delta Dataset
# ====================================================================
log("=" * 60)
log("STEP 2: Delta Dataset Construction")
log("=" * 60)

delta_rows = {"train": [], "val": [], "test": []}
unique_old = set()
unique_repl = set()

with open(MANIFEST, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        split = d["split"]
        old_smi = d["old_fragment_smiles"]
        attach_sig = d["attachment_signature"]
        core_key = d["core_key"]
        transform_keys = d.get("transform_key_set", [])

        z_old = fragment_fp_cache[old_smi]
        unique_old.add(old_smi)

        for repl_smi in d["positive_replacement_set"]:
            unique_repl.add(repl_smi)
            z_repl = fragment_fp_cache.get(repl_smi)
            if z_repl is None:
                continue
            delta_z = z_repl - z_old

            # Build transform keys for this pair
            # A transform key identifies the (old_fragment_hash -> replacement_fragment_hash) pair
            # From manifest: "9931b5f3f085d3c1->0473b1a4379f48b6"
            # We need to match old->replacement hash pairs
            # Since we don't have direct hashing, we group by old_fragment_smiles and replacement_fmiles

            row = {
                "old_fragment": old_smi,
                "replacement_fragment": repl_smi,
                "attachment_signature": attach_sig,
                "core_key": core_key,
                "transform_key_set": transform_keys,
                "split": split,
                "z_old": z_old.tolist(),
                "z_replacement": z_repl.tolist(),
                "delta_z": delta_z.tolist(),
            }

            if repl_smi in vocab_freq:
                row["global_freq"] = vocab_freq[repl_smi]["global_freq"]
                row["attach_freq"] = vocab_freq[repl_smi]["attach_freq"]
            else:
                row["global_freq"] = 0
                row["attach_freq"] = 0

            delta_rows[split].append(row)

log(f"Delta rows: train={len(delta_rows['train'])}, val={len(delta_rows['val'])}, test={len(delta_rows['test'])}")
log(f"Unique old fragments: {len(unique_old)}, unique replacements: {len(unique_repl)}")

# Write delta datasets
for split_name in ["train", "val", "test"]:
    write_jsonl(f"d4a2g_delta_dataset_{split_name}.jsonl", delta_rows[split_name])

delta_dataset_summary = {
    "n_train": len(delta_rows["train"]),
    "n_val": len(delta_rows["val"]),
    "n_test": len(delta_rows["test"]),
    "n_unique_old": len(unique_old),
    "n_unique_replacement": len(unique_repl),
}
write_json("d4a2g_delta_dataset_summary.json", delta_dataset_summary)


# ====================================================================
# STEP 3: Gate A — Delta Structure Test
# ====================================================================
log("=" * 60)
log("STEP 3: Gate A — Delta Structure Test")
log("=" * 60)

train_deltas = delta_rows["train"]
n_train = len(train_deltas)
log(f"Train deltas: {n_train}")

if n_train == 0:
    log("ERROR: No training deltas found!")
    sys.exit(1)

# Convert to arrays
delta_arrays = np.array([r["delta_z"] for r in train_deltas], dtype=np.float32)
transform_keys_list = [r["transform_key_set"] for r in train_deltas]
attach_sigs = [r["attachment_signature"] for r in train_deltas]

# --- 3.1: Delta norm distribution ---
delta_norms = np.linalg.norm(delta_arrays, axis=1)
norm_metrics = {
    "metric": "delta_norm",
    "mean": float(np.mean(delta_norms)),
    "std": float(np.std(delta_norms)),
    "min": float(np.min(delta_norms)),
    "max": float(np.max(delta_norms)),
    "p1": float(np.percentile(delta_norms, 1)),
    "p5": float(np.percentile(delta_norms, 5)),
    "p25": float(np.percentile(delta_norms, 25)),
    "p50": float(np.percentile(delta_norms, 50)),
    "p75": float(np.percentile(delta_norms, 75)),
    "p95": float(np.percentile(delta_norms, 95)),
    "p99": float(np.percentile(delta_norms, 99)),
}
log(f"Delta norm: mean={norm_metrics['mean']:.4f}, std={norm_metrics['std']:.4f}")

# --- 3.2: Within-transform cosine similarity ---
# Each delta row may have multiple transform keys. We use the first transform key per row for grouping.
# Transform key format: "hash1->hash2"
# We group by the first transform_key (the most specific transform)
# Actually, since each pair (old->replacement) corresponds to exactly one transform,
# let's compute a unique transform key from the pair.

# Better approach: group delta rows by (old_fragment, replacement_fragment)
# But the spec says "for each unique transform_key"
# Let's group by the first transform key for simplicity
# Actually, let me re-read: "for each unique transform_key"
# Transforms from manifest's transform_key_set. Each row has a transform_key_set.

# The transform key format: "hash1->hash2" where hash1 = old fragment hash, hash2 = replacement hash
# We'll group by the combination of (old_fragment, replacement_fragment) since that's the transform

# Actually, let's just group by the hash-pair from the first transform_key
transform_groups = defaultdict(list)
for i, row in enumerate(train_deltas):
    # Use the first transform key as the group identifier
    for tk in row.get("transform_key_set", []):
        transform_groups[tk].append(i)
    if not row.get("transform_key_set"):
        # Fallback: use old + replacement as key
        key = row["old_fragment"] + "->" + row["replacement_fragment"]
        transform_groups[key].append(i)

# Within-transform pairwise cosine similarity
within_cos_sims = []
for tk, indices in transform_groups.items():
    if len(indices) < 2:
        continue
    group_deltas = delta_arrays[indices]
    # Normalize
    norms = np.linalg.norm(group_deltas, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = group_deltas / norms
    # Pairwise cosine similarity
    cos_sim_matrix = normalized @ normalized.T
    # Upper triangle (excluding diagonal)
    n = len(indices)
    upper_tri = cos_sim_matrix[np.triu_indices(n, k=1)]
    if len(upper_tri) > 0:
        within_cos_sims.extend(upper_tri.tolist())

mean_within_cos = float(np.mean(within_cos_sims)) if within_cos_sims else 0.0
log(f"Within-transform cos_sim: mean={mean_within_cos:.4f}, n_pairs={len(within_cos_sims)}")

# --- 3.3: Cross-transform cosine similarity ---
# Random pairs from different transform keys
rng = np.random.RandomState(SEED)
tk_list = list(transform_groups.keys())
cross_cos_sims = []
n_cross_samples = min(50000, len(tk_list) * 100)

for _ in range(n_cross_samples):
    tk_a, tk_b = rng.choice(tk_list, size=2, replace=False)
    idx_a = rng.choice(transform_groups[tk_a])
    idx_b = rng.choice(transform_groups[tk_b])
    vec_a = delta_arrays[idx_a]
    vec_b = delta_arrays[idx_b]
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a > 0 and norm_b > 0:
        cos_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        cross_cos_sims.append(cos_sim)

mean_cross_cos = float(np.mean(cross_cos_sims)) if cross_cos_sims else 0.0
log(f"Cross-transform cos_sim: mean={mean_cross_cos:.4f}, n_pairs={len(cross_cos_sims)}")

# --- 3.4: Within/Between ratio ---
within_between_ratio = mean_within_cos / max(mean_cross_cos, 1e-10)
log(f"Within/Between ratio: {within_between_ratio:.4f}")

# --- 3.5: Attachment-conditioned analysis ---
attach_groups = defaultdict(list)
for i, sig in enumerate(attach_sigs):
    attach_groups[sig].append(i)

attach_metrics = {}
for sig, indices in attach_groups.items():
    group_deltas = delta_arrays[indices]
    mean_delta = np.mean(group_deltas, axis=0)
    # Within-attachment variance (mean of per-dimension variance)
    within_var = np.mean(np.var(group_deltas, axis=0))
    # Mean norm
    mean_norm = float(np.mean(np.linalg.norm(group_deltas, axis=1)))
    attach_metrics[sig] = {
        "attachment_signature": sig,
        "n_deltas": len(indices),
        "mean_norm": mean_norm,
        "within_attach_variance": float(within_var),
    }
    log(f"  Attachment {sig}: n={len(indices)}, mean_norm={mean_norm:.4f}, variance={within_var:.6f}")

# --- 3.6: kNN Purity ---
# For each delta, find k=5 nearest neighbors. Purity = fraction with same transform_key.
k_knn = min(5, n_train - 1)
log(f"Computing kNN purity (k={k_knn})...")

# We need a fast way to find nearest neighbors.
# Sample if too many points
knn_sample_size = min(5000, n_train)
knn_indices = rng.choice(n_train, size=knn_sample_size, replace=False) if n_train > 5000 else np.arange(n_train)

# Normalize deltas for cosine similarity
d_norms = np.linalg.norm(delta_arrays, axis=1, keepdims=True)
d_norms[d_norms == 0] = 1e-10
normalized_deltas = delta_arrays / d_norms

# For each sampled point, find kNN by cosine distance
n_purities = []
for i in knn_indices:
    query_vec = normalized_deltas[i:i+1]
    similarities = normalized_deltas @ query_vec.T  # n x 1
    similarities = similarities.flatten()
    similarities[i] = -1  # Exclude self

    nearest = np.argsort(-similarities)[:k_knn]
    # Check if any neighbor shares a transform key with current point
    current_tk_set = set(train_deltas[i].get("transform_key_set", []))
    shared = 0
    for n_idx in nearest:
        neighbor_tk_set = set(train_deltas[n_idx].get("transform_key_set", []))
        if current_tk_set & neighbor_tk_set:
            shared += 1
    purity = shared / k_knn
    n_purities.append(purity)

mean_purity = float(np.mean(n_purities))
# Random baseline: 1 / num_unique_transform_keys
num_tk = len(transform_groups)
random_baseline_pct = 100.0 / max(num_tk, 1)
log(f"kNN purity (k={k_knn}): mean={mean_purity:.4f} (random baseline: {random_baseline_pct:.4f}%)")
log(f"kNN purity vs baseline: {mean_purity*100:.2f}% vs {random_baseline_pct:.4f}% (ratio={mean_purity/max(random_baseline_pct/100, 1e-10):.2f}x)")

# --- 3.7: Within-attachment vs cross-attachment cosine sim ---
within_attach_sims = []
cross_attach_sims = []

for sig, indices in attach_groups.items():
    if len(indices) < 2:
        continue
    group_deltas = delta_arrays[indices]
    gnorms = np.linalg.norm(group_deltas, axis=1, keepdims=True)
    gnorms[gnorms == 0] = 1e-10
    gnormalized = group_deltas / gnorms
    cs_matrix = gnormalized @ gnormalized.T
    upper = cs_matrix[np.triu_indices(len(indices), k=1)]
    within_attach_sims.extend(upper.tolist())

# Cross-attachment: sample pairs from different attachments
attach_sig_list = list(attach_groups.keys())
for _ in range(min(50000, n_train)):
    sig_a, sig_b = rng.choice(attach_sig_list, size=2, replace=False)
    idx_a = rng.choice(attach_groups[sig_a])
    idx_b = rng.choice(attach_groups[sig_b])
    vec_a = delta_arrays[idx_a]
    vec_b = delta_arrays[idx_b]
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na > 0 and nb > 0:
        cross_cos = float(np.dot(vec_a, vec_b) / (na * nb))
        cross_attach_sims.append(cross_cos)

mean_within_attach = float(np.mean(within_attach_sims)) if within_attach_sims else 0.0
mean_cross_attach = float(np.mean(cross_attach_sims)) if cross_attach_sims else 0.0
log(f"Within-attachment cos_sim: {mean_within_attach:.4f}")
log(f"Cross-attachment cos_sim: {mean_cross_attach:.4f}")


# ============================================================
# Gate A verdict
# ============================================================
purity_ratio = mean_purity / max(random_baseline_pct / 100.0, 1e-10)
gate_a_pass = within_between_ratio > 1.5 or purity_ratio > 2.0
gate_a_reason = ""
if gate_a_pass:
    gate_a_reason = "PASS"
    if within_between_ratio > 1.5:
        gate_a_reason += f" (within/between ratio={within_between_ratio:.2f} > 1.5)"
    if purity_ratio > 2.0:
        gate_a_reason += f" (kNN purity ratio={purity_ratio:.2f}x > 2x)"
else:
    gate_a_reason = f"FAIL (within/between={within_between_ratio:.2f} <= 1.5, purity_ratio={purity_ratio:.2f}x <= 2x)"

log(f"\nGate A verdict: {'PASS' if gate_a_pass else 'FAIL'}")
log(f"  Reason: {gate_a_reason}")

# Write Gate A metrics
gate_a_metrics = [
    {"metric": "delta_norm_mean", "value": f"{norm_metrics['mean']:.6f}"},
    {"metric": "delta_norm_std", "value": f"{norm_metrics['std']:.6f}"},
    {"metric": "delta_norm_p50", "value": f"{norm_metrics['p50']:.6f}"},
    {"metric": "delta_norm_p95", "value": f"{norm_metrics['p95']:.6f}"},
    {"metric": "within_transform_mean_cos_sim", "value": f"{mean_within_cos:.6f}"},
    {"metric": "cross_transform_mean_cos_sim", "value": f"{mean_cross_cos:.6f}"},
    {"metric": "within_between_cos_ratio", "value": f"{within_between_ratio:.6f}"},
    {"metric": "kNN_purity_mean", "value": f"{mean_purity:.6f}"},
    {"metric": "random_baseline_purity_pct", "value": f"{random_baseline_pct:.6f}"},
    {"metric": "kNN_purity_vs_baseline_ratio", "value": f"{purity_ratio:.6f}"},
    {"metric": "within_attachment_mean_cos_sim", "value": f"{mean_within_attach:.6f}"},
    {"metric": "cross_attachment_mean_cos_sim", "value": f"{mean_cross_attach:.6f}"},
    {"metric": "n_unique_transform_keys", "value": str(num_tk)},
    {"metric": "n_train_deltas", "value": str(n_train)},
    {"metric": "n_within_cos_pairs", "value": str(len(within_cos_sims))},
    {"metric": "n_cross_cos_pairs", "value": str(len(cross_cos_sims))},
    {"metric": "gate_A_pass", "value": str(gate_a_pass)},
    {"metric": "gate_A_reason", "value": gate_a_reason},
]

write_csv("d4a2g_gateA_delta_structure_metrics.csv", gate_a_metrics,
           ["metric", "value"])

# Write cluster summary
cluster_summary = {
    "within_between_cos_ratio": within_between_ratio,
    "mean_within_transform_cos_sim": mean_within_cos,
    "mean_cross_transform_cos_sim": mean_cross_cos,
    "kNN_purity_mean": mean_purity,
    "kNN_purity_random_baseline": random_baseline_pct / 100.0,
    "kNN_purity_ratio_vs_baseline": purity_ratio,
    "within_attachment_mean_cos_sim": mean_within_attach,
    "cross_attachment_mean_cos_sim": mean_cross_attach,
    "n_unique_transform_keys": num_tk,
    "gate_A_pass": gate_a_pass,
    "per_attachment": attach_metrics,
}
write_json("d4a2g_gateA_delta_cluster_summary.json", cluster_summary)

log("\n" + "=" * 60)
log(f"Script 1 complete. Gate A: {'PASS' if gate_a_pass else 'FAIL'}")
log("=" * 60)
