"""
Route A D1: MMP Mining — chunked BRICS fragmentation + global grouping + pair generation.
Reads D0 standardized chunks; writes fragment records, reduce index, pairs.
"""
import json, os, sys, hashlib, time, glob, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS, rdMolDescriptors

BASE = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\routeA_chembl37k_d0d3_engineering_safe")
CHUNK_DIR = BASE / "01_d0_data_audit/standardized_chunks"
FRAG_DIR = BASE / "02_d1_fragment_map/fragment_records_chunks"
IDX_DIR = BASE / "03_d1_reduce_index"
PAIR_DIR = BASE / "04_d1_pair_generation/pair_shards"
CHK_DIR = BASE / "checkpoints"
for d in [FRAG_DIR, IDX_DIR, PAIR_DIR, CHK_DIR]: d.mkdir(parents=True, exist_ok=True)

HASH_BUCKETS = 256
MAX_PAIRS_PER_GROUP = 10000
MAX_PER_BUCKET = 500000
CORE_MIN_HEAVY = 8
FRAG_MIN = 1
FRAG_MAX = 25
MOL_MIN = 10
MOL_MAX = 80

t0 = time.time()
chunks = sorted(glob.glob(str(CHUNK_DIR / "standardized_chunk_*.jsonl")))
print(f"Found {len(chunks)} D0 chunks")

if len(chunks) == 0:
    print("No D0 chunks found — D0 may not be complete yet.")
    print("Please wait for D0 to finish or check the chunk directory.")
    sys.exit(0)

# ================================================================
# D1 MAP: Fragmentation (chunked, resumable)
# ================================================================
print("\n=== D1 MAP: BRICS Fragmentation ===")
all_manifest = []
chunk_stats = []
total_frags = 0

for chunk_path in chunks:
    chunk_name = os.path.basename(chunk_path).replace(".jsonl", "")
    ckpt = CHK_DIR / f"fragment_{chunk_name}.done"
    if ckpt.exists():
        print(f"  SKIP {chunk_name} (checkpoint exists)")
        continue

    mols = []
    with open(chunk_path) as f:
        for line in f:
            mols.append(json.loads(line))

    frag_records = []
    for m in mols:
        mol = Chem.MolFromSmiles(m["canonical_smiles"])
        if mol is None:
            continue
        n_heavy = m["num_heavy_atoms"]
        if n_heavy < MOL_MIN or n_heavy > MOL_MAX:
            continue

        # BRICS fragmentation
        try:
            bonds = list(BRICS.FindBRICSBonds(mol))
        except:
            continue

        for (a1, a2), envs in bonds:
            # Single-cut MMP only
            frag_mol = Chem.FragmentOnBonds(mol, [mol.GetBondBetweenAtoms(int(a1),int(a2)).GetIdx()])
            # Get fragments
            frags = Chem.GetMolFrags(frag_mol, asMols=True)
            if len(frags) != 2:
                continue

            # Identify core (larger) and fragment (smaller)
            h1, h2 = frags[0].GetNumHeavyAtoms(), frags[1].GetNumHeavyAtoms()
            if h1 >= h2:
                core, frag = frags[0], frags[1]
                core_h, frag_h = h1, h2
            else:
                core, frag = frags[1], frags[0]
                core_h, frag_h = h2, h1

            if core_h < CORE_MIN_HEAVY or frag_h < FRAG_MIN or frag_h > FRAG_MAX:
                continue

            try:
                core_smi = Chem.MolToSmiles(core, isomericSmiles=False)
                frag_smi = Chem.MolToSmiles(frag, isomericSmiles=False)
                core_key = hashlib.md5(core_smi.encode()).hexdigest()[:16]
                frag_key = hashlib.md5(frag_smi.encode()).hexdigest()[:16]
            except:
                continue

            # Attachment signature: atom symbols at cut bond
            a1_sym = mol.GetAtomWithIdx(int(a1)).GetSymbol()
            a2_sym = mol.GetAtomWithIdx(int(a2)).GetSymbol()
            attach_sig = "|".join(sorted([a1_sym, a2_sym]))

            rec = {
                "fragment_record_id": f"fr{total_frags:08d}",
                "mol_id": m["mol_id"],
                "canonical_smiles": m["canonical_smiles"],
                "core_smiles": core_smi, "core_key": core_key,
                "fragment_smiles": frag_smi, "fragment_key": frag_key,
                "attachment_signature": attach_sig, "attachment_count": 1,
                "cut_type": "single", "num_cuts": 1,
                "fragment_heavy_atoms": frag_h, "core_heavy_atoms": core_h,
                "molecule_heavy_atoms": n_heavy,
                "fragmentation_status": "OK", "failure_reason": "",
            }
            frag_records.append(rec)
            total_frags += 1

    # Write chunk
    out_path = FRAG_DIR / f"fragment_records_{chunk_name}.jsonl"
    with open(out_path, "w") as f:
        for fr in frag_records:
            f.write(json.dumps(fr) + "\n")

    ckpt.touch()
    chunk_stats.append({"chunk_id": chunk_name, "input_molecules": len(mols),
        "fragment_records": len(frag_records), "fragment_failures": len(mols) - len(frag_records)})
    if len(chunk_stats) % 10 == 0:
        print(f"  Processed {len(chunk_stats)}/{len(chunks)} chunks, {total_frags} fragments ({time.time()-t0:.0f}s)")

print(f"  Total fragments: {total_frags} in {len(chunk_stats)} chunks")
pd.DataFrame(chunk_stats).to_csv(BASE/"02_d1_fragment_map/fragment_chunk_stats.csv", index=False)

# Manifest
manifest_rows = []
for cp in sorted(glob.glob(str(FRAG_DIR/"fragment_records_*.jsonl"))):
    manifest_rows.append({"chunk_path": cp, "status": "written"})
pd.DataFrame(manifest_rows).to_csv(BASE/"02_d1_fragment_map/d1_fragment_records_all_manifest.csv", index=False)

# Summary
summary = {"total_fragment_records": total_frags, "total_chunks_processed": len(chunk_stats)}
with open(BASE/"02_d1_fragment_map/d1_fragment_records_summary.csv","w") as f:
    f.write("metric,value\n")
    for k,v in summary.items(): f.write(f"{k},{v}\n")

with open(BASE/"02_d1_fragment_map/D1_FRAGMENT_MAP_VERDICT.md","w") as f:
    f.write(f"# D1 Fragment Map Verdict\n\nTotal fragments: {total_frags}\nChunks: {len(chunk_stats)}\nStatus: OK\n")

print(f"  D1 MAP complete: {total_frags} fragments, {time.time()-t0:.0f}s")

# ================================================================
# D1 REDUCE: Global Grouping via hash buckets
# ================================================================
print("\n=== D1 REDUCE: Global Grouping ===")
t1 = time.time()

# Hash-bucket approach: distribute fragment records to bucket files by group_key hash
buckets = defaultdict(list)
all_group_sizes = defaultdict(int)

frag_files = sorted(glob.glob(str(FRAG_DIR/"fragment_records_*.jsonl")))
for ff in frag_files:
    with open(ff) as f:
        for line in f:
            rec = json.loads(line)
            gkey = rec["core_key"] + "|" + rec["attachment_signature"]
            bid = int(hashlib.md5(gkey.encode()).hexdigest(), 16) % HASH_BUCKETS
            buckets[bid].append(rec)
            all_group_sizes[gkey] += 1

# Write bucket files
bucket_dir = IDX_DIR / "group_buckets"
bucket_dir.mkdir(parents=True, exist_ok=True)
for bid, recs in buckets.items():
    with open(bucket_dir / f"bucket_{bid:03d}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

# Group summary
group_rows = []
capped = 0
for gkey, gsize in all_group_sizes.items():
    n_pairs = gsize * (gsize - 1) // 2
    will_cap = n_pairs > MAX_PAIRS_PER_GROUP
    if will_cap: capped += 1
    group_rows.append({"group_key": gkey, "group_size": gsize, "estimated_pair_count": n_pairs,
        "pair_explosion_flag": will_cap, "will_cap_pairs": will_cap})

pd.DataFrame(group_rows).to_csv(IDX_DIR/"d1_core_group_summary.csv", index=False)
print(f"  Groups: {len(group_rows)}, capped: {capped}, buckets: {len(buckets)}")

# ================================================================
# D1 PAIR GENERATION
# ================================================================
print("\n=== D1 PAIR GENERATION ===")
t2 = time.time()
pair_id = 0
pair_count = 0
shard_idx = 0
shard_pairs = []
transform_freq = defaultdict(int)

for bid in sorted(buckets.keys()):
    recs = buckets[bid]
    # Group by group_key
    groups = defaultdict(list)
    for r in recs:
        gkey = r["core_key"] + "|" + r["attachment_signature"]
        groups[gkey].append(r)

    for gkey, grecs in groups.items():
        n = len(grecs)
        if n < 2: continue
        max_pairs_in_group = min(n * (n - 1) // 2, MAX_PAIRS_PER_GROUP) if n > 100 else n * (n - 1) // 2
        pair_idx = 0

        # Generate sampled or all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if pair_idx >= max_pairs_in_group:
                    break
                ri, rj = grecs[i], grecs[j]
                # Avoid self-pairs
                if ri["fragment_key"] == rj["fragment_key"]:
                    continue
                trans_key = ri["fragment_key"] + "→" + rj["fragment_key"]
                transform_freq[trans_key] += 1

                pair = {
                    "pair_id": f"p{pair_id:08d}", "group_key": gkey,
                    "core_key": ri["core_key"], "core_smiles": ri["core_smiles"],
                    "attachment_signature": ri["attachment_signature"],
                    "old_mol_id": ri["mol_id"], "replacement_mol_id": rj["mol_id"],
                    "old_fragment_smiles": ri["fragment_smiles"],
                    "replacement_fragment_smiles": rj["fragment_smiles"],
                    "old_fragment_key": ri["fragment_key"],
                    "replacement_fragment_key": rj["fragment_key"],
                    "num_cuts": 1, "pair_direction": "directed",
                    "transform_key": trans_key, "same_core_group_size": n,
                    "pair_generation_mode": "capped" if n > 100 else "all_pairs",
                    "capped_group_flag": n > 100, "pair_status": "OK",
                }
                shard_pairs.append(pair)
                pair_id += 1
                pair_count += 1
                pair_idx += 1

                if len(shard_pairs) >= 500000:
                    spath = PAIR_DIR / f"d1_mmp_pairs_filtered_shard_{shard_idx:05d}.jsonl"
                    with open(spath, "w") as f:
                        for p in shard_pairs: f.write(json.dumps(p) + "\n")
                    print(f"  Wrote shard {shard_idx}: {len(shard_pairs)} pairs")
                    shard_idx += 1
                    shard_pairs = []

            if pair_idx >= max_pairs_in_group:
                break

# Final shard
if shard_pairs:
    spath = PAIR_DIR / f"d1_mmp_pairs_filtered_shard_{shard_idx:05d}.jsonl"
    with open(spath, "w") as f:
        for p in shard_pairs: f.write(json.dumps(p) + "\n")
    shard_idx += 1

# Transform frequency
tf_rows = [{"transform_key": k, "frequency": v} for k, v in sorted(transform_freq.items(), key=lambda x: -x[1])[:1000]]
pd.DataFrame(tf_rows).to_csv(BASE/"04_d1_pair_generation/d1_transform_frequency.csv", index=False)

# Capped groups
capped_df = pd.DataFrame(group_rows)[pd.DataFrame(group_rows)["will_cap_pairs"]]
capped_df.to_csv(BASE/"04_d1_pair_generation/d1_pair_explosion_capped_groups.csv", index=False)

# Summary
psum = {"num_fragment_records": total_frags, "num_core_attachment_groups": len(group_rows),
    "num_pairs_written": pair_count, "num_capped_groups": capped,
    "num_unique_transforms": len(transform_freq),
    "num_recurrent_transforms": sum(1 for v in transform_freq.values() if v >= 5),
    "num_transforms_seen_across_scaffolds": len([k for k,v in transform_freq.items() if v >= 3])}
with open(BASE/"04_d1_pair_generation/d1_pair_generation_summary.json", "w") as f: json.dump(psum, f, indent=2)

with open(BASE/"04_d1_pair_generation/D1_MMP_MINING_VERDICT.md", "w") as f:
    f.write(f"# D1 MMP Mining Verdict\n\nPairs: {pair_count}\nGroups: {len(group_rows)}\nTransforms: {len(transform_freq)}\n")

print(f"\nD1 complete: {pair_count} pairs, {len(transform_freq)} transforms, {time.time()-t0:.0f}s total")
