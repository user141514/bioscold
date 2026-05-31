"""
D4A3: Geometry-backed A4C Evaluation of Learned Ranker Proposals
Parts A-D: Eval set selection, top-K proposals, RDKit geometry, A4C review
"""
import json, os, sys, warnings, random, time, collections, math
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, FilterCatalog
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

# ---- Paths ----
D4A0 = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\routeA_chembl37k_d0d3_engineering_safe\07_d4a0_matrix_freeze")
D4A1 = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\routeA_chembl37k_d4a1_learned_ranker")
D4A2 = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\routeA_chembl37k_d4a2_canonical_ranker_and_controls")
OUTDIR = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\routeA_chembl37k_d4a3_geometry_a4c_evaluation")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- Helpers ----
def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def hcap_smiles(smiles):
    """Replace * dummy atom with [H] for RDKit processing."""
    return smiles.replace("*", "[H]")

def run_fragment_geometry(smiles, n_conformers=10, seed=42):
    """
    RDKit geometry pipeline for fragment SMILES (with * dummy).
    Returns dict with geometry results.
    """
    result = {
        "smiles": smiles,
        "h_capped_smiles": None,
        "geometry_success": False,
        "num_conformers": 0,
        "num_mmff_optimized": 0,
        "mmff_success": False,
        "uff_fallback": False,
        "selected_energy": None,
        "error_message": None,
    }
    try:
        h_smi = hcap_smiles(smiles)
        result["h_capped_smiles"] = h_smi
        mol = Chem.MolFromSmiles(h_smi)
        if mol is None:
            result["error_message"] = "RDKit parse failed"
            return result

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        n_confs = len(list(conf_ids))
        result["num_conformers"] = n_confs

        if n_confs < 1:
            result["error_message"] = "Embedding failed (0 conformers)"
            return result

        # Re-embed because we consumed the iterator
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        mmff_results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)

        energies = []
        n_mmff_ok = 0
        for i, (res, energy) in enumerate(mmff_results):
            if res == 0:
                energies.append(energy)
                n_mmff_ok += 1

        result["num_mmff_optimized"] = n_mmff_ok

        if n_mmff_ok > 0:
            result["mmff_success"] = True
        else:
            # UFF fallback
            uff_results = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0)
            n_uff_ok = 0
            for i, (res, energy) in enumerate(uff_results):
                if res == 0:
                    energies.append(energy)
                    n_uff_ok += 1
            result["uff_fallback"] = True
            if n_uff_ok == 0:
                result["error_message"] = "MMFF and UFF both failed"
                return result

        if energies:
            result["selected_energy"] = float(min(energies))
            result["geometry_success"] = True
        else:
            result["error_message"] = "No valid energies after optimization"

    except Exception as e:
        result["error_message"] = str(e)

    return result

# ---- Load Data ----
print("=" * 60)
print("D4A3: Geometry-backed A4C Evaluation")
print("=" * 60)
print(f"Output: {OUTDIR}")

print("\n=== Loading data ===")

# Load query manifest (test split only)
manifest = load_jsonl(D4A0 / "d4a0_query_split_manifest.jsonl")
test_queries_list = [q for q in manifest if q["split"] == "test"]
test_queries_dict = {q["query_id"]: q for q in test_queries_list}
print(f"Test queries in manifest: {len(test_queries_list)}")

# Load D4A1 test predictions
print("Loading D4A1 test predictions...")
d4a1_predictions = load_jsonl(D4A1 / "d4a1_test_predictions.jsonl")
print(f"  Total predictions: {len(d4a1_predictions)}")

# Group predictions by query_id
preds_by_query = collections.defaultdict(list)
for p in d4a1_predictions:
    preds_by_query[p["query_id"]].append(p)
print(f"  Queries with predictions: {len(preds_by_query)}")

# Load test shard data (for attach_freq)
test_shard_dir = D4A0 / "matrices" / "test"
test_shard_files = sorted(test_shard_dir.glob("test_features_shard_*.jsonl"))
print(f"Test shard files: {len(test_shard_files)}")

# Build attach_freq lookup from test shards: (query_id, candidate) -> attach_freq
attach_freq_lookup = {}
for sf in test_shard_files:
    for entry in load_jsonl(sf):
        attach_freq_lookup[(entry["query_id"], entry["candidate"])] = entry["attach_freq"]

print(f"  attach_freq entries: {len(attach_freq_lookup)}")

# Also record which candidates are positive (label=1)
positive_replacement_check = collections.defaultdict(set)
for p in d4a1_predictions:
    if p["label"] == 1:
        positive_replacement_check[p["query_id"]].add(p["candidate"])

# ============================================================
# PART A: Select Evaluation Set
# ============================================================
print("\n=== Part A: Select Evaluation Set ===")

# For each query, determine HGB top-10 hit status and attach_freq top-10 hit status
query_strata = {}
for qid, preds in preds_by_query.items():
    if qid not in test_queries_dict:
        continue

    # Sort by HGB score desc
    sorted_hgb = sorted(preds, key=lambda x: x["score"], reverse=True)
    # Sort by attach_freq desc (use stored attach_freq)
    sorted_attach = sorted(preds, key=lambda x: attach_freq_lookup.get((qid, x["candidate"]), 0), reverse=True)

    hgb_top10_pos = any(p["label"] == 1 for p in sorted_hgb[:10])
    attach_top10_pos = any(p["label"] == 1 for p in sorted_attach[:10])

    if hgb_top10_pos and attach_top10_pos:
        stratum = "both_hit"
    elif hgb_top10_pos:
        stratum = "learned_only_hits"
    elif attach_top10_pos:
        stratum = "attach_only_hits"
    else:
        stratum = "both_miss"

    qinfo = test_queries_dict[qid]
    query_strata[qid] = {
        "stratum": stratum,
        "num_positive_replacements": qinfo["num_positive_replacements"],
        "hard_subset": not attach_top10_pos,
        "rare": qinfo["num_positive_replacements"] <= 1,
        "frequent": qinfo["num_positive_replacements"] >= 4,
        "hit_status": f"HGB_top10={int(hgb_top10_pos)}_Attach_top10={int(attach_top10_pos)}"
    }

# Stratum counts
stratum_counts = collections.Counter(s["stratum"] for s in query_strata.values())
print("Stratum counts:")
for s, c in sorted(stratum_counts.items()):
    print(f"  {s}: {c}")

hard_subset_count = sum(1 for s in query_strata.values() if s["hard_subset"])
rare_count = sum(1 for s in query_strata.values() if s["rare"])
frequent_count = sum(1 for s in query_strata.values() if s["frequent"])
print(f"  hard_subset (attach top10=0): {hard_subset_count}")
print(f"  rare_replacement (<=1): {rare_count}")
print(f"  frequent_replacement (>=4): {frequent_count}")

# Stratified sampling
eval_query_ids = set()
# ALL from learned_only_hits, attach_only_hits, hard_subset
for qid, s in query_strata.items():
    if s["stratum"] in ("learned_only_hits", "attach_only_hits"):
        eval_query_ids.add(qid)

hard_subset_ids = [qid for qid, s in query_strata.items() if s["hard_subset"]]
print(f"  Hard subset queries: {len(hard_subset_ids)}")

# Sample 500 each from both_hit, both_miss, rare, frequent
for stratum_name in ["both_hit", "both_miss"]:
    pool = [qid for qid, s in query_strata.items() if s["stratum"] == stratum_name]
    sample_size = min(500, len(pool))
    sampled = random.sample(pool, sample_size)
    eval_query_ids.update(sampled)
    print(f"  Sampled {sample_size}/{len(pool)} from {stratum_name}")

for attr_name, attr_key in [("rare", "rare"), ("frequent", "frequent")]:
    pool = [qid for qid, s in query_strata.items() if s[attr_key]]
    sample_size = min(500, len(pool))
    sampled = random.sample(pool, sample_size)
    eval_query_ids.update(sampled)
    print(f"  Sampled {sample_size}/{len(pool)} from {attr_name}")

# Add ALL hard_subset queries
# Note: hard_subset overlaps with learned_only_hits, already included
# But also includes both_miss queries. Add remaining ones.
already_in_hard = eval_query_ids & set(hard_subset_ids)
print(f"  Hard subset already in eval: {len(already_in_hard)}")
remaining_hard = [qid for qid in hard_subset_ids if qid not in eval_query_ids]
if remaining_hard:
    # Take up to the total target ~6000
    current_count = len(eval_query_ids)
    target_count = max(6000, current_count)
    remaining_needed = min(target_count - current_count, len(remaining_hard))
    if remaining_needed > 0:
        eval_query_ids.update(random.sample(remaining_hard, remaining_needed))
        print(f"  Added {remaining_needed} more hard_subset queries to reach ~{target_count}")

eval_query_ids = sorted(eval_query_ids)
print(f"  Total eval queries: {len(eval_query_ids)}")

# Stratum breakdown in eval set
eval_stratum_counts = collections.Counter(query_strata[qid]["stratum"] for qid in eval_query_ids)
print("  Eval set stratum breakdown:")
for s, c in sorted(eval_stratum_counts.items()):
    print(f"    {s}: {c}")

# Save eval query set
eval_rows = []
for qid in eval_query_ids:
    s = query_strata[qid]
    qinfo = test_queries_dict[qid]
    eval_rows.append({
        "query_id": qid,
        "old_fragment": qinfo["old_fragment_smiles"],
        "attachment_signature": qinfo["attachment_signature"],
        "positive_replacement_set": json.dumps(qinfo["positive_replacement_set"]),
        "num_positive_replacements": qinfo["num_positive_replacements"],
        "stratum": s["stratum"],
        "hard_subset": int(s["hard_subset"]),
        "rare": int(s["rare"]),
        "frequent": int(s["frequent"]),
        "hit_status": s["hit_status"],
    })
eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUTDIR / "d4a3_eval_query_set.csv", index=False)
print(f"  Saved d4a3_eval_query_set.csv ({len(eval_df)} rows)")

# ============================================================
# PART B: Prepare Top-K Proposals
# ============================================================
print("\n=== Part B: Prepare Top-K Proposals ===")

all_proposals = []
k = 10

for qid in eval_query_ids:
    preds = preds_by_query[qid]
    if not preds:
        continue

    # M0: attachment_frequency (top-10 by attach_freq desc)
    attach_sorted = sorted(preds, key=lambda x: attach_freq_lookup.get((qid, x["candidate"]), 0), reverse=True)
    for rank, p in enumerate(attach_sorted[:k]):
        all_proposals.append({
            "query_id": qid,
            "method": "M0_attachment_frequency",
            "rank": rank + 1,
            "replacement_smiles": p["candidate"],
            "is_exact_positive": int(p["label"] == 1),
            "proposal_source": "test_shard_attach_freq",
            "candidate_in_train_vocab": 1,
            "replacement_frequency": attach_freq_lookup.get((qid, p["candidate"]), 0),
            "attachment_signature": test_queries_dict[qid]["attachment_signature"],
            "old_fragment": test_queries_dict[qid]["old_fragment_smiles"],
        })

    # M1: canonical HGB (top-10 by score desc)
    hgb_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    for rank, p in enumerate(hgb_sorted[:k]):
        all_proposals.append({
            "query_id": qid,
            "method": "M1_canonical_HGB",
            "rank": rank + 1,
            "replacement_smiles": p["candidate"],
            "is_exact_positive": int(p["label"] == 1),
            "proposal_source": "d4a1_test_predictions",
            "candidate_in_train_vocab": 1,
            "replacement_frequency": attach_freq_lookup.get((qid, p["candidate"]), 0),
            "score": float(p["score"]),
            "attachment_signature": test_queries_dict[qid]["attachment_signature"],
            "old_fragment": test_queries_dict[qid]["old_fragment_smiles"],
        })

    # M2: best D4A2 ranker (B0_HGB_reproduced, using D4A1 HGB scores)
    # Note: Same as M1 since B0 is HGB reproduction and no per-candidate preds available
    for rank, p in enumerate(hgb_sorted[:k]):
        all_proposals.append({
            "query_id": qid,
            "method": "M2_best_D4A2_ranker",
            "rank": rank + 1,
            "replacement_smiles": p["candidate"],
            "is_exact_positive": int(p["label"] == 1),
            "proposal_source": "d4a2_canonical_hgb_B0_reproduction",
            "candidate_in_train_vocab": 1,
            "replacement_frequency": attach_freq_lookup.get((qid, p["candidate"]), 0),
            "score": float(p["score"]),
            "attachment_signature": test_queries_dict[qid]["attachment_signature"],
            "old_fragment": test_queries_dict[qid]["old_fragment_smiles"],
        })

    # M3: random_global (random score, top-10)
    random_sorted = list(preds)
    random.shuffle(random_sorted)
    for rank, p in enumerate(random_sorted[:k]):
        all_proposals.append({
            "query_id": qid,
            "method": "M3_random_global",
            "rank": rank + 1,
            "replacement_smiles": p["candidate"],
            "is_exact_positive": int(p["label"] == 1),
            "proposal_source": "random_shuffle",
            "candidate_in_train_vocab": 1,
            "replacement_frequency": attach_freq_lookup.get((qid, p["candidate"]), 0),
            "attachment_signature": test_queries_dict[qid]["attachment_signature"],
            "old_fragment": test_queries_dict[qid]["old_fragment_smiles"],
        })

print(f"Total proposals: {len(all_proposals)}")

# Save proposals
with open(OUTDIR / "d4a3_topk_proposals.jsonl", "w") as f:
    for prop in all_proposals:
        f.write(json.dumps(prop) + "\n")
print(f"  Saved d4a3_topk_proposals.jsonl")

# Quick stats
method_counts = collections.Counter(p["method"] for p in all_proposals)
for m, c in sorted(method_counts.items()):
    pos = sum(1 for p in all_proposals if p["method"] == m and p["is_exact_positive"])
    print(f"  {m}: {c} proposals, {pos} exact positives ({100*pos/c:.1f}%)")

# ============================================================
# PART C: RDKit Geometry
# ============================================================
print("\n=== Part C: RDKit Geometry ===")

# Deduplicate by SMILES for geometry computation
unique_smiles = set(p["replacement_smiles"] for p in all_proposals)
print(f"Unique SMILES to compute geometry: {len(unique_smiles)}")

# Geometry cache: smiles -> geometry results
geometry_cache = {}
geo_success = 0
geo_fail = 0
geo_failure_records = []

t0 = time.time()
for i, smi in enumerate(sorted(unique_smiles)):
    geo_result = run_fragment_geometry(smi, n_conformers=10, seed=SEED)
    geometry_cache[smi] = geo_result

    if geo_result["geometry_success"]:
        geo_success += 1
    else:
        geo_fail += 1
        geo_failure_records.append(geo_result)

    # Progress + checkpoint after first 500
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        success_rate = geo_success / (i + 1) * 100
        print(f"  Geometry: {i+1}/{len(unique_smiles)} processed ({elapsed:.1f}s), success rate: {success_rate:.1f}%")

        if i + 1 == 500:
            if success_rate < 50:
                print(f"  *** WARNING: geometry_success_rate < 50% ({success_rate:.1f}%) ***")
                print(f"  *** Fragment-level geometry may have issues with some SMILES patterns ***")

elapsed = time.time() - t0
print(f"Geometry complete: {geo_success} success, {geo_fail} failed, {elapsed:.1f}s total")
print(f"  Overall success rate: {100*geo_success/(geo_success+geo_fail):.1f}%")

# Build geometry lookup for proposals
geo_lookup = {}
for smi, gr in geometry_cache.items():
    geo_lookup[smi] = {
        "geometry_success": int(gr["geometry_success"]),
        "num_conformers": gr["num_conformers"],
        "num_mmff_optimized": gr["num_mmff_optimized"],
        "mmff_success": int(gr["mmff_success"]),
        "uff_fallback": int(gr["uff_fallback"]),
        "selected_energy": gr["selected_energy"],
        "error_message": gr["error_message"] or "",
    }

# Save geometry results (per proposal)
geo_rows = []
for p in all_proposals:
    gl = geo_lookup.get(p["replacement_smiles"], {})
    geo_rows.append({
        "query_id": p["query_id"],
        "method": p["method"],
        "rank": p["rank"],
        "replacement_smiles": p["replacement_smiles"],
        "old_fragment": p.get("old_fragment", ""),
        "is_exact_positive": p["is_exact_positive"],
        "geometry_success": gl.get("geometry_success", 0),
        "num_conformers": gl.get("num_conformers", 0),
        "num_mmff_optimized": gl.get("num_mmff_optimized", 0),
        "mmff_success": gl.get("mmff_success", 0),
        "uff_fallback": gl.get("uff_fallback", 0),
        "selected_energy": gl.get("selected_energy"),
        "error_message": gl.get("error_message", ""),
    })

geo_df = pd.DataFrame(geo_rows)
geo_df.to_csv(OUTDIR / "d4a3_rdkit_geometry_results.csv", index=False)
print(f"Saved d4a3_rdkit_geometry_results.csv ({len(geo_df)} rows)")

# Save failure cases
if geo_failure_records:
    fail_df = pd.DataFrame(geo_failure_records)
    fail_df.to_csv(OUTDIR / "d4a3_geometry_failure_cases.csv", index=False)
    print(f"Saved d4a3_geometry_failure_cases.csv ({len(fail_df)} rows)")

# ============================================================
# PART D: A4C Review
# ============================================================
print("\n=== Part D: A4C Review ===")

# Initialize PAINS/Brenk filter catalog
catalog = None
try:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(params)
    print("  PAINS/Brenk catalogs loaded")
except Exception as e:
    print(f"  Warning: FilterCatalog init failed: {e}")

review_rows = []
bucket_counts = collections.Counter()

def compute_molecular_properties(smiles):
    """Compute molecular properties for a SMILES string (with *)."""
    h_smi = hcap_smiles(smiles)
    mol = Chem.MolFromSmiles(h_smi)
    if mol is None:
        return None
    try:
        return {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol),
        }
    except:
        return None

def check_alerts(smiles, catalog):
    """Check PAINS/Brenk alerts for a SMILES."""
    h_smi = hcap_smiles(smiles)
    mol = Chem.MolFromSmiles(h_smi)
    if mol is None or catalog is None:
        return {"pains_alerts": 0, "brenk_alerts": 0}
    try:
        cat_entries = catalog.GetMatches(mol)
        pains = 0
        brenk = 0
        for entry in cat_entries:
            desc = entry.GetDescription()
            if "PAINS" in desc.upper():
                pains += 1
            else:
                brenk += 1
        return {"pains_alerts": pains, "brenk_alerts": brenk}
    except:
        return {"pains_alerts": 0, "brenk_alerts": 0}

# Compute properties for all unique SMILES (old fragments and replacements)
smiles_props = {}
smiles_alerts = {}
all_smiles_needed = set()
for p in all_proposals:
    all_smiles_needed.add(p["replacement_smiles"])
    all_smiles_needed.add(p.get("old_fragment", ""))

for smi in all_smiles_needed:
    if smi:
        smiles_props[smi] = compute_molecular_properties(smi)
        smiles_alerts[smi] = check_alerts(smi, catalog)

n_a4c = 0
n_review_ready = 0
n_hard_reject = 0
n_chem_alert = 0
n_prop_extreme = 0
n_prop_warn = 0

for p in all_proposals:
    qid, smi, method, rank = p["query_id"], p["replacement_smiles"], p["method"], p["rank"]
    old_smi = p.get("old_fragment", "")
    gl = geo_lookup.get(smi, {})

    # Geometry checks
    geo_success = gl.get("geometry_success", 0)
    n_confs = gl.get("num_conformers", 0)

    hard_geometry_reject = (not geo_success) or (n_confs < 3)

    # Chemistry alerts
    alerts = smiles_alerts.get(smi, {"pains_alerts": 0, "brenk_alerts": 0})
    has_alert = (alerts["pains_alerts"] > 0) or (alerts["brenk_alerts"] > 0)

    # Property deltas
    old_props = smiles_props.get(old_smi, {})
    new_props = smiles_props.get(smi, {})

    prop_deltas = {}
    if old_props and new_props:
        for key in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]:
            old_val = old_props.get(key, 0) or 0
            new_val = new_props.get(key, 0) or 0
            prop_deltas[f"delta_{key}"] = new_val - old_val
    else:
        for key in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]:
            prop_deltas[f"delta_{key}"] = None

    # Property shift classification
    delta_mw = abs(prop_deltas.get("delta_MW", 0) or 0)
    delta_logp = abs(prop_deltas.get("delta_LogP", 0) or 0)
    delta_tpsa = abs(prop_deltas.get("delta_TPSA", 0) or 0)

    prop_extreme = delta_mw > 100 or delta_logp > 3 or delta_tpsa > 80
    prop_warning = delta_mw > 50 or delta_logp > 1.5 or delta_tpsa > 40

    # A4C bucket classification
    if hard_geometry_reject:
        bucket = "HARD_GEOMETRY_REJECT"
        n_hard_reject += 1
    elif has_alert:
        bucket = "HARD_CHEMISTRY_ALERT"
        n_chem_alert += 1
    elif prop_extreme:
        bucket = "PROPERTY_SHIFT_WARNING"
        n_prop_extreme += 1
    elif prop_warning:
        bucket = "REVIEW_READY_WITH_WARNING"
        n_prop_warn += 1
    else:
        bucket = "REVIEW_READY"
        n_review_ready += 1

    bucket_counts[bucket] += 1
    n_a4c += 1

    review_rows.append({
        "query_id": qid,
        "method": method,
        "rank": rank,
        "replacement_smiles": smi,
        "old_fragment": old_smi,
        "is_exact_positive": p["is_exact_positive"],
        "geometry_success": geo_success,
        "num_conformers": n_confs,
        "pains_alerts": alerts["pains_alerts"],
        "brenk_alerts": alerts["brenk_alerts"],
        "has_alert": int(has_alert),
        **prop_deltas,
        "hard_geometry_reject": int(hard_geometry_reject),
        "hard_chemistry_alert": int(has_alert),
        "property_shift_extreme": int(prop_extreme),
        "property_shift_warning": int(prop_warning),
        "a4c_bucket": bucket,
    })

review_df = pd.DataFrame(review_rows)
review_df.to_csv(OUTDIR / "d4a3_a4c_review_results.csv", index=False)
print(f"\nSaved d4a3_a4c_review_results.csv ({len(review_df)} rows)")

# Bucket distribution
print("\nA4C Bucket Distribution:")
for bucket, count in sorted(bucket_counts.items()):
    pct = 100 * count / max(n_a4c, 1)
    print(f"  {bucket}: {count} ({pct:.1f}%)")

# Save bucket distribution
bucket_rows = [{"bucket": k, "count": v, "percentage": 100*v/max(n_a4c, 1)}
               for k, v in sorted(bucket_counts.items())]
bucket_df = pd.DataFrame(bucket_rows)
bucket_df.to_csv(OUTDIR / "d4a3_a4c_bucket_distribution.csv", index=False)
print(f"Saved d4a3_a4c_bucket_distribution.csv")

# DISCRIMINATION CHECK
total_review_ready = bucket_counts.get("REVIEW_READY", 0) + bucket_counts.get("REVIEW_READY_WITH_WARNING", 0)
review_ready_pct = 100 * total_review_ready / max(n_a4c, 1)
print(f"\nDISCRIMINATION CHECK:")
print(f"  REVIEW_READY + REVIEW_READY_WITH_WARNING: {total_review_ready}/{n_a4c} ({review_ready_pct:.1f}%)")

if review_ready_pct > 95:
    print("  *** A4C_NON_DISCRIMINATIVE = True ***")
    print("  *** Provisional rules too lenient; need stricter thresholds or v1B final rules ***")
    with open(OUTDIR / "d4a3_a4c_non_discriminative.flag", "w") as f:
        f.write(f"A4C_NON_DISCRIMINATIVE=True\n")
        f.write(f"REVIEW_READY_pct={review_ready_pct:.1f}%\n")
        f.write(f"Note: Provisional rules too lenient; need stricter thresholds or v1B final rules\n")
else:
    print("  A4C_NON_DISCRIMINATIVE = False (discriminative enough)")

print(f"\n=== Script 1 Complete ===")
print(f"Output files:")
print(f"  {OUTDIR / 'd4a3_eval_query_set.csv'}")
print(f"  {OUTDIR / 'd4a3_topk_proposals.jsonl'}")
print(f"  {OUTDIR / 'd4a3_rdkit_geometry_results.csv'}")
print(f"  {OUTDIR / 'd4a3_geometry_failure_cases.csv'}")
print(f"  {OUTDIR / 'd4a3_a4c_review_results.csv'}")
print(f"  {OUTDIR / 'd4a3_a4c_bucket_distribution.csv'}")
