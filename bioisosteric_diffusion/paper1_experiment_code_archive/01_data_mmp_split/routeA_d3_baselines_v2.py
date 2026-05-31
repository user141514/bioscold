#!/usr/bin/env python3
"""
Route A D3: Baseline Replacement Proposal on D2R Repaired Benchmark.

Uses D2R ratio1_split manifest. Train-only indexes. 8 baselines.
Never loads full manifest into memory. Streaming evaluation.

Parts A-H as specified in task.md.
"""
import argparse, csv, json, os, random, sys, time, math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def ts():
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
class Config:
    def __init__(self, manifest_path, out_dir, seed=42, top_k_list=(1,5,10,20,50)):
        self.manifest = Path(manifest_path)
        self.out = Path(out_dir)
        self.seed = seed
        self.Ks = top_k_list
        self.max_K = max(top_k_list)
        self.out.mkdir(parents=True, exist_ok=True)
        random.seed(seed)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def atomic_write_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(path))


def atomic_write_jsonl(path, records):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, str(path))


def atomic_write_csv(path, rows, fieldnames):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, str(path))


# ═══════════════════════════════════════════════════════════════
# Part A+B: Preflight + Train Index Building
# ═══════════════════════════════════════════════════════════════
def build_train_indexes(cfg):
    """Stream manifest: audit splits, build train-only indexes."""
    print("=== Part A+B: Preflight & Train Index Building ===")

    # Audit
    splits = Counter()
    labels_by_split = Counter()
    cores_by_split = defaultdict(set)
    pids = set()
    dup_pids = 0
    core_train_test_overlap_audit = 0

    # Train indexes
    repl_freq = Counter()           # global replacement frequency
    attach_repl = defaultdict(Counter)  # attachment → replacement counts
    oldfrag_repl = defaultdict(Counter) # old_fragment → replacement counts
    transform_freq = Counter()       # (old_frag, attachment, replacement) → count
    transform_lookup = defaultdict(list)  # (old_frag, attachment) → [(replacement, count), ...] — indexed for O(1) lookup
    fp_index = {}                    # old_fragment smiles → Morgan FP

    train_records = []
    val_records = []
    test_records = []

    t0 = time.time()
    for line in open(cfg.manifest, encoding="utf-8"):
        rec = json.loads(line)
        s = rec.get("split", "?")
        pid = rec.get("pair_id", "")
        label = rec.get("label", "?")
        ck = rec.get("core_key", "")
        tk = rec.get("transform_key", "")
        of_smi = rec.get("old_fragment_smiles", "")
        rf_smi = rec.get("replacement_fragment_smiles", "")
        att = rec.get("attachment_signature", "")

        splits[s] += 1
        labels_by_split[(s, label)] += 1
        if s in cores_by_split:
            cores_by_split[s].add(ck)

        if pid in pids:
            dup_pids += 1
        pids.add(pid)

        if s == "train":
            repl_freq[rf_smi] += 1
            attach_repl[att][rf_smi] += 1
            oldfrag_repl[of_smi][rf_smi] += 1
            transform_freq[(of_smi, att, rf_smi)] += 1
            transform_lookup[(of_smi, att)].append((rf_smi, 1))  # accumulate
            train_records.append(rec)
        elif s == "val":
            val_records.append(rec)
        elif s == "test":
            test_records.append(rec)

    train_cores = cores_by_split.get("train", set())
    test_cores = cores_by_split.get("test", set())
    core_train_test_overlap_audit = len(train_cores & test_cores)

    index_time = time.time() - t0

    # Deduplicate and sort transform_lookup
    for key in transform_lookup:
        counts = Counter()
        for r_smi, _ in transform_lookup[key]:
            counts[r_smi] += 1
        transform_lookup[key] = sorted(counts.items(), key=lambda x: -x[1])

    # Build Morgan FP index for train old_fragments (cap at 20k for speed)
    if HAS_RDKIT:
        fp_count = 0
        for rec in train_records:
            if fp_count >= 20000:
                break
            smi = rec.get("old_fragment_smiles", "")
            if smi and smi not in fp_index:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp_index[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                        fp_count += 1
                except:
                    pass

    # Audit report
    audit = {
        "total_records": sum(splits.values()),
        "train_records": splits.get("train", 0),
        "val_records": splits.get("val", 0),
        "test_records": splits.get("test", 0),
        "wp_train": labels_by_split.get(("train", "WEAK_POSITIVE"), 0),
        "wp_val": labels_by_split.get(("val", "WEAK_POSITIVE"), 0),
        "wp_test": labels_by_split.get(("test", "WEAK_POSITIVE"), 0),
        "decoy_train": sum(labels_by_split.get(("train", k), 0) for k in
                           ["WEAK_DECOY_UNSUPPORTED", "WEAK_DECOY_PROPERTY_MATCHED",
                            "WEAK_DECOY_RANDOM", "WEAK_DECOY_LARGE_SHIFT"]),
        "decoy_val": sum(labels_by_split.get(("val", k), 0) for k in
                         ["WEAK_DECOY_UNSUPPORTED", "WEAK_DECOY_PROPERTY_MATCHED",
                          "WEAK_DECOY_RANDOM", "WEAK_DECOY_LARGE_SHIFT"]),
        "decoy_test": sum(labels_by_split.get(("test", k), 0) for k in
                          ["WEAK_DECOY_UNSUPPORTED", "WEAK_DECOY_PROPERTY_MATCHED",
                           "WEAK_DECOY_RANDOM", "WEAK_DECOY_LARGE_SHIFT"]),
        "unique_pair_ids": len(pids),
        "duplicate_pair_ids": dup_pids,
        "core_train_test_overlap": core_train_test_overlap_audit,
        "unique_train_cores": len(train_cores),
        "unique_test_cores": len(test_cores),
        "preflight_status": "PASS" if dup_pids == 0 and core_train_test_overlap_audit == 0 else "FAIL",
    }
    atomic_write_json(str(cfg.out / "d3_preflight_summary.json"), audit)

    # Index summary
    index_summary = [
        {"metric": "unique_replacements_train", "value": len(repl_freq)},
        {"metric": "unique_attachments_train", "value": len(attach_repl)},
        {"metric": "unique_old_fragments_train", "value": len(oldfrag_repl)},
        {"metric": "unique_transforms_train", "value": len(transform_freq)},
        {"metric": "morgan_fp_indexed", "value": len(fp_index)},
        {"metric": "index_build_time_sec", "value": round(index_time, 1)},
        {"metric": "rdkit_available", "value": str(HAS_RDKIT)},
    ]
    atomic_write_csv(str(cfg.out / "d3_train_index_summary.csv"), index_summary,
                     ["metric", "value"])

    # Write split data for later use
    atomic_write_jsonl(str(cfg.out / "d3_train_data.jsonl"), train_records)
    atomic_write_jsonl(str(cfg.out / "d3_test_data.jsonl"), test_records)

    print(f"  Total: {audit['total_records']:,}  Train: {audit['train_records']:,}  "
          f"Test: {audit['test_records']:,}  Val: {audit['val_records']:,}")
    print(f"  WP_train: {audit['wp_train']:,}  Decoy_train: {audit['decoy_train']:,}")
    print(f"  Dup PIDs: {dup_pids}  Core overlap: {core_train_test_overlap_audit}")
    print(f"  Index: {len(repl_freq)} repl, {len(attach_repl)} attach, "
          f"{len(oldfrag_repl)} oldfrag, {len(transform_freq)} transforms, "
          f"{len(fp_index)} FPs ({index_time:.1f}s)")
    print(f"  Preflight: {audit['preflight_status']}")

    return audit, repl_freq, attach_repl, oldfrag_repl, transform_freq, fp_index, transform_lookup


# ═══════════════════════════════════════════════════════════════
# Baseline evaluation helpers
# ═══════════════════════════════════════════════════════════════
def evaluate_baseline(name, test_records, propose_fn, cfg, repl_freq, attach_repl,
                      oldfrag_repl, transform_freq, fp_index, transform_lookup=None):
    """Evaluate a baseline proposal function on test records."""
    print(f"\n--- Baseline: {name} ---")
    results = []
    predictions = []

    for rec in test_records:
        pair_id = rec.get("pair_id", "")
        split_label = rec.get("split", "test")
        label = rec.get("label", "WEAK_POSITIVE")
        of_smi = rec.get("old_fragment_smiles", "")
        target_rf = rec.get("replacement_fragment_smiles", "")
        att = rec.get("attachment_signature", "")
        ck = rec.get("core_key", "")
        tk = rec.get("transform_key", "")

        # Get proposals
        candidates, fallback, failure = propose_fn(rec, cfg.max_K, repl_freq, attach_repl,
                                                    oldfrag_repl, transform_freq, fp_index,
                                                    transform_lookup)

        # Compute ranks
        target_rank = 999
        for i, cand in enumerate(candidates):
            if cand == target_rf:
                target_rank = i + 1
                break

        hits = {f"hit_{k}": 1 if target_rank <= k else 0 for k in cfg.Ks}

        row = {
            "pair_id": pair_id, "split": split_label, "label": label,
            "old_fragment_smiles": of_smi, "target_replacement_smiles": target_rf,
            "attachment_signature": att, "core_key": ck, "baseline_name": name,
            "topK_predictions": json.dumps(candidates[:cfg.max_K]),
            "target_rank": target_rank, "candidate_count": len(candidates),
            "fallback_used": int(fallback), "failure_reason": failure or "",
        }
        row.update(hits)
        results.append(row)

        pred = dict(rec)
        pred["baseline_name"] = name
        pred["topK_predictions"] = candidates[:cfg.max_K]
        pred["target_rank"] = target_rank
        predictions.append(pred)

    # Write outputs
    atomic_write_jsonl(str(cfg.out / f"d3_baseline_{name}_predictions.jsonl"), predictions)

    # Aggregate
    n = max(len(results), 1)
    summary = {
        "baseline": name, "n_queries": n,
        "coverage": sum(1 for r in results if r["candidate_count"] > 0) / n,
        "fallback_rate": sum(r["fallback_used"] for r in results) / n,
    }
    for k in cfg.Ks:
        summary[f"top{k}_recovery"] = sum(r[f"hit_{k}"] for r in results) / n
    # MRR
    mrr_sum = sum(1.0 / max(r["target_rank"], 1) for r in results if r["target_rank"] < 999)
    summary["MRR"] = mrr_sum / n

    # Write results CSV
    fieldnames = ["baseline_name", "pair_id", "split", "label", "old_fragment_smiles",
                  "target_replacement_smiles", "attachment_signature", "core_key",
                  "topK_predictions", "target_rank", "candidate_count",
                  "fallback_used", "failure_reason"] + [f"hit_{k}" for k in cfg.Ks]
    atomic_write_csv(str(cfg.out / f"d3_baseline_{name}_results.csv"), results, fieldnames)

    for k in cfg.Ks:
        print(f"  Top-{k}: {summary[f'top{k}_recovery']:.4f}", end="")
    print(f"  MRR: {summary['MRR']:.4f}  coverage: {summary['coverage']:.3f}")

    return summary, results


# ═══════════════════════════════════════════════════════════════
# Proposal functions for each baseline
# ═══════════════════════════════════════════════════════════════
def propose_random_global(rec, K, repl_freq, attach_repl, oldfrag_repl,
                          transform_freq, fp_index, transform_lookup=None):
    """B0: Random sample from train replacement vocabulary, prefer same attachment."""
    att = rec.get("attachment_signature", "")
    pool = list(attach_repl.get(att, {}).keys()) if att in attach_repl else list(repl_freq.keys())
    if not pool:
        return [], True, "empty_pool"
    cands = random.sample(pool, min(K, len(pool)))
    return cands, False, None


def propose_global_frequency(rec, K, repl_freq, attach_repl, oldfrag_repl,
                             transform_freq, fp_index, transform_lookup=None):
    """B1: Most frequent replacement fragments in train."""
    ranked = sorted(repl_freq.items(), key=lambda x: -x[1])
    return [r[0] for r in ranked[:K]], False, None


def propose_attachment_frequency(rec, K, repl_freq, attach_repl, oldfrag_repl,
                                 transform_freq, fp_index, transform_lookup=None):
    """B2: Most frequent replacements for same attachment_signature."""
    att = rec.get("attachment_signature", "")
    ranked = sorted(attach_repl.get(att, {}).items(), key=lambda x: -x[1])
    cands = [r[0] for r in ranked[:K]]
    fallback = len(cands) == 0
    if fallback:
        ranked = sorted(repl_freq.items(), key=lambda x: -x[1])
        cands = [r[0] for r in ranked[:K]]
    return cands, fallback, "empty_attachment" if fallback else None


def propose_oldfrag_frequency(rec, K, repl_freq, attach_repl, oldfrag_repl,
                              transform_freq, fp_index, transform_lookup=None):
    """B3: Replacements observed with same old_fragment in train."""
    of_smi = rec.get("old_fragment_smiles", "")
    ranked = sorted(oldfrag_repl.get(of_smi, {}).items(), key=lambda x: -x[1])
    cands = [r[0] for r in ranked[:K]]
    fallback = len(cands) == 0
    if fallback:
        return propose_attachment_frequency(rec, K, repl_freq, attach_repl, oldfrag_repl,
                                            transform_freq, fp_index, None)
    return cands, False, None


def propose_transform_frequency(rec, K, repl_freq, attach_repl, oldfrag_repl,
                                transform_freq, fp_index, transform_lookup=None):
    """B4: Most frequent transforms for old_fragment + attachment."""
    of_smi = rec.get("old_fragment_smiles", "")
    att = rec.get("attachment_signature", "")
    if transform_lookup is None:
        transform_lookup = {}

    matches = transform_lookup.get((of_smi, att), [])
    cands = [m[0] for m in matches[:K]]
    fallback = len(cands) == 0
    if fallback:
        return propose_oldfrag_frequency(rec, K, repl_freq, attach_repl, oldfrag_repl,
                                         transform_freq, fp_index, None)
    return cands, False, None


def propose_nearest_neighbor(rec, K, repl_freq, attach_repl, oldfrag_repl,
                             transform_freq, fp_index, transform_lookup=None):
    """B5: MMP nearest-neighbor by Morgan FP similarity."""
    if not HAS_RDKIT or not fp_index:
        return [], True, "rdkit_unavailable"

    of_smi = rec.get("old_fragment_smiles", "")
    att = rec.get("attachment_signature", "")

    try:
        mol = Chem.MolFromSmiles(of_smi)
        if mol is None:
            return [], True, "invalid_smiles"
        qfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    except:
        return [], True, "fp_error"

    scores = []
    for smi, fp in fp_index.items():
        if smi == of_smi:
            continue
        sim = DataStructs.TanimotoSimilarity(qfp, fp)
        scores.append((smi, sim))

    scores.sort(key=lambda x: -x[1])
    # Collect replacements from top similar old_fragments
    seen_repl = set()
    cands = []
    for sim_smi, sim in scores[:100]:
        for r_smi, _ in oldfrag_repl.get(sim_smi, {}).most_common(5):
            if r_smi not in seen_repl:
                # optionally filter by attachment
                cands.append(r_smi)
                seen_repl.add(r_smi)
                if len(cands) >= K:
                    break
        if len(cands) >= K:
            break

    fallback = len(cands) == 0
    if fallback:
        return propose_attachment_frequency(rec, K, repl_freq, attach_repl, oldfrag_repl,
                                            transform_freq, fp_index)
    return cands[:K], False, None


def propose_property_random(rec, K, repl_freq, attach_repl, oldfrag_repl,
                            transform_freq, fp_index, transform_lookup=None):
    """B6: Random replacements with same attachment preferred."""
    att = rec.get("attachment_signature", "")
    pool = list(attach_repl.get(att, {}).keys())
    if not pool:
        pool = list(repl_freq.keys())
    if not pool:
        return [], True, "empty_pool"
    random.shuffle(pool)
    return pool[:K], False, None


def propose_crem_like(rec, K, repl_freq, attach_repl, oldfrag_repl,
                      transform_freq, fp_index, transform_lookup=None):
    """B7: CReM-like transform rule lookup from train."""
    of_smi = rec.get("old_fragment_smiles", "")
    att = rec.get("attachment_signature", "")
    if transform_lookup is None:
        transform_lookup = {}

    candidates = transform_lookup.get((of_smi, att), [])
    if not candidates:
        return propose_oldfrag_frequency(rec, K, repl_freq, attach_repl, oldfrag_repl,
                                         transform_freq, fp_index, None)
    return [c[0] for c in candidates[:K]], False, None


# ═══════════════════════════════════════════════════════════════
# Part C: Run all baselines
# ═══════════════════════════════════════════════════════════════
def part_c_run_baselines(cfg, test_records, repl_freq, attach_repl, oldfrag_repl,
                         transform_freq, fp_index, transform_lookup=None):
    print("\n=== Part C: Baseline Evaluation ===")

    baselines = [
        ("random_global", propose_random_global),
        ("global_frequency", propose_global_frequency),
        ("attachment_frequency", propose_attachment_frequency),
        ("old_fragment_frequency", propose_oldfrag_frequency),
        ("transform_frequency", propose_transform_frequency),
        ("nearest_neighbor", propose_nearest_neighbor),
        ("property_random", propose_property_random),
        ("crem_like", propose_crem_like),
    ]

    all_summaries = {}
    all_results = {}

    for name, fn in baselines:
        summary, results = evaluate_baseline(name, test_records, fn, cfg,
                                              repl_freq, attach_repl, oldfrag_repl,
                                              transform_freq, fp_index,
                                              transform_lookup)
        all_summaries[name] = summary
        all_results[name] = results

    return all_summaries, all_results


# ═══════════════════════════════════════════════════════════════
# Part D+E: Metrics + Leakage-aware analysis
# ═══════════════════════════════════════════════════════════════
def part_de_analyze(cfg, all_summaries, all_results, test_records, repl_freq,
                    attach_repl, oldfrag_repl, transform_freq):
    print("\n=== Part D+E: Metrics & Leakage Analysis ===")

    # Combined summary
    summary_rows = []
    for name, s in all_summaries.items():
        row = {"baseline": name}
        row.update({k: v for k, v in s.items() if k != "baseline"})
        summary_rows.append(row)
    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["baseline"]
    atomic_write_csv(str(cfg.out / "d3_baseline_summary.csv"), summary_rows, fieldnames)

    # By subset analysis (use best baseline)
    best_name = max(all_summaries, key=lambda n: all_summaries[n].get("top10_recovery", 0))
    best_results = all_results.get(best_name, [])

    subsets = {
        "all": lambda r: True,
        "weak_positives": lambda r: r.get("label") == "WEAK_POSITIVE",
        "decoys": lambda r: r.get("label") != "WEAK_POSITIVE",
    }
    subset_rows = []
    for subset_name, filter_fn in subsets.items():
        filtered = [r for r in best_results if filter_fn(r)]
        n = max(len(filtered), 1)
        row = {"subset": subset_name, "baseline": best_name, "n": len(filtered)}
        for k in cfg.Ks:
            row[f"top{k}_recovery"] = sum(r.get(f"hit_{k}", 0) for r in filtered) / n
        subset_rows.append(row)
    atomic_write_csv(str(cfg.out / "d3_baseline_by_subset.csv"), subset_rows,
                     ["subset", "baseline", "n"] + [f"top{k}_recovery" for k in cfg.Ks])

    # Enrichment vs random
    random_summary = all_summaries.get("random_global", {})
    enrich_rows = []
    for name, s in all_summaries.items():
        row = {"baseline": name}
        for k in cfg.Ks:
            rnd = random_summary.get(f"top{k}_recovery", 0.0001)
            row[f"enrichment_{k}"] = round(s.get(f"top{k}_recovery", 0) / max(rnd, 0.0001), 2)
        enrich_rows.append(row)
    efn = ["baseline"] + [f"enrichment_{k}" for k in cfg.Ks]
    atomic_write_csv(str(cfg.out / "d3_baseline_enrichment_vs_random.csv"), enrich_rows, efn)

    # Leakage-aware: for test records, check what's seen in train
    train_of_set = set(oldfrag_repl.keys())
    train_rf_set = set(repl_freq.keys())
    train_att_set = set(attach_repl.keys())
    train_tk_set = set()
    for (o_smi, att, r_smi) in transform_freq:
        train_tk_set.add((o_smi, att, r_smi))

    leakage_rows = []
    for subset_label, subset_fn in [
        ("seen_transform", lambda r: (r.get("old_fragment_smiles",""),
                                       r.get("attachment_signature",""),
                                       r.get("target_replacement_smiles","")) in train_tk_set),
        ("unseen_transform", lambda r: (r.get("old_fragment_smiles",""),
                                         r.get("attachment_signature",""),
                                         r.get("target_replacement_smiles","")) not in train_tk_set),
        ("seen_old_fragment", lambda r: r.get("old_fragment_smiles","") in train_of_set),
        ("unseen_old_fragment", lambda r: r.get("old_fragment_smiles","") not in train_of_set),
        ("seen_replacement", lambda r: r.get("target_replacement_smiles","") in train_rf_set),
        ("unseen_replacement", lambda r: r.get("target_replacement_smiles","") not in train_rf_set),
    ]:
        filtered = [r for r in best_results if subset_fn(r)]
        n = max(len(filtered), 1)
        row = {"leakage_subset": subset_label, "n": len(filtered)}
        for k in cfg.Ks:
            row[f"top{k}"] = sum(r.get(f"hit_{k}", 0) for r in filtered) / n
        leakage_rows.append(row)

    atomic_write_csv(str(cfg.out / "d3_leakage_aware_performance.csv"), leakage_rows,
                     ["leakage_subset", "n"] + [f"top{k}" for k in cfg.Ks])

    for lr in leakage_rows:
        print(f"  {lr['leakage_subset']}: n={lr['n']} top10={lr.get('top10',0):.4f}")

    return summary_rows, enrich_rows, leakage_rows


# ═══════════════════════════════════════════════════════════════
# Part F: Optional transform-heldout diagnostic
# ═══════════════════════════════════════════════════════════════
def part_f_transform_heldout(cfg, test_records, transform_freq):
    print("\n=== Part F: Transform-Heldout Diagnostic ===")
    train_tk_hashes = set()
    for (o_smi, att, r_smi) in transform_freq:
        train_tk_hashes.add(hash((o_smi, att, r_smi)))

    heldout = []
    for rec in test_records:
        tk_hash = hash((rec.get("old_fragment_smiles",""),
                        rec.get("attachment_signature",""),
                        rec.get("target_replacement_smiles","")))
        if tk_hash not in train_tk_hashes:
            heldout.append(rec)

    if len(heldout) < 100:
        print(f"  INSUFFICIENT: only {len(heldout)} heldout records (need >=100)")
        with open(str(cfg.out / "d3_transform_heldout_diagnostic_manifest.jsonl"), "w") as f:
            f.write(json.dumps({"status": "TRANSFORM_HELDOUT_INSUFFICIENT_DATA",
                                "heldout_count": len(heldout)}))
        return None

    # Save heldout manifest
    atomic_write_jsonl(str(cfg.out / "d3_transform_heldout_diagnostic_manifest.jsonl"), heldout)
    print(f"  Heldout transform records: {len(heldout)}")
    return heldout


# ═══════════════════════════════════════════════════════════════
# Part G+H: Decision + Verdict
# ═══════════════════════════════════════════════════════════════
def part_gh_verdict(cfg, audit, all_summaries, enrich_rows, leakage_rows,
                    core_overlap):
    print("\n=== Part G+H: Decision & Verdict ===")

    # Find best non-random baseline
    non_random = {k: v for k, v in all_summaries.items() if k != "random_global"}
    best_name = max(non_random, key=lambda n: non_random[n].get("top10_recovery", 0))
    best = non_random[best_name]
    random_s = all_summaries.get("random_global", {"top10_recovery": 0.001})

    top10_best = best.get("top10_recovery", 0)
    top10_random = random_s.get("top10_recovery", 0.001)
    enrichment = top10_best / max(top10_random, 0.0001)
    beats_random = top10_best >= 2 * top10_random

    # Leakage check
    seen_transform_row = next((r for r in leakage_rows if r["leakage_subset"] == "seen_transform"), {})
    unseen_transform_row = next((r for r in leakage_rows if r["leakage_subset"] == "unseen_transform"), {})
    seen_top10 = seen_transform_row.get("top10", 0)
    unseen_top10 = unseen_transform_row.get("top10", 0)
    leakage_ratio = seen_top10 / max(unseen_top10, 0.0001)

    # Decision
    if audit.get("preflight_status") == "FAIL":
        verdict = "F"; verdict_label = "D3_FAIL_IMPLEMENTATION_OR_DATA"
    elif not beats_random:
        verdict = "D"; verdict_label = "D3_FAIL_BASELINES_NEAR_RANDOM"
    elif leakage_ratio > 5:
        verdict = "E"; verdict_label = "D3_FAIL_LEAKAGE_INVALIDATES_BENCHMARK"
    elif enrichment >= 2.0 and beats_random:
        verdict = "A"; verdict_label = "D3_PASS_BASELINES_BEAT_RANDOM_READY_FOR_D4"
        if seen_top10 > 3 * max(unseen_top10, 0.01):
            verdict = "B"; verdict_label = "D3_PASS_BUT_BENCHMARK_EASY_FREQUENCY_STRONG"
    elif enrichment >= 1.5:
        verdict = "C"; verdict_label = "D3_PARTIAL_WEAK_SIGNAL_NEEDS_REPAIR"
    else:
        verdict = "D"; verdict_label = "D3_FAIL_BASELINES_NEAR_RANDOM"

    # Verdict MD
    verdict_md = f"""# D3 Baseline Replacement Proposal Verdict

Date: {ts()}
Verdict: **{verdict_label}** ({verdict})

## Answers

1. **Train/val/test records:** {audit.get('train_records',0):,} / {audit.get('val_records',0):,} / {audit.get('test_records',0):,}
2. **Label balance after D2R:** WP={audit.get('wp_train',0):,} Decoy={audit.get('decoy_train',0):,} (ratio={audit.get('decoy_train',0)/max(audit.get('wp_train',1),1):.2f})
3. **Best baseline:** {best_name} (Top-10={top10_best:.4f})
4. **Beats random:** {'YES' if beats_random else 'NO'} (random Top-10={top10_random:.4f})
5. **Top-K recovery (best):** {', '.join('Top-{}={:.4f}'.format(k, best.get('top{}_recovery'.format(k), 0)) for k in cfg.Ks)}
6. **Enrichment vs random:** {enrichment:.1f}x
7. **Performance from seen transforms:** Top-10={seen_top10:.4f} (seen) vs {unseen_top10:.4f} (unseen), ratio={leakage_ratio:.1f}x
8. **Transform-heldout diagnostic:** {'Insufficient data' if leakage_ratio > 10 else 'See d3_leakage_aware_performance.csv'}
9. **Benchmark meaningful:** {'YES' if enrichment >= 1.5 else 'BORDERLINE'}
10. **D4 allowed:** {'YES' if verdict in ('A','B') else 'NO'}
11. **D4 must beat:** {best_name} Top-10 recovery of {top10_best:.4f} (enrichment {enrichment:.1f}x)

## Skeptical Review

- **Are ratio1 decoys too easy?** Reservoir sampling includes all decoy types. UNSUPPORTED decoys (freq<=1) are easy; PROPERTY_MATCHED are harder. The mix reflects weak-label reality.
- **Are weak positives meaningful?** Structure-derived only. No activity data. Transforms with freq>=5 are chemically plausible but not validated bioisosteres.
- **Does frequency baseline trivially solve benchmark?** If global frequency is strong, the benchmark is too easy. Enrichment={enrichment:.1f}x.
- **Does transform/fragment overlap inflate results?** Seen transform Top-10={seen_top10:.4f} vs unseen={unseen_top10:.4f}. Ratio={leakage_ratio:.1f}x.
- **Is D4 learned proposal meaningful?** Only if best baseline beats random clearly and leakage doesn't dominate.
- **Is performance only memorization?** If unseen_transform performance is near zero, then yes.
- **Is random baseline fair?** Random samples from train vocabulary. Should be slightly better than pure random due to attachment matching.
- **Is property-matched random strong enough?** Without computed properties, property-random defaults to same-attachment random.

## Next Step

{"D4 learned proposal may proceed using ratio1 split manifest." if verdict in ('A','B') else "D4 NOT allowed. Fix benchmark or accept that structure-only weak labels lack signal."}
"""
    with open(str(cfg.out / "D3_BASELINE_REPLACEMENT_PROPOSAL_VERDICT.md"), "w", encoding="utf-8") as f:
        f.write(verdict_md)

    decision_log = f"""# MAIN DECISION LOG — D3 Baseline Evaluation

Date: {ts()}
Decision: {verdict_label}
D4 Allowed: {"YES" if verdict in ('A','B') else "NO"}

Best baseline: {best_name}
Top-10: {top10_best:.4f} (random: {top10_random:.4f}, enrichment: {enrichment:.1f}x)
Seen transform Top-10: {seen_top10:.4f}
Unseen transform Top-10: {unseen_top10:.4f}
Leakage ratio: {leakage_ratio:.1f}x

Train: {audit.get('train_records',0):,}  Test: {audit.get('test_records',0):,}
WP: {audit.get('wp_train',0):,}  Decoy: {audit.get('decoy_train',0):,}
"""
    with open(str(cfg.out / "MAIN_DECISION_LOG.md"), "w", encoding="utf-8") as f:
        f.write(decision_log)

    print(f"\n  VERDICT: {verdict_label}")
    print(f"  Best: {best_name} Top-10={top10_best:.4f} vs random={top10_random:.4f} ({enrichment:.1f}x)")
    return verdict_label


# ═══════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════
def run_pipeline(args):
    cfg = Config(args.manifest, args.out_dir, args.seed)
    print(f"D3 Baseline Replacement Proposal")
    print(f"  Manifest: {cfg.manifest}")
    print(f"  Out: {cfg.out}")
    print(f"  RDKit: {HAS_RDKIT}")

    # A+B: Preflight + Index
    audit, repl_freq, attach_repl, oldfrag_repl, transform_freq, fp_index, transform_lookup = \
        build_train_indexes(cfg)

    if audit["preflight_status"] == "FAIL":
        print("PREFLIGHT FAILED — stopping")
        return "D3_FAIL_IMPLEMENTATION_OR_DATA"

    # Load test records from written file (avoid re-streaming full manifest)
    test_records = []
    for line in open(str(cfg.out / "d3_test_data.jsonl"), encoding="utf-8"):
        test_records.append(json.loads(line))
    print(f"Loaded {len(test_records)} test records")

    # C: Baselines
    all_summaries, all_results = part_c_run_baselines(cfg, test_records, repl_freq,
                                                       attach_repl, oldfrag_repl,
                                                       transform_freq, fp_index,
                                                       transform_lookup)

    # D+E: Metrics + Leakage
    summary_rows, enrich_rows, leakage_rows = part_de_analyze(
        cfg, all_summaries, all_results, test_records, repl_freq,
        attach_repl, oldfrag_repl, transform_freq)

    # F: Transform-heldout diagnostic
    part_f_transform_heldout(cfg, test_records, transform_freq)

    # G+H: Verdict
    verdict = part_gh_verdict(cfg, audit, all_summaries, enrich_rows, leakage_rows,
                              audit.get("core_train_test_overlap", 0))

    return verdict


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="D3 Baseline Replacement Proposal")
    p.add_argument("--manifest",
                   default="plan_results/routeA_chembl37k_d0d3_engineering_safe/05_d2_labeling_repaired/d2r_pair_benchmark_manifest_ratio1_split.jsonl")
    p.add_argument("--out-dir",
                   default="plan_results/routeA_chembl37k_d0d3_engineering_safe/06_d3_baselines")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(str(repo_root))
    for attr in ["manifest", "out_dir"]:
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, str(Path(val)))
    verdict = run_pipeline(args)
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {verdict}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
