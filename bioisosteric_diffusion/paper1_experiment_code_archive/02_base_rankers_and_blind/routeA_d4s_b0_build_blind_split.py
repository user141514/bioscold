#!/usr/bin/env python3
"""Route-A D4S-B0 build secondary blind split and replay inputs."""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260525
SHARD_SIZE = 50_000
random.seed(SEED)
np.random.seed(SEED)

BASE = Path("E:/zuhui/bioisosteric_diffusion")
PLAN = BASE / "plan_results"
OUT = PLAN / "routeA_chembl37k_d4s_b0_blind_split_baseline"
OUT.mkdir(parents=True, exist_ok=True)

SRC_MANIFEST = PLAN / "routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_query_split_manifest.jsonl"
POSITIVE_MANIFEST = PLAN / "routeA_chembl37k_d0d3_engineering_safe/06_d3r_exam_repair/d3r_query_positive_manifest.jsonl"
PHASE1_QUERY = PLAN / "routeA_chembl37k_d4p1_phase1_subset_robustness/d4p1_phase1_query_level_canonical_table.csv"
SPLIT_SUMMARY = PLAN / "routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze/d4a0_split_summary.json"


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def pipe_join(values) -> str:
    seen = []
    marked = set()
    for value in values:
        text = str(value).strip()
        if not text or text in marked:
            continue
        seen.append(text)
        marked.add(text)
    return "|".join(seen)


def pct(series: pd.Series) -> float:
    return float(series.astype(float).mean()) if len(series) else 0.0


def build_input_discovery() -> pd.DataFrame:
    specs = [
        (POSITIVE_MANIFEST, "source_positive_manifest", "required"),
        (SRC_MANIFEST, "source_query_manifest", "required"),
        (PHASE1_QUERY, "old_canonical_test_query_table", "required"),
        (SPLIT_SUMMARY, "old_split_summary", "required"),
        (PLAN / "routeA_chembl37k_d0d3_engineering_safe/06_d3r_exam_repair/d3r_query_split_transform_heldout_primary.jsonl", "historical_primary_split", "optional"),
        (PLAN / "routeA_chembl37k_d4a1_learned_ranker/d4a1_test_predictions.jsonl", "old_hgb_predictions", "optional"),
        (PLAN / "routeA_chembl37k_d4a2d1r_dual_encoder_robustness/d4a2d1r_standardized_predictions.jsonl", "old_de_predictions", "optional"),
        (PLAN / "routeA_chembl37k_d4p1_phase0_metric_lock/d4p1_phase0_canonical_proposal_table.csv", "old_canonical_metric_table", "optional"),
        (PLAN / "routeA_chembl37k_d4s0_sota_opportunity/D4S0_SOTA_OPPORTUNITY_VERDICT.md", "d4s0_verdict", "optional"),
        (BASE / "core/scripts/routeA_d4a0_matrix_freeze.py", "old_split_freeze_script", "optional"),
        (BASE / "core/scripts/routeA_d4a1_train_rankers.py", "old_hgb_script", "optional"),
        (BASE / "core/scripts/routeA_d4a2d1_full_gate.py", "old_de_script", "optional"),
        (BASE / "core/scripts/routeA_d4a2d2_ensemble_minimal.py", "old_borda_script", "optional"),
    ]
    rows = []
    for path, role, req in specs:
        row = {
            "file_path": str(path),
            "role": role,
            "required_or_optional": req,
            "has_query_id": False,
            "has_transform_key": False,
            "has_old_fragment": False,
            "has_attachment_signature": False,
            "has_positive_set": False,
            "has_replacement_vocab": False,
            "has_split": False,
            "status": "FOUND" if path.exists() else "MISSING",
            "notes": "",
        }
        if not path.exists():
            rows.append(row)
            continue
        if path.suffix == ".jsonl":
            first = next(load_jsonl(path))
            keys = set(first.keys())
            row["has_query_id"] = "query_id" in keys
            row["has_transform_key"] = "transform_key_set" in keys or "transform_key" in keys
            row["has_old_fragment"] = "old_fragment_smiles" in keys
            row["has_attachment_signature"] = "attachment_signature" in keys
            row["has_positive_set"] = "positive_replacement_set" in keys
            row["has_split"] = "split" in keys
        elif path.suffix == ".csv":
            head = pd.read_csv(path, nrows=2)
            keys = set(head.columns)
            row["has_query_id"] = "query_id" in keys
            row["has_transform_key"] = "transform_key" in keys or "transform_key_set" in keys
            row["has_old_fragment"] = "old_fragment_smiles" in keys
            row["has_attachment_signature"] = "attachment_signature" in keys
            row["has_positive_set"] = "positive_replacements_exact" in keys or "positive_replacement_set" in keys
            row["has_replacement_vocab"] = "replacement_smiles" in keys
            row["has_split"] = "split" in keys
        else:
            row["notes"] = "script_or_markdown"
        rows.append(row)
    return pd.DataFrame(rows)


def require_columns(df: pd.DataFrame):
    needed = {"query_id", "split", "old_fragment_smiles", "attachment_signature", "positive_replacement_set", "transform_key_set"}
    if not needed.issubset(df.columns):
        raise SystemExit("MISSING_QUERY_OR_TRANSFORM_KEYS")


def parse_list_cell(value):
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    if value is None or str(value).lower() == "nan":
        return []
    text = str(value)
    return [part for part in text.split("|") if part]


def build_components(df: pd.DataFrame) -> pd.DataFrame:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for keys in df["transform_key_set"]:
        keys = parse_list_cell(keys)
        if not keys:
            continue
        base = keys[0]
        parent.setdefault(base, base)
        for key in keys[1:]:
            parent.setdefault(key, key)
            union(base, key)

    df = df.copy()
    df["component_id"] = df["transform_key_set"].apply(lambda xs: find(parse_list_cell(xs)[0]) if parse_list_cell(xs) else "NO_KEY")
    return df


def choose_split(df: pd.DataFrame, canonical_qids: set[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["is_old_canonical_test"] = df["query_id"].isin(canonical_qids)
    df["is_old_split_test"] = df["split"].eq("test")
    comp_rows = []
    comp_groups = {}
    for comp, sub in df.groupby("component_id"):
        comp_groups[comp] = sub
        comp_rows.append(
            {
                "component_id": comp,
                "n_queries": int(len(sub)),
                "has_old_split_test": bool(sub["is_old_split_test"].any()),
                "has_old_canonical_test": bool(sub["is_old_canonical_test"].any()),
            }
        )
    comp_df = pd.DataFrame(comp_rows)
    fixed_train = comp_df.loc[comp_df["has_old_split_test"], "component_id"].tolist()
    eligible = comp_df.loc[~comp_df["has_old_split_test"], ["component_id", "n_queries"]].to_dict("records")
    target_val = int(round(len(df) * 0.10))
    target_blind = int(round(len(df) * 0.10))
    best = None
    for trial in range(256):
        rng = random.Random(SEED + trial)
        order = eligible[:]
        rng.shuffle(order)
        blind, val, train = [], [], []
        n_blind = n_val = 0
        for item in order:
            d_blind = abs((n_blind + item["n_queries"]) - target_blind)
            d_val = abs((n_val + item["n_queries"]) - target_val)
            if n_blind < target_blind and (d_blind <= d_val or n_val >= target_val):
                blind.append(item)
                n_blind += item["n_queries"]
            elif n_val < target_val:
                val.append(item)
                n_val += item["n_queries"]
            else:
                train.append(item)
        for comp in fixed_train:
            train.append({"component_id": comp, "n_queries": int(comp_groups[comp].shape[0])})
        split_map = {}
        for items, split_new in [(train, "train"), (val, "val"), (blind, "blind_test")]:
            for item in items:
                split_map[item["component_id"]] = split_new
        train_mask = df["component_id"].map(split_map).eq("train")
        train_vocab = Counter()
        for repls in df.loc[train_mask, "positive_replacement_set"]:
            for repl in parse_list_cell(repls):
                train_vocab[repl] += 1
        split_stats = {}
        for split_new in ["val", "blind_test"]:
            sub = df.loc[df["component_id"].map(split_map).eq(split_new)]
            any_seen = sub["positive_replacement_set"].apply(lambda xs: any(r in train_vocab for r in parse_list_cell(xs)))
            all_seen = sub["positive_replacement_set"].apply(lambda xs: all(r in train_vocab for r in parse_list_cell(xs)))
            split_stats[split_new] = {
                "n_queries": int(len(sub)),
                "any_seen_rate": pct(any_seen),
                "all_seen_rate": pct(all_seen),
                "n_components": int(sub["component_id"].nunique()),
            }
        score = (
            split_stats["blind_test"]["any_seen_rate"] >= 0.90 and split_stats["val"]["any_seen_rate"] >= 0.90,
            split_stats["blind_test"]["n_queries"] >= 5000,
            -abs(split_stats["blind_test"]["n_queries"] - target_blind) - abs(split_stats["val"]["n_queries"] - target_val),
            round(split_stats["blind_test"]["any_seen_rate"] + split_stats["val"]["any_seen_rate"], 6),
            split_stats["blind_test"]["n_components"] + split_stats["val"]["n_components"],
        )
        if best is None or score > best["score"]:
            best = {"score": score, "split_map": split_map, "train_vocab": train_vocab, "stats": split_stats}
    df["split_new"] = df["component_id"].map(best["split_map"])
    df["target_any_seen_vocab"] = df["positive_replacement_set"].apply(lambda xs: any(r in best["train_vocab"] for r in parse_list_cell(xs)))
    df["target_all_seen_vocab"] = df["positive_replacement_set"].apply(lambda xs: all(r in best["train_vocab"] for r in parse_list_cell(xs)))
    df["source_status"] = np.select(
        [
            df["is_old_canonical_test"],
            df["is_old_split_test"],
            df["split_new"].eq("blind_test"),
            df["split_new"].eq("val"),
        ],
        [
            "old_canonical_test_train_only_quarantine",
            "old_split_test_train_only_quarantine",
            "resplit_blind_eval",
            "resplit_val_selection",
        ],
        default="resplit_train_pool",
    )
    meta = {
        "target_val_queries": target_val,
        "target_blind_queries": target_blind,
        "fixed_train_components": len(fixed_train),
        "best_score": best["score"],
        "blind_stats": best["stats"]["blind_test"],
        "val_stats": best["stats"]["val"],
    }
    return df, meta


def write_quarantine(df: pd.DataFrame, canonical_qids: set[str]):
    old = df.loc[df["query_id"].isin(canonical_qids)].copy()
    old["transform_key"] = old["transform_key_set"].apply(lambda xs: pipe_join(parse_list_cell(xs)))
    old["positive_replacement_set"] = old["positive_replacement_set"].apply(lambda xs: pipe_join(parse_list_cell(xs)))
    cols = ["query_id", "split", "transform_key", "old_fragment_smiles", "attachment_signature", "positive_replacement_set"]
    old[cols].to_csv(OUT / "d4s_b0_old_test_quarantine.csv", index=False)
    summary = {
        "n_old_test_queries": int(len(old)),
        "n_old_test_transform_keys": int(len({key for cell in old["transform_key_set"] for key in parse_list_cell(cell)})),
        "n_old_test_old_fragments": int(old["old_fragment_smiles"].nunique()),
        "n_old_test_replacements": int(len({r for cell in old["positive_replacement_set"] for r in parse_list_cell(cell)})),
    }
    write_json(OUT / "d4s_b0_old_test_quarantine_summary.json", summary)


def write_secondary_manifest(df: pd.DataFrame):
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "query_id": row.query_id,
                "split_new": row.split_new,
                "old_split_if_available": row.split,
                "transform_key": pipe_join(parse_list_cell(row.transform_key_set)),
                "old_fragment_smiles": row.old_fragment_smiles,
                "attachment_signature": row.attachment_signature,
                "positive_replacement_set": parse_list_cell(row.positive_replacement_set),
                "num_positives": int(row.num_positive_replacements),
                "target_any_seen_vocab": bool(row.target_any_seen_vocab),
                "target_all_seen_vocab": bool(row.target_all_seen_vocab),
                "source_status": row.source_status,
            }
        )
    write_jsonl(OUT / "d4s_b0_secondary_split_manifest.jsonl", rows)

    summary_rows = []
    for split_new, sub in df.groupby("split_new"):
        tkeys = {k for cell in sub["transform_key_set"] for k in parse_list_cell(cell)}
        summary_rows.append(
            {
                "split_new": split_new,
                "n_queries": int(len(sub)),
                "n_components": int(sub["component_id"].nunique()),
                "n_transform_keys": int(len(tkeys)),
                "target_any_seen_vocab_n": int(sub["target_any_seen_vocab"].sum()),
                "target_any_seen_vocab_rate": pct(sub["target_any_seen_vocab"]),
                "target_all_seen_vocab_rate": pct(sub["target_all_seen_vocab"]),
                "n_old_split_test_queries": int(sub["is_old_split_test"].sum()),
                "n_old_canonical_test_queries": int(sub["is_old_canonical_test"].sum()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT / "d4s_b0_secondary_split_summary.csv", index=False)


def write_leakage_audit(df: pd.DataFrame):
    transform_sets = {}
    query_sets = {}
    old_fragment_sets = {}
    att_sets = {}
    for split_new, sub in df.groupby("split_new"):
        transform_sets[split_new] = {k for cell in sub["transform_key_set"] for k in parse_list_cell(cell)}
        query_sets[split_new] = set(sub["query_id"])
        old_fragment_sets[split_new] = set(sub["old_fragment_smiles"])
        att_sets[split_new] = set(sub["attachment_signature"])
    blind = df["split_new"].eq("blind_test")
    audit_rows = [
        {"check": "transform_overlap_train_val", "value": len(transform_sets["train"] & transform_sets["val"]), "status": "PASS"},
        {"check": "transform_overlap_train_blind", "value": len(transform_sets["train"] & transform_sets["blind_test"]), "status": "PASS" if len(transform_sets["train"] & transform_sets["blind_test"]) == 0 else "FAIL"},
        {"check": "transform_overlap_val_blind", "value": len(transform_sets["val"] & transform_sets["blind_test"]), "status": "PASS" if len(transform_sets["val"] & transform_sets["blind_test"]) == 0 else "FAIL"},
        {"check": "query_overlap_train_blind", "value": len(query_sets["train"] & query_sets["blind_test"]), "status": "PASS"},
        {"check": "query_overlap_val_blind", "value": len(query_sets["val"] & query_sets["blind_test"]), "status": "PASS"},
        {"check": "old_canonical_test_in_new_blind", "value": int((df["is_old_canonical_test"] & blind).sum()), "status": "PASS" if int((df["is_old_canonical_test"] & blind).sum()) == 0 else "FAIL"},
        {"check": "old_split_test_in_new_blind", "value": int((df["is_old_split_test"] & blind).sum()), "status": "PASS" if int((df["is_old_split_test"] & blind).sum()) == 0 else "FAIL"},
        {"check": "blind_query_count", "value": int(blind.sum()), "status": "PASS" if int(blind.sum()) >= 2000 else "FAIL"},
        {"check": "blind_query_count_preferred", "value": int(blind.sum()), "status": "PASS" if int(blind.sum()) >= 5000 else "WARN"},
        {"check": "blind_target_any_seen_vocab_rate", "value": float(df.loc[blind, "target_any_seen_vocab"].mean()), "status": "PASS" if float(df.loc[blind, "target_any_seen_vocab"].mean()) >= 0.90 else "FAIL"},
        {"check": "blind_target_all_seen_vocab_rate", "value": float(df.loc[blind, "target_all_seen_vocab"].mean()), "status": "INFO"},
        {"check": "old_fragment_overlap_train_blind", "value": len(old_fragment_sets["train"] & old_fragment_sets["blind_test"]), "status": "INFO"},
        {"check": "attachment_signature_overlap_train_blind", "value": len(att_sets["train"] & att_sets["blind_test"]), "status": "INFO"},
    ]
    pd.DataFrame(audit_rows).to_csv(OUT / "d4s_b0_split_leakage_audit.csv", index=False)
    summary = {
        "transform_overlap_train_blind": len(transform_sets["train"] & transform_sets["blind_test"]),
        "transform_overlap_val_blind": len(transform_sets["val"] & transform_sets["blind_test"]),
        "blind_queries": int(blind.sum()),
        "blind_target_any_seen_vocab_rate": float(df.loc[blind, "target_any_seen_vocab"].mean()),
        "blind_target_all_seen_vocab_rate": float(df.loc[blind, "target_all_seen_vocab"].mean()),
        "old_canonical_in_blind": int((df["is_old_canonical_test"] & blind).sum()),
        "old_split_test_in_blind": int((df["is_old_split_test"] & blind).sum()),
    }
    write_json(OUT / "d4s_b0_split_leakage_summary.json", summary)


def build_train_vocab(df: pd.DataFrame):
    train = df.loc[df["split_new"].eq("train")].copy()
    global_counts = Counter()
    attach_counts = defaultdict(Counter)
    for row in train.itertuples(index=False):
        positives = parse_list_cell(row.positive_replacement_set)
        for repl in positives:
            global_counts[repl] += 1
            attach_counts[row.attachment_signature][repl] += 1
    vocab_rows = []
    for repl, freq in sorted(global_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        atts = [att for att, counter in attach_counts.items() if repl in counter]
        vocab_rows.append(
            {
                "replacement_smiles": repl,
                "global_train_frequency": int(freq),
                "attachment_signatures": pipe_join(sorted(atts)),
                "num_attachments_train": int(len(atts)),
            }
        )
    vocab_df = pd.DataFrame(vocab_rows)
    vocab_df.to_csv(OUT / "d4s_b0_train_vocab.csv", index=False)

    full_vocab = [row["replacement_smiles"] for row in vocab_rows]
    full_set = set(full_vocab)
    summary_rows = []
    matrix_manifest_rows = []
    for split_new, sub in df.groupby("split_new"):
        sizes = []
        for row in sub.itertuples(index=False):
            att_vocab = list(attach_counts[row.attachment_signature].keys())
            candidates = att_vocab if att_vocab else full_vocab
            sizes.append(len(candidates))
        summary_rows.append(
            {
                "split_new": split_new,
                "n_queries": int(len(sub)),
                "new_train_vocab_size": int(len(full_vocab)),
                "candidate_set_size_mean": float(np.mean(sizes)) if sizes else 0.0,
                "candidate_set_size_p50": float(np.percentile(sizes, 50)) if sizes else 0.0,
                "candidate_set_size_p90": float(np.percentile(sizes, 90)) if sizes else 0.0,
                "candidate_set_size_max": int(max(sizes)) if sizes else 0,
                "seen_vocab_eval_queries": int(sub["target_any_seen_vocab"].sum()),
            }
        )
        out_dir = OUT / "matrices" / split_new
        out_dir.mkdir(parents=True, exist_ok=True)
        shard = []
        shard_id = 0
        total_rows = 0
        for row in sub.itertuples(index=False):
            positives = set(parse_list_cell(row.positive_replacement_set))
            att_counter = attach_counts[row.attachment_signature]
            ordered = sorted(att_counter.items(), key=lambda kv: (-kv[1], -global_counts[kv[0]], kv[0])) if att_counter else [(cand, global_counts[cand]) for cand in full_vocab]
            candidates = [cand for cand, _ in ordered] if att_counter else full_vocab
            for cand in candidates:
                shard.append(
                    {
                        "query_id": row.query_id,
                        "split": split_new,
                        "candidate": cand,
                        "label": int(cand in positives),
                        "global_freq": int(global_counts.get(cand, 0)),
                        "attach_freq": int(att_counter.get(cand, 0)),
                    }
                )
                total_rows += 1
                if len(shard) >= SHARD_SIZE:
                    path = out_dir / f"{split_new}_features_shard_{shard_id:04d}.jsonl"
                    write_jsonl(path, shard)
                    shard = []
                    shard_id += 1
        if shard:
            path = out_dir / f"{split_new}_features_shard_{shard_id:04d}.jsonl"
            write_jsonl(path, shard)
            shard_id += 1
        matrix_manifest_rows.append({"split_new": split_new, "num_shards": int(shard_id), "total_rows": int(total_rows)})
    pd.DataFrame(summary_rows).to_csv(OUT / "d4s_b0_candidate_universe_summary.csv", index=False)
    pd.DataFrame(matrix_manifest_rows).to_csv(OUT / "d4s_b0_matrix_shard_manifest.csv", index=False)
    return vocab_df, full_set


def main():
    input_df = build_input_discovery()
    input_df.to_csv(OUT / "d4s_b0_input_discovery.csv", index=False)
    if not POSITIVE_MANIFEST.exists() or not SRC_MANIFEST.exists() or not PHASE1_QUERY.exists():
        raise SystemExit("F. D4S_B0_INPUTS_MISSING")
    source_df = pd.read_json(POSITIVE_MANIFEST, lines=True)
    frozen_df = pd.read_json(SRC_MANIFEST, lines=True)
    source_df["query_id"] = source_df["query_id"].astype(str)
    frozen_df["query_id"] = frozen_df["query_id"].astype(str)
    frozen_meta = frozen_df[["query_id", "split"]].copy()
    manifest = source_df.merge(frozen_meta, on="query_id", how="left", validate="one_to_one")
    require_columns(manifest)
    manifest["positive_replacement_set"] = manifest["positive_replacement_set"].apply(parse_list_cell)
    manifest["transform_key_set"] = manifest["transform_key_set"].apply(parse_list_cell)
    canonical_qids = set(pd.read_csv(PHASE1_QUERY, usecols=["query_id"])["query_id"].astype(str))
    manifest["query_id"] = manifest["query_id"].astype(str)
    write_quarantine(manifest, canonical_qids)
    manifest = build_components(manifest)
    manifest, meta = choose_split(manifest, canonical_qids)
    write_secondary_manifest(manifest)
    write_leakage_audit(manifest)
    build_train_vocab(manifest)
    write_json(
        OUT / "d4s_b0_build_state.json",
        {
            "seed": SEED,
            "n_queries_total": int(len(manifest)),
            "split_meta": meta,
            "out_dir": str(OUT),
        },
    )


if __name__ == "__main__":
    main()
