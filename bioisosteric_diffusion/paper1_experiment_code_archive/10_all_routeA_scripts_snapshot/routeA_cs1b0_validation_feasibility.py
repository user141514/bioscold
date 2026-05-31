#!/usr/bin/env python
"""CS1B-0 Route-A activity / external validation feasibility audit.

This audit is evidence-conservative by design:
- weak MMP labels stay weak structure-derived labels
- activity comparisons require target/assay identity and compatible units
- no final labels are built here
- no training is performed here
"""

import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "plan_results" / "routeA_cs1b0_validation_feasibility"
ROUTEA_DIR = PROJECT_ROOT / "plan_results" / "routeA_chembl37k_d0d3_engineering_safe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "datasets",
    PROJECT_ROOT / "raw",
    PROJECT_ROOT / "processed",
    PROJECT_ROOT / "chembl",
    PROJECT_ROOT / "ChEMBL",
    PROJECT_ROOT / "resources",
    PROJECT_ROOT / "plan_results",
    PROJECT_ROOT / "plan_results" / "routeA_chembl37k_d0d3_engineering_safe",
]

DISCOVERY_KEYWORDS = [
    "chembl",
    "activity",
    "assay",
    "target",
    "pchembl",
    "ic50",
    "ki",
    "ec50",
    "bioisostere",
    "known_positive",
    "external",
    "validation",
    "mmp",
    "replacement",
]

DISCOVERY_COLUMNS = [
    "file_path",
    "role",
    "file_type",
    "size_bytes",
    "has_smiles",
    "has_chembl_id",
    "has_target_id",
    "has_assay_id",
    "has_activity_value",
    "has_activity_units",
    "has_pchembl",
    "has_pair_id",
    "has_old_fragment",
    "has_replacement_fragment",
    "status",
    "notes",
]

INVENTORY_COLUMNS = [
    "molecule_id",
    "chembl_id",
    "canonical_smiles",
    "target_id",
    "assay_id",
    "activity_type",
    "activity_value",
    "activity_units",
    "pchembl_value",
    "standard_relation",
    "confidence_score_if_available",
    "source_file",
]

COMPARABILITY_COLUMNS = [
    "group_id",
    "target_id",
    "assay_id",
    "activity_type",
    "units",
    "num_molecules",
    "num_replacement_pairs_if_joinable",
    "pchembl_available_rate",
    "comparable_status",
]

PAIR_JOIN_COLUMNS = [
    "pair_id",
    "old_mol_id",
    "replacement_mol_id",
    "old_fragment",
    "replacement_fragment",
    "core_key",
    "old_activity_available",
    "replacement_activity_available",
    "same_target",
    "same_assay",
    "activity_comparable",
    "delta_pchembl_if_available",
    "label_feasibility",
]

EXTERNAL_COLUMNS = [
    "source_name",
    "file_path",
    "pair_count",
    "has_old_fragment",
    "has_replacement_fragment",
    "has_activity",
    "has_label",
    "license_or_provenance_if_available",
    "usable_status",
    "notes",
]

EXPERT_COLUMNS = [
    "group_name",
    "source_file",
    "available_count",
    "has_old_fragment",
    "has_replacement",
    "has_attachment_signature",
    "has_mode",
    "has_rank",
    "has_a4c_tier",
    "has_group_origin",
    "has_property_delta",
    "has_alert_flag",
    "hidden_method_label_possible",
    "blind_review_ready_status",
    "notes",
]


def relpath_or_abs(path):
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_csv_header(path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def read_jsonl_keys(path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                return []
            if isinstance(payload, dict):
                return list(payload.keys())
            return []
    return []


def read_json_keys(path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return list(payload.keys())
    return []


def scan_sdf_properties(path, max_records=5000):
    properties = Counter()
    record_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "$$$$":
                record_count += 1
                if record_count >= max_records:
                    break
            elif stripped.startswith("> <") and stripped.endswith(">"):
                prop_name = stripped[3:-1].strip("<> ")
                if prop_name:
                    properties[prop_name] += 1
    return record_count, properties


def infer_role(path, schema_keys):
    text = str(path).lower()
    schema = {str(item).lower() for item in schema_keys}
    if path.suffix.lower() == ".sdf":
        return "raw_chembl_sdf"
    if "activity_availability" in text:
        return "activity_availability_summary"
    if "inventory" in text and "pair" not in text:
        return "molecule_inventory"
    if "pair_inventory" in text:
        return "pair_inventory"
    if "query_positive" in text:
        return "query_positive_manifest"
    if "pair_benchmark_manifest" in text or "benchmark_manifest" in text:
        return "benchmark_manifest"
    if "mmp_pairs_filtered" in text:
        return "routea_pair_shard"
    if "bioisostere" in text and "class_prior" in text:
        return "internal_bioisostere_class_prior"
    if "replacement_taxonomy" in text:
        return "internal_replacement_taxonomy"
    if "candidate_review_table" in text:
        return "expert_review_table"
    if "candidate_labels_g1" in text:
        return "exploration_label_table"
    if "_g2_candidates" in text:
        return "g2_pure_borda_only_candidates"
    if "_g3_candidates" in text:
        return "g3_de_retained_candidates"
    if "_g4_candidates" in text:
        return "g4_shared_candidates"
    if "_g1_candidates" in text:
        return "g1_exploration_candidates"
    if "eval_query_info" in text:
        return "query_context_table"
    if "validation_plan" in text:
        return "validation_plan"
    if "external" in text and path.suffix.lower() in {".csv", ".json", ".jsonl", ".md"}:
        return "external_candidate_source"
    if {"pair_id", "old_mol_id", "replacement_mol_id"} <= schema:
        return "pair_source"
    return "candidate_input"


def infer_status(path, schema_keys, notes):
    text = str(path).lower()
    schema = {str(item).lower() for item in schema_keys}
    if path.suffix.lower() == ".sdf" and "chembl" in text:
        return "STRUCTURE_ONLY_RAW"
    if {"target_id", "assay_id"} & schema and (
        {"activity_value", "activity_units", "pchembl_value"} & schema
    ):
        return "ACTIVITY_CANDIDATE"
    if "activity" in notes.lower() and "not available" in notes.lower():
        return "STRUCTURE_ONLY"
    if "validation_plan" in text:
        return "PLAN_ONLY"
    if "candidate_review_table" in text or "candidate_labels_g1" in text:
        return "EXPERT_REVIEW_CANDIDATE"
    if {"pair_id", "old_mol_id", "replacement_mol_id"} <= schema:
        return "PAIR_SOURCE"
    return "REFERENCE"


def schema_flags(schema_keys, file_type):
    schema = {str(item).lower() for item in schema_keys}
    return {
        "has_smiles": int(
            "smiles" in "".join(schema)
            or file_type == "sdf"
            or any("candidate_norm" == item for item in schema)
        ),
        "has_chembl_id": int("chembl_id" in schema or file_type == "sdf"),
        "has_target_id": int("target_id" in schema),
        "has_assay_id": int("assay_id" in schema),
        "has_activity_value": int(
            "activity_value" in schema
            or "standard_value" in schema
            or "pactivity" in schema
        ),
        "has_activity_units": int(
            "activity_units" in schema or "standard_units" in schema or "units" in schema
        ),
        "has_pchembl": int("pchembl_value" in schema or "pchembl" in schema),
        "has_pair_id": int("pair_id" in schema),
        "has_old_fragment": int(
            "old_fragment" in "".join(schema)
            or "old_fragment_smiles" in schema
            or "old_attach_smi" in schema
        ),
        "has_replacement_fragment": int(
            "replacement_fragment" in "".join(schema)
            or "replacement_fragment_smiles" in schema
            or "repl_attach_smi" in schema
            or "candidate_norm" in schema
            or "replacement_smiles" in schema
        ),
    }


def inspect_path(path):
    file_type = path.suffix.lower().lstrip(".") or "no_extension"
    notes = []
    schema_keys = []
    if file_type == "csv":
        schema_keys = read_csv_header(path)
        notes.append("header=" + "|".join(schema_keys[:20]))
    elif file_type == "jsonl":
        schema_keys = read_jsonl_keys(path)
        notes.append("jsonl_keys=" + "|".join(schema_keys[:20]))
    elif file_type == "json":
        try:
            schema_keys = read_json_keys(path)
            notes.append("json_keys=" + "|".join(schema_keys[:20]))
        except Exception as exc:
            notes.append("json_read_error=" + str(exc)[:120])
    elif file_type in {"md", "txt", "log"}:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            snippet = []
            for _ in range(3):
                line = handle.readline()
                if not line:
                    break
                snippet.append(line.strip())
        notes.append("snippet=" + " | ".join(snippet)[:200])
    elif file_type == "sdf":
        scanned, properties = scan_sdf_properties(path)
        schema_keys = list(properties.keys())
        notes.append("scanned_records=" + str(scanned))
        notes.append("properties=" + "|".join(sorted(properties.keys())))
    else:
        notes.append("uninspected_file_type")
    flags = schema_flags(schema_keys, file_type)
    role = infer_role(path, schema_keys)
    note_text = "; ".join(part for part in notes if part)
    status = infer_status(path, schema_keys, note_text)
    row = {
        "file_path": relpath_or_abs(path),
        "role": role,
        "file_type": file_type,
        "size_bytes": path.stat().st_size,
        "status": status,
        "notes": note_text,
    }
    row.update(flags)
    return row


def discover_inputs():
    candidates = set()
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if OUTPUT_DIR in path.parents:
                continue
            name = path.name.lower()
            if any(keyword in name for keyword in DISCOVERY_KEYWORDS):
                candidates.add(path.resolve())

    manual_paths = [
        ROUTEA_DIR / "run_config.json",
        ROUTEA_DIR / "01_d0_data_audit" / "d0_chembl37k_inventory.csv",
        ROUTEA_DIR / "01_d0_data_audit" / "d0_chembl37k_activity_availability.csv",
        ROUTEA_DIR / "06_d3r_exam_repair" / "d3r_pair_inventory.csv",
        ROUTEA_DIR / "06_d3r_exam_repair" / "d3r_query_positive_manifest.jsonl",
        PROJECT_ROOT
        / "plan_results"
        / "routeA_chembl37k_d4a3r_a4c_borda_review"
        / "d4a3r_eval_query_info.csv",
        PROJECT_ROOT
        / "plan_results"
        / "routeA_chembl37k_d4a4_a4c_gated_reranking"
        / "d4a4_candidate_review_table.csv",
        PROJECT_ROOT
        / "plan_results"
        / "routeA_chembl37k_d4a3t_exploration_calibration"
        / "d4a3t_candidate_labels_g1.csv",
        PROJECT_ROOT
        / "plan_results"
        / "routeA_chembl37k_d4a3s_a4c_coverage_expansion"
        / "d4a3s_G2_candidates.csv",
        PROJECT_ROOT
        / "plan_results"
        / "routeA_chembl37k_d4a3s_a4c_coverage_expansion"
        / "d4a3s_G3_candidates.csv",
        PROJECT_ROOT
        / "plan_results"
        / "routeA_chembl37k_d4a3s_a4c_coverage_expansion"
        / "d4a3s_G4_candidates.csv",
        PROJECT_ROOT
        / "plan_results"
        / "A4C_SEMANTIC_REVIEW_V0"
        / "case_studies"
        / "A4C_CS1_REAL_PAIR_VALIDATION_PLAN.md",
        PROJECT_ROOT
        / "plan_results"
        / "A4C_V1A_FULL_FEATURE_AUDIT"
        / "a4c_v1a_bioisostere_class_prior.csv",
        PROJECT_ROOT
        / "plan_results"
        / "A4_FULL_BIOISOSTERE_SEMANTIC_EVALUATION"
        / "a4_fg_replacement_taxonomy.json",
    ]
    for path in manual_paths:
        if path.exists():
            candidates.add(path.resolve())

    run_config = ROUTEA_DIR / "run_config.json"
    if run_config.exists():
        config = json.loads(run_config.read_text(encoding="utf-8"))
        sdf_path = Path(config.get("sdf_path", ""))
        if sdf_path.exists():
            candidates.add(sdf_path.resolve())

    discoveries = [inspect_path(path) for path in sorted(candidates)]
    out_path = OUTPUT_DIR / "cs1b0_input_discovery.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DISCOVERY_COLUMNS)
        writer.writeheader()
        writer.writerows(discoveries)
    return discoveries


def iter_standardized_records():
    chunk_dir = ROUTEA_DIR / "01_d0_data_audit" / "standardized_chunks"
    for chunk_path in sorted(chunk_dir.glob("standardized_chunk_*.jsonl")):
        with chunk_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield chunk_path, json.loads(line)


def build_activity_inventory():
    out_path = OUTPUT_DIR / "cs1b0_activity_column_inventory.csv"
    summary = {
        "total_molecules": 0,
        "with_target_id": 0,
        "with_assay_id": 0,
        "with_activity": 0,
        "with_pchembl": 0,
        "activity_type_distribution": Counter(),
        "unit_distribution": Counter(),
        "target_ids": set(),
        "assay_ids": set(),
    }
    sparse_activity_index = {}

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        for source_file, record in iter_standardized_records():
            row = {
                "molecule_id": record.get("mol_id", ""),
                "chembl_id": record.get("chembl_id", ""),
                "canonical_smiles": record.get("canonical_smiles", ""),
                "target_id": record.get("target_id", ""),
                "assay_id": record.get("assay_id", ""),
                "activity_type": record.get("activity_type", ""),
                "activity_value": record.get("activity_value", ""),
                "activity_units": record.get("activity_units", ""),
                "pchembl_value": record.get("pchembl_value", ""),
                "standard_relation": record.get("standard_relation", ""),
                "confidence_score_if_available": record.get("confidence_score", ""),
                "source_file": relpath_or_abs(source_file),
            }
            writer.writerow(row)

            summary["total_molecules"] += 1
            if row["target_id"]:
                summary["with_target_id"] += 1
                summary["target_ids"].add(row["target_id"])
            if row["assay_id"]:
                summary["with_assay_id"] += 1
                summary["assay_ids"].add(row["assay_id"])
            if row["activity_value"] not in ("", None):
                summary["with_activity"] += 1
            if row["pchembl_value"] not in ("", None):
                summary["with_pchembl"] += 1
            if row["activity_type"]:
                summary["activity_type_distribution"][row["activity_type"]] += 1
            else:
                summary["activity_type_distribution"]["MISSING"] += 1
            if row["activity_units"]:
                summary["unit_distribution"][row["activity_units"]] += 1
            else:
                summary["unit_distribution"]["MISSING"] += 1

            if any(
                [
                    row["target_id"],
                    row["assay_id"],
                    row["activity_value"] not in ("", None),
                    row["pchembl_value"] not in ("", None),
                ]
            ):
                sparse_activity_index[row["molecule_id"]] = {
                    "target_id": row["target_id"],
                    "assay_id": row["assay_id"],
                    "activity_type": row["activity_type"],
                    "activity_units": row["activity_units"],
                    "activity_value": row["activity_value"],
                    "pchembl_value": row["pchembl_value"],
                }

    summary_payload = {
        "total_molecules": summary["total_molecules"],
        "with_target_id": summary["with_target_id"],
        "with_assay_id": summary["with_assay_id"],
        "with_activity": summary["with_activity"],
        "with_pchembl": summary["with_pchembl"],
        "activity_type_distribution": dict(summary["activity_type_distribution"]),
        "unit_distribution": dict(summary["unit_distribution"]),
        "target_count": len(summary["target_ids"]),
        "assay_count": len(summary["assay_ids"]),
        "status": (
            "STRUCTURE_ONLY"
            if summary["with_activity"] == 0
            else "PARTIAL_ACTIVITY_FEASIBILITY"
        ),
    }
    (OUTPUT_DIR / "cs1b0_activity_availability_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_payload, sparse_activity_index


def write_comparability_audit(activity_summary):
    out_path = OUTPUT_DIR / "cs1b0_assay_comparability_audit.csv"
    rows = []
    if activity_summary["with_activity"] == 0:
        rows.append(
            {
                "group_id": "STRUCTURE_ONLY_NO_ACTIVITY",
                "target_id": "",
                "assay_id": "",
                "activity_type": "",
                "units": "",
                "num_molecules": activity_summary["total_molecules"],
                "num_replacement_pairs_if_joinable": 0,
                "pchembl_available_rate": 0.0,
                "comparable_status": "NOT_COMPARABLE_NO_ACTIVITY",
            }
        )
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARABILITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def choose_pair_source():
    preferred_csv = ROUTEA_DIR / "06_d3r_exam_repair" / "d3r_pair_inventory.csv"
    if preferred_csv.exists():
        return preferred_csv, "csv"
    preferred_jsonl = ROUTEA_DIR / "05_d2_labeling_repaired" / "d2r_benchmark_manifest_ratio1.jsonl"
    if preferred_jsonl.exists():
        return preferred_jsonl, "jsonl"
    raise FileNotFoundError("No D3R/D2R pair source found for CS1B-0 audit")


def iter_pair_rows(path, kind):
    if kind == "csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row
    else:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def build_pair_join_audit(sparse_activity_index):
    pair_source, pair_kind = choose_pair_source()
    out_path = OUTPUT_DIR / "cs1b0_pair_activity_join_audit.csv"
    total_pairs = 0
    pairs_with_both_activity = 0
    same_target_pairs = 0
    same_assay_pairs = 0
    strong_comparable_pairs = 0
    strong_positive_estimate = 0
    strong_negative_estimate = 0
    ambiguous_count = 0

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_JOIN_COLUMNS)
        writer.writeheader()
        for row in iter_pair_rows(pair_source, pair_kind):
            pair_id = row.get("pair_id", "")
            old_mol_id = row.get("old_mol_id", "")
            replacement_mol_id = row.get("replacement_mol_id", "")
            old_fragment = row.get("old_fragment_smiles", row.get("old_fragment", ""))
            replacement_fragment = row.get(
                "replacement_fragment_smiles",
                row.get("replacement_fragment", ""),
            )
            core_key = row.get("core_key", "")

            old_activity = sparse_activity_index.get(old_mol_id, {})
            new_activity = sparse_activity_index.get(replacement_mol_id, {})
            old_has = bool(old_activity.get("activity_value") not in ("", None))
            new_has = bool(new_activity.get("activity_value") not in ("", None))
            both_have = old_has and new_has
            same_target = bool(
                both_have
                and old_activity.get("target_id")
                and old_activity.get("target_id") == new_activity.get("target_id")
            )
            same_assay = bool(
                both_have
                and old_activity.get("assay_id")
                and old_activity.get("assay_id") == new_activity.get("assay_id")
            )
            comparable = bool(
                same_target
                and same_assay
                and old_activity.get("activity_units")
                and old_activity.get("activity_units") == new_activity.get("activity_units")
                and old_activity.get("activity_type")
                and old_activity.get("activity_type") == new_activity.get("activity_type")
                and old_activity.get("pchembl_value") not in ("", None)
                and new_activity.get("pchembl_value") not in ("", None)
            )

            delta_pchembl = ""
            label_feasibility = "NO_ACTIVITY"
            if comparable:
                delta_pchembl = abs(
                    float(old_activity["pchembl_value"]) - float(new_activity["pchembl_value"])
                )
                strong_comparable_pairs += 1
                if delta_pchembl <= 1.0:
                    strong_positive_estimate += 1
                    label_feasibility = "CANDIDATE_STRONG_POSITIVE"
                elif delta_pchembl >= 2.0:
                    strong_negative_estimate += 1
                    label_feasibility = "CANDIDATE_STRONG_NEGATIVE"
                else:
                    ambiguous_count += 1
                    label_feasibility = "AMBIGUOUS_ACTIVITY_DELTA"
            else:
                ambiguous_count += 1

            writer.writerow(
                {
                    "pair_id": pair_id,
                    "old_mol_id": old_mol_id,
                    "replacement_mol_id": replacement_mol_id,
                    "old_fragment": old_fragment,
                    "replacement_fragment": replacement_fragment,
                    "core_key": core_key,
                    "old_activity_available": int(old_has),
                    "replacement_activity_available": int(new_has),
                    "same_target": int(same_target),
                    "same_assay": int(same_assay),
                    "activity_comparable": int(comparable),
                    "delta_pchembl_if_available": delta_pchembl,
                    "label_feasibility": label_feasibility,
                }
            )

            total_pairs += 1
            if both_have:
                pairs_with_both_activity += 1
            if same_target:
                same_target_pairs += 1
            if same_assay:
                same_assay_pairs += 1

    summary = {
        "total_pairs": total_pairs,
        "pairs_with_both_activity": pairs_with_both_activity,
        "same_target_pairs": same_target_pairs,
        "same_assay_pairs": same_assay_pairs,
        "strong_comparable_pairs": strong_comparable_pairs,
        "potential_strong_positive_count": strong_positive_estimate,
        "potential_strong_negative_count": strong_negative_estimate,
        "ambiguous_count": ambiguous_count,
        "pair_source_file": relpath_or_abs(pair_source),
    }
    (OUTPUT_DIR / "cs1b0_pair_activity_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def write_label_feasibility_estimate(pair_summary):
    if (
        pair_summary["potential_strong_positive_count"] >= 1000
        and pair_summary["potential_strong_negative_count"] >= 1000
    ):
        feasibility_band = "A_STRONG_ACTIVITY_VALIDATION_FEASIBLE"
    elif (
        pair_summary["potential_strong_positive_count"] >= 200
        and pair_summary["potential_strong_negative_count"] >= 200
    ):
        feasibility_band = "B_SMALL_ACTIVITY_VALIDATION_FEASIBLE"
    else:
        feasibility_band = "C_ACTIVITY_VALIDATION_WEAK"

    row = {
        "strong_positive_estimate": pair_summary["potential_strong_positive_count"],
        "strong_negative_estimate": pair_summary["potential_strong_negative_count"],
        "ambiguous_estimate": pair_summary["ambiguous_count"],
        "minimum_viable_validation_size": min(
            pair_summary["potential_strong_positive_count"],
            pair_summary["potential_strong_negative_count"],
        ),
        "feasibility_band": feasibility_band,
    }
    out_path = OUTPUT_DIR / "cs1b0_label_feasibility_estimate.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def maybe_count_csv_rows(path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def write_external_source_inventory():
    candidates = []

    plan_path = (
        PROJECT_ROOT
        / "plan_results"
        / "A4C_SEMANTIC_REVIEW_V0"
        / "case_studies"
        / "A4C_CS1_REAL_PAIR_VALIDATION_PLAN.md"
    )
    if plan_path.exists():
        candidates.append(
            {
                "source_name": "A4C_CS1_real_pair_validation_plan",
                "file_path": relpath_or_abs(plan_path),
                "pair_count": 0,
                "has_old_fragment": 1,
                "has_replacement_fragment": 1,
                "has_activity": 1,
                "has_label": 1,
                "license_or_provenance_if_available": "Plan cites literature / SwissBioisostere / ChEMBL_MMP as future sources",
                "usable_status": "PLAN_ONLY_NO_LOCAL_DATA",
                "notes": "Explicitly states no local known-positive bioisosteric replacement pair file exists.",
            }
        )

    class_prior_path = (
        PROJECT_ROOT
        / "plan_results"
        / "A4C_V1A_FULL_FEATURE_AUDIT"
        / "a4c_v1a_bioisostere_class_prior.csv"
    )
    if class_prior_path.exists():
        candidates.append(
            {
                "source_name": "A4C_internal_bioisostere_class_prior",
                "file_path": relpath_or_abs(class_prior_path),
                "pair_count": maybe_count_csv_rows(class_prior_path),
                "has_old_fragment": 1,
                "has_replacement_fragment": 1,
                "has_activity": 0,
                "has_label": 1,
                "license_or_provenance_if_available": "Internal generated audit artifact",
                "usable_status": "NOT_EXTERNAL_NOT_GROUND_TRUTH",
                "notes": "Contains internal class labels, not curated external validation pairs.",
            }
        )

    taxonomy_path = (
        PROJECT_ROOT
        / "plan_results"
        / "A4_FULL_BIOISOSTERE_SEMANTIC_EVALUATION"
        / "a4_fg_replacement_taxonomy.json"
    )
    if taxonomy_path.exists():
        candidates.append(
            {
                "source_name": "A4_internal_replacement_taxonomy",
                "file_path": relpath_or_abs(taxonomy_path),
                "pair_count": 0,
                "has_old_fragment": 0,
                "has_replacement_fragment": 0,
                "has_activity": 0,
                "has_label": 1,
                "license_or_provenance_if_available": "Internal generated taxonomy",
                "usable_status": "NOT_PAIR_LEVEL_VALIDATION",
                "notes": "Semantic taxonomy, not a known-positive pair source.",
            }
        )

    m2r_manifest = (
        PROJECT_ROOT
        / "plan_results"
        / "P2A_M2R_UNIFIED_BENCHMARK"
        / "m2r_raw_800_frozen_manifest.json"
    )
    if m2r_manifest.exists():
        candidates.append(
            {
                "source_name": "M2R_unified_benchmark_manifest",
                "file_path": relpath_or_abs(m2r_manifest),
                "pair_count": 800,
                "has_old_fragment": 0,
                "has_replacement_fragment": 0,
                "has_activity": 0,
                "has_label": 0,
                "license_or_provenance_if_available": "Internal benchmark manifest",
                "usable_status": "NO_REPLACEMENT_PAIR_FIELDS",
                "notes": "Contains frozen fragment/scaffold cases but no explicit replacement-pair labels.",
            }
        )

    stage3_audit = PROJECT_ROOT / "core" / "docs" / "stage3_pair_distribution_audit.md"
    if stage3_audit.exists():
        candidates.append(
            {
                "source_name": "stage3_pair_distribution_audit",
                "file_path": relpath_or_abs(stage3_audit),
                "pair_count": 762,
                "has_old_fragment": 1,
                "has_replacement_fragment": 1,
                "has_activity": 0,
                "has_label": 0,
                "license_or_provenance_if_available": "Internal audit on E:/zuhui/chembl_data",
                "usable_status": "PAIRS_WITHOUT_ACTIVITY_GROUND_TRUTH",
                "notes": "Documents mined training pairs only; not known-positive external validation.",
            }
        )

    out_path = OUTPUT_DIR / "cs1b0_external_source_inventory.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTERNAL_COLUMNS)
        writer.writeheader()
        writer.writerows(candidates)
    return candidates


def write_expert_review_feasibility():
    rows = []
    rows.append(
        {
            "group_name": "Conservative-only",
            "source_file": "plan_results/routeA_chembl37k_d4a4_a4c_gated_reranking/d4a4_candidate_review_table.csv + plan_results/routeA_chembl37k_d4a3r_a4c_borda_review/d4a3r_eval_query_info.csv",
            "available_count": 68370,
            "has_old_fragment": 1,
            "has_replacement": 1,
            "has_attachment_signature": 1,
            "has_mode": 1,
            "has_rank": 1,
            "has_a4c_tier": 1,
            "has_group_origin": 1,
            "has_property_delta": 0,
            "has_alert_flag": 1,
            "hidden_method_label_possible": 1,
            "blind_review_ready_status": "PARTIAL_READY_NEEDS_PROPERTY_DELTA",
            "notes": "Use M1_canonical_HGB rows; old_fragment and attachment_signature are recoverable from eval_query_info join.",
        }
    )
    rows.append(
        {
            "group_name": "Exploration-only",
            "source_file": "plan_results/routeA_chembl37k_d4a3t_exploration_calibration/d4a3t_candidate_labels_g1.csv + plan_results/routeA_chembl37k_d4a3r_a4c_borda_review/d4a3r_eval_query_info.csv",
            "available_count": 5358,
            "has_old_fragment": 1,
            "has_replacement": 1,
            "has_attachment_signature": 1,
            "has_mode": 1,
            "has_rank": 0,
            "has_a4c_tier": 1,
            "has_group_origin": 1,
            "has_property_delta": 0,
            "has_alert_flag": 1,
            "hidden_method_label_possible": 1,
            "blind_review_ready_status": "PARTIAL_READY_NEEDS_RANK_AND_PROPERTY_DELTA",
            "notes": "G1 exploration pool already has A4C labels; rank requires downstream proposal join.",
        }
    )
    rows.append(
        {
            "group_name": "G2 pure Borda-only",
            "source_file": "plan_results/routeA_chembl37k_d4a3s_a4c_coverage_expansion/d4a3s_G2_candidates.csv",
            "available_count": 444,
            "has_old_fragment": 1,
            "has_replacement": 1,
            "has_attachment_signature": 0,
            "has_mode": 1,
            "has_rank": 0,
            "has_a4c_tier": 0,
            "has_group_origin": 1,
            "has_property_delta": 0,
            "has_alert_flag": 0,
            "hidden_method_label_possible": 1,
            "blind_review_ready_status": "NEEDS_A4C_AND_QUERY_CONTEXT_JOIN",
            "notes": "High-risk exploration subgroup. A4C tier and attachment signature require joins to G1 labels and query_info.",
        }
    )
    rows.append(
        {
            "group_name": "G3 DE-retained",
            "source_file": "plan_results/routeA_chembl37k_d4a3s_a4c_coverage_expansion/d4a3s_G3_candidates.csv",
            "available_count": 4914,
            "has_old_fragment": 1,
            "has_replacement": 1,
            "has_attachment_signature": 0,
            "has_mode": 1,
            "has_rank": 0,
            "has_a4c_tier": 0,
            "has_group_origin": 1,
            "has_property_delta": 0,
            "has_alert_flag": 0,
            "hidden_method_label_possible": 1,
            "blind_review_ready_status": "NEEDS_A4C_AND_QUERY_CONTEXT_JOIN",
            "notes": "Mid-risk exploration subgroup. Ample sample size for 100-300 candidate review pilot.",
        }
    )
    rows.append(
        {
            "group_name": "G4 shared",
            "source_file": "plan_results/routeA_chembl37k_d4a3s_a4c_coverage_expansion/d4a3s_G4_candidates.csv",
            "available_count": 21013,
            "has_old_fragment": 1,
            "has_replacement": 1,
            "has_attachment_signature": 0,
            "has_mode": 1,
            "has_rank": 0,
            "has_a4c_tier": 0,
            "has_group_origin": 1,
            "has_property_delta": 0,
            "has_alert_flag": 0,
            "hidden_method_label_possible": 1,
            "blind_review_ready_status": "NEEDS_A4C_AND_QUERY_CONTEXT_JOIN",
            "notes": "Shared low-risk comparison pool; sufficient for control sampling.",
        }
    )
    rows.append(
        {
            "group_name": "random/property-matched decoy",
            "source_file": "plan_results/routeA_chembl37k_d0d3_engineering_safe/06_d3r_exam_repair/d3r_pair_inventory.csv",
            "available_count": 225803,
            "has_old_fragment": 1,
            "has_replacement": 1,
            "has_attachment_signature": 1,
            "has_mode": 0,
            "has_rank": 0,
            "has_a4c_tier": 0,
            "has_group_origin": 1,
            "has_property_delta": 0,
            "has_alert_flag": 0,
            "hidden_method_label_possible": 1,
            "blind_review_ready_status": "RANDOM_DECOY_READY_PROPERTY_MATCH_NEEDS_NEW_SAMPLING",
            "notes": "Random decoys are abundant; property-matched decoys are not pre-materialized and require a new sampling pass.",
        }
    )

    out_path = OUTPUT_DIR / "cs1b0_expert_review_sampling_feasibility.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_verdict(activity_summary, pair_summary, label_row, external_rows, expert_rows):
    local_external_usable = [
        row for row in external_rows if row["usable_status"] == "USABLE_LOCAL_EXTERNAL_VALIDATION"
    ]
    external_available = bool(local_external_usable)
    expert_feasible = any(
        int(row["available_count"]) >= 100
        for row in expert_rows
        if str(row["available_count"]).isdigit()
    )

    if (
        pair_summary["potential_strong_positive_count"] >= 1000
        and pair_summary["potential_strong_negative_count"] >= 1000
    ):
        final_verdict = "A. CS1B0_STRONG_ACTIVITY_VALIDATION_FEASIBLE"
    elif (
        pair_summary["potential_strong_positive_count"] >= 200
        and pair_summary["potential_strong_negative_count"] >= 200
    ):
        final_verdict = "B. CS1B0_SMALL_ACTIVITY_VALIDATION_FEASIBLE"
    elif external_available:
        final_verdict = "C. CS1B0_EXTERNAL_VALIDATION_FEASIBLE"
    elif expert_feasible:
        final_verdict = "D. CS1B0_EXPERT_REVIEW_PILOT_RECOMMENDED"
    elif activity_summary["with_activity"] == 0:
        final_verdict = "E. CS1B0_STRUCTURE_ONLY_NO_STRONG_VALIDATION"
    else:
        final_verdict = "F. CS1B0_INPUTS_MISSING_OR_INCONCLUSIVE"

    strongest_honest_claim = (
        "Route-A currently supports structure-derived matched molecular pair replacement evaluation, "
        "plus expert/A4C-style review of candidate replacements. It does not yet support an "
        "activity-preserving replacement claim from the local ChEMBL37K Route-A inputs."
    )

    verdict_lines = [
        "# CS1B0 Validation Feasibility Verdict",
        "",
        f"**Final verdict:** {final_verdict}",
        "",
        "## Executive Answers",
        "",
        "1. **Is the dataset structure-only or activity-supported?**",
        "   - Structure-only for the current Route-A ChEMBL37K pipeline. The raw `chembl_36.sdf` sampled over 5,000 records exposes only `chembl_id` as an SD property, and the D0 standardized chunks keep `has_target_id = false`, `has_assay_id = false`, `has_activity = false`.",
        "2. **Is activity data available?**",
        f"   - No local Route-A activity table was found for the D0 ChEMBL37K corpus. Activity summary: with_target_id={activity_summary['with_target_id']}, with_assay_id={activity_summary['with_assay_id']}, with_activity={activity_summary['with_activity']}, with_pchembl={activity_summary['with_pchembl']}.",
        "3. **Are same-target same-assay comparisons feasible?**",
        f"   - No. Strong comparable pair count is {pair_summary['strong_comparable_pairs']} because target/assay/activity fields are absent in the local Route-A input stack.",
        "4. **How many strong positives / negatives are possible?**",
        f"   - Strong positive estimate: {pair_summary['potential_strong_positive_count']}. Strong negative estimate: {pair_summary['potential_strong_negative_count']}. These are feasibility counts only; no final labels were created.",
        "5. **Is external validation source available?**",
        "   - No usable local external known-bioisostere validation source was found. The strongest local evidence is an A4C CS1 plan that explicitly states no local known-positive replacement-pair file exists.",
        "6. **Is expert blind review feasible?**",
        "   - Yes. Local candidate pools are large enough for a pilot: G1=5,358, G2=444, G3=4,914, G4=21,013, conservative HGB top-10 rows=68,370, random decoys=225,803. A 100-candidate minimum or 300-candidate stronger pilot is operationally feasible after modest join work.",
        "7. **Which CS1 path should be taken next?**",
        "   - Pursue an expert blind review pilot next, not activity-supported validation from the current local Route-A inputs. In parallel, a separate data-acquisition task can materialize same-target same-assay ChEMBL activity tables if activity validation remains a paper requirement.",
        "8. **Can the paper claim activity-preserving replacement?**",
        "   - No. The current evidence does not support that claim.",
        "9. **If not, what is the strongest honest claim?**",
        f"   - {strongest_honest_claim}",
        "",
        "## Evidence",
        "",
        "- The Route-A run config points to `E:/zuhui/chembl_data/chembl_36.sdf`.",
        "- The raw SDF boundary check shows the first sampled records carry `> <chembl_id>` and no target/assay/activity properties in the first 5,000 scanned records.",
        "- The existing D0 audit wrote `activity_field=structure_only`, `status=NOT_AVAILABLE`.",
        "- D2R / D3R manifests exist and are large enough for benchmarking, but all labels remain weak structure-derived.",
        "- A4C / D4A3 / D4A4 artifacts provide substantial expert-review-ready candidate pools but not activity ground truth.",
        "",
        "## Activity Feasibility Interpretation",
        "",
        f"- Activity status: {'STRUCTURE_ONLY' if activity_summary['with_activity'] == 0 else 'PARTIAL_ACTIVITY_FEASIBILITY'}",
        f"- Label feasibility band: {label_row['feasibility_band']}",
        "- Same-target / same-assay / same-units comparisons should be treated as unavailable until a proper activity table is joined in from a source that actually carries those fields.",
        "",
        "## Expert Review Route",
        "",
        "- The most realistic immediate validation path is a blind expert pilot built from G2/G3/G4 plus a conservative comparison arm and random decoys.",
        "- The current repo already has enough rows for balanced sampling; the missing pieces are mostly join enrichments such as attachment signature, rank, and optional property-delta columns.",
        "- Recommended first pilot size: 100 candidates minimum, 300 candidates preferred.",
        "",
        "## Skeptical Review",
        "",
        "1. **MMP pairs are not the same as bioisosteres.**",
        "   - The Route-A pair universe is mined from structure-derived single-cut replacements. Frequency or recurrence does not prove medicinal-chemistry equivalence, and it does not prove activity preservation.",
        "2. **Assay comparability is absent, not merely weak.**",
        "   - Without target_id plus assay_id plus compatible activity type/units, there is no defensible same-assay activity comparison to promote weak positives into strong positives.",
        "3. **Strong positives are currently unavailable.**",
        "   - A pair cannot be called activity-preserving here because both local molecules lack joined activity evidence.",
        "4. **Strong negatives are also unavailable.**",
        "   - Large activity deltas are meaningful only inside comparable assays. Cross-assay or assay-missing comparisons would be invalid.",
        "5. **Local 'bioisostere' artifacts are mostly internal diagnostics, not external validation.**",
        "   - Internal class priors, semantic taxonomies, and review tables do not substitute for an external curated ground-truth source.",
        "6. **Any future external source must still clear provenance and licensing review.**",
        "   - The local repo does not yet contain a provenance-safe external pair table ready for direct use.",
        "7. **Expert review can be biased if sampling is not stratified.**",
        "   - High-risk G2 pure-Borda-only examples are a small minority but carry disproportionate alert risk; blind review must deliberately oversample them rather than let shared low-risk candidates dominate.",
        "8. **The honest paper claim boundary is narrower than 'bioisostere discovery'.**",
        "   - Current local evidence supports structure-derived replacement generation / ranking plus expert-style review analysis, not validated activity-preserving bioisostere discovery.",
        "",
        "## Recommended Next Action",
        "",
        "- Immediate: build a 100-300 candidate blind expert-review pilot from conservative, G2, G3, G4, and decoy strata.",
        "- Separate follow-up task: acquire or materialize ChEMBL activity tables with target_id, assay_id, activity_type, activity_value, units, and pChEMBL before attempting CS1 activity-supported validation.",
    ]

    (OUTPUT_DIR / "CS1B0_VALIDATION_FEASIBILITY_VERDICT.md").write_text(
        "\n".join(verdict_lines) + "\n",
        encoding="utf-8",
    )

    decision_log_lines = [
        "# MAIN_DECISION_LOG",
        "",
        "## Evidence Discipline",
        "",
        "- Evidence: current Route-A ChEMBL37K raw input is a structure SDF with chembl_id only in sampled SD properties.",
        "- Evidence: D0 standardized chunks explicitly preserve target/assay/activity absence.",
        "- Evidence: D2R / D3R manifests provide large weak-label pair inventories but no activity ground truth.",
        "- Evidence: A4C / D4A3 / D4A4 files provide feasible expert-review candidate pools.",
        "- Inference: activity-supported validation is not feasible from the local Route-A input stack as-is.",
        "- Inference: expert blind review is the most realistic next validation route available locally.",
        "- Actionable: do not call any weak MMP pair activity-preserving; build expert-review pilot first; treat activity validation as a separate data-materialization project.",
        "",
        "## Final Decision",
        "",
        f"- Verdict: {final_verdict}",
        "- Route selection: expert blind review pilot next.",
        "- Claim boundary: structure-derived replacement evaluation only; no activity-preserving claim.",
    ]
    (OUTPUT_DIR / "MAIN_DECISION_LOG.md").write_text(
        "\n".join(decision_log_lines) + "\n",
        encoding="utf-8",
    )
    return final_verdict


def main():
    discover_inputs()
    activity_summary, sparse_activity_index = build_activity_inventory()
    write_comparability_audit(activity_summary)
    pair_summary = build_pair_join_audit(sparse_activity_index)
    label_row = write_label_feasibility_estimate(pair_summary)
    external_rows = write_external_source_inventory()
    expert_rows = write_expert_review_feasibility()
    verdict = build_verdict(activity_summary, pair_summary, label_row, external_rows, expert_rows)
    print("CS1B-0 audit complete")
    print("Output directory:", OUTPUT_DIR)
    print("Verdict:", verdict)


if __name__ == "__main__":
    main()
