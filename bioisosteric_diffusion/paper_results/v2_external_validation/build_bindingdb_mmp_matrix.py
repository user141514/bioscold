#!/usr/bin/env python3
"""Build a BindingDB-derived MMP replacement candidate matrix.

The output follows the standardized matrix contract consumed by
run_external_validation.py. Positives are activity-supported active-active
single-cut BRICS replacements within the same target, endpoint, core, and
attachment signature. Negatives are in-matrix unlabeled candidates sampled from
other compatible fragment contexts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import BRICS

    RDLogger.DisableLog("rdApp.*")
except Exception:  # pragma: no cover - tested in the execution environment
    Chem = None
    BRICS = None


ENDPOINT_COLUMNS = {
    "Ki": "Ki (nM)",
    "IC50": "IC50 (nM)",
    "Kd": "Kd (nM)",
    "EC50": "EC50 (nM)",
}

DOWNLOAD_PAGE_URL = "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp"
DIRECT_ZIP_URL = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202606_tsv.zip"
DIRECT_MD5_URL = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202606_tsv.md5"


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def parse_bindingdb_nm(raw: object) -> tuple[str, float] | None:
    """Parse a BindingDB nM field into (relation, numeric_nM)."""

    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"na", "n/a", "nan", "none", "null"}:
        return None
    text = text.replace(",", "")
    match = re.match(r"^(<=|>=|<|>|=|~)?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", text)
    if not match:
        return None
    relation = match.group(1) or "="
    if relation == "~":
        relation = "="
    value = float(match.group(2))
    if not math.isfinite(value) or value <= 0:
        return None
    return relation, value


def pactivity_from_nm(value_nm: float) -> float:
    return 9.0 - math.log10(value_nm)


def iter_active_measurements(
    row: dict[str, object],
    active_threshold: float = 6.0,
    endpoints: Iterable[str] | None = None,
) -> Iterable[dict[str, object]]:
    """Yield endpoint activity records that can safely support positives."""

    endpoint_names = list(endpoints or ENDPOINT_COLUMNS.keys())
    for endpoint in endpoint_names:
        column = ENDPOINT_COLUMNS[endpoint]
        parsed = parse_bindingdb_nm(row.get(column, ""))
        if parsed is None:
            continue
        relation, value_nm = parsed
        pactivity = pactivity_from_nm(value_nm)

        # Exact values above threshold are active. Upper bounds below the
        # threshold value are also guaranteed active. Lower bounds are ambiguous
        # for positive construction and are skipped.
        if relation in {"=", "<", "<="} and pactivity >= active_threshold:
            yield {
                "endpoint": endpoint,
                "source_column": column,
                "relation": relation,
                "value_nm": value_nm,
                "pactivity": pactivity,
            }


def _require_rdkit() -> None:
    if Chem is None or BRICS is None:
        raise RuntimeError(
            "RDKit is required for BindingDB MMP construction. "
            "Use an environment such as E:\\Anaconda3\\envs\\accfg\\python.exe."
        )


def canonicalize_smiles(smiles: object, largest_fragment: bool = True) -> str | None:
    _require_rdkit()
    text = str(smiles or "").strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    if largest_fragment:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if frags:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def fragment_smiles_brics(
    canonical_smiles: str,
    mol_id: str,
    core_min_heavy: int = 8,
    fragment_min_heavy: int = 1,
    fragment_max_heavy: int = 25,
    mol_min_heavy: int = 10,
    mol_max_heavy: int = 80,
) -> list[dict[str, object]]:
    """Return single-cut BRICS fragment records using the internal D1 convention."""

    _require_rdkit()
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return []
    mol_heavy = mol.GetNumHeavyAtoms()
    if mol_heavy < mol_min_heavy or mol_heavy > mol_max_heavy:
        return []

    records: list[dict[str, object]] = []
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))
    except Exception:
        return []

    for (a1, a2), _envs in bonds:
        bond = mol.GetBondBetweenAtoms(int(a1), int(a2))
        if bond is None:
            continue
        try:
            frag_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()])
            frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
        except Exception:
            continue
        if len(frags) != 2:
            continue

        h1, h2 = frags[0].GetNumHeavyAtoms(), frags[1].GetNumHeavyAtoms()
        if h1 >= h2:
            core, fragment = frags[0], frags[1]
            core_h, fragment_h = h1, h2
        else:
            core, fragment = frags[1], frags[0]
            core_h, fragment_h = h2, h1

        if core_h < core_min_heavy:
            continue
        if fragment_h < fragment_min_heavy or fragment_h > fragment_max_heavy:
            continue

        try:
            core_smiles = Chem.MolToSmiles(core, isomericSmiles=False)
            fragment_smiles = Chem.MolToSmiles(fragment, isomericSmiles=False)
        except Exception:
            continue

        atom_symbols = sorted(
            [mol.GetAtomWithIdx(int(a1)).GetSymbol(), mol.GetAtomWithIdx(int(a2)).GetSymbol()]
        )
        attachment_signature = "|".join(atom_symbols)
        records.append(
            {
                "mol_id": mol_id,
                "canonical_smiles": canonical_smiles,
                "core_smiles": core_smiles,
                "core_key": stable_hash(core_smiles),
                "fragment_smiles": fragment_smiles,
                "fragment_key": stable_hash(fragment_smiles),
                "attachment_signature": attachment_signature,
                "fragment_heavy_atoms": fragment_h,
                "core_heavy_atoms": core_h,
                "molecule_heavy_atoms": mol_heavy,
            }
        )
    return records


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def target_key_from_row(row: dict[str, object], max_chains: int = 50) -> str:
    ids: list[str] = []
    for idx in range(1, max_chains + 1):
        swiss = _clean_text(row.get(f"UniProt (SwissProt) Primary ID of Target Chain {idx}", ""))
        trembl = _clean_text(row.get(f"UniProt (TrEMBL) Primary ID of Target Chain {idx}", ""))
        target_id = swiss or trembl
        if target_id and target_id not in ids:
            ids.append(target_id)
    if ids:
        return "|".join(ids)

    target_name = _clean_text(row.get("Target Name", "unknown_target"))
    organism = _clean_text(row.get("Target Source Organism According to Curator or DataSource", ""))
    return "name:" + stable_hash(f"{target_name}|{organism}", 24)


def load_bindingdb_active_fragments(
    tsv_path: Path,
    active_threshold: float = 6.0,
    max_rows: int | None = None,
    core_min_heavy: int = 8,
    fragment_min_heavy: int = 1,
    fragment_max_heavy: int = 25,
    mol_min_heavy: int = 10,
    mol_max_heavy: int = 80,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Stream BindingDB TSV rows and return active target-conditioned fragment records."""

    _require_rdkit()
    stats: Counter[str] = Counter()
    fragment_cache: dict[str, list[dict[str, object]]] = {}
    records: list[dict[str, object]] = []
    seen_records: set[tuple[object, ...]] = set()

    with tsv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            if max_rows is not None and row_idx > max_rows:
                break
            stats["rows_read"] += 1
            measurements = list(iter_active_measurements(row, active_threshold=active_threshold))
            if not measurements:
                stats["rows_without_active_measurement"] += 1
                continue

            canonical = canonicalize_smiles(row.get("Ligand SMILES", ""))
            if canonical is None:
                stats["rows_with_invalid_smiles"] += 1
                continue

            if canonical not in fragment_cache:
                mol_id = _clean_text(row.get("BindingDB MonomerID", "")) or stable_hash(canonical)
                fragment_cache[canonical] = fragment_smiles_brics(
                    canonical,
                    mol_id=mol_id,
                    core_min_heavy=core_min_heavy,
                    fragment_min_heavy=fragment_min_heavy,
                    fragment_max_heavy=fragment_max_heavy,
                    mol_min_heavy=mol_min_heavy,
                    mol_max_heavy=mol_max_heavy,
                )
            fragments = fragment_cache[canonical]
            if not fragments:
                stats["active_rows_without_fragments"] += 1
                continue

            target_key = target_key_from_row(row)
            target_name = _clean_text(row.get("Target Name", ""))
            reactant_set_id = _clean_text(row.get("BindingDB Reactant_set_id", "")) or f"row_{row_idx}"
            ligand_id = _clean_text(row.get("BindingDB MonomerID", "")) or stable_hash(canonical)
            date_publication = _clean_text(row.get("Date of publication", ""))
            date_bindingdb = _clean_text(row.get("Date in BindingDB", ""))

            stats["active_rows_with_fragments"] += 1
            stats["active_measurements"] += len(measurements)
            for measurement in measurements:
                for fragment in fragments:
                    key = (
                        target_key,
                        measurement["endpoint"],
                        fragment["core_key"],
                        fragment["fragment_smiles"],
                        reactant_set_id,
                    )
                    if key in seen_records:
                        continue
                    seen_records.add(key)
                    rec = dict(fragment)
                    rec.update(
                        {
                            "dataset_source": "BindingDB_BindingDB_Articles_202606",
                            "target_key": target_key,
                            "target_name": target_name,
                            "endpoint": measurement["endpoint"],
                            "relation": measurement["relation"],
                            "value_nm": measurement["value_nm"],
                            "pactivity": measurement["pactivity"],
                            "reactant_set_id": reactant_set_id,
                            "bindingdb_monomer_id": ligand_id,
                            "date_publication": date_publication,
                            "date_bindingdb": date_bindingdb,
                        }
                    )
                    records.append(rec)

    stats["unique_canonical_ligands_fragmented"] = len(fragment_cache)
    stats["active_fragment_records"] = len(records)
    stats["unique_targets"] = len({r["target_key"] for r in records})
    stats["unique_fragments"] = len({r["fragment_smiles"] for r in records})
    stats["unique_context_groups"] = len(
        {
            (r["target_key"], r["endpoint"], r["core_key"], r["attachment_signature"])
            for r in records
        }
    )
    return records, dict(stats)


def build_candidate_matrix_from_fragments(
    fragment_records: list[dict[str, object]],
    dataset: str = "bindingdb_mmp_202606_articles",
    min_active_fragments_per_group: int = 2,
    negative_ratio: int = 5,
    min_candidates_per_query: int = 20,
    max_groups: int | None = None,
    max_queries: int | None = None,
    random_seed: int = 20260609,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = random.Random(random_seed)
    by_group: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    fragments_by_attachment: dict[str, set[str]] = defaultdict(set)
    all_fragments: set[str] = set()

    for rec in fragment_records:
        key = (
            str(rec["target_key"]),
            str(rec["endpoint"]),
            str(rec["core_key"]),
            str(rec["attachment_signature"]),
        )
        by_group[key].append(rec)
        fragment = str(rec["fragment_smiles"])
        attach = str(rec["attachment_signature"])
        fragments_by_attachment[attach].add(fragment)
        all_fragments.add(fragment)

    eligible: list[tuple[tuple[str, str, str, str], list[dict[str, object]], dict[str, set[str]]]] = []
    for key, records in by_group.items():
        frag_ligands: dict[str, set[str]] = defaultdict(set)
        for rec in records:
            frag_ligands[str(rec["fragment_smiles"])].add(str(rec.get("bindingdb_monomer_id") or rec["canonical_smiles"]))
        if len(frag_ligands) >= min_active_fragments_per_group:
            eligible.append((key, records, frag_ligands))

    eligible.sort(key=lambda item: (-sum(len(v) for v in item[2].values()), -len(item[2]), item[0]))
    if max_groups is not None:
        eligible = eligible[:max_groups]

    rows: list[dict[str, object]] = []
    query_seen: set[tuple[str, str]] = set()
    queries_written = 0
    skipped_without_negatives = 0

    for (target_key, endpoint, core_key, attach), records, frag_ligands in eligible:
        rep = records[0]
        active_fragments = sorted(frag_ligands)
        same_attach_pool = sorted(fragments_by_attachment[attach] - set(active_fragments))
        global_pool = sorted(all_fragments - set(active_fragments) - set(same_attach_pool))
        negative_pool = same_attach_pool + global_pool
        rng.shuffle(negative_pool)

        for old_fragment in active_fragments:
            positives = [f for f in active_fragments if f != old_fragment]
            if not positives:
                continue
            requested_negatives = max(
                int(negative_ratio) * len(positives),
                int(min_candidates_per_query) - len(positives),
                0,
            )
            negatives = [f for f in negative_pool if f != old_fragment and f not in positives]
            negatives = negatives[:requested_negatives]
            if not negatives:
                skipped_without_negatives += 1
                continue

            query_id = f"{dataset}:{stable_hash('|'.join([target_key, endpoint, core_key, attach, old_fragment]), 24)}"
            candidates = [(candidate, 1) for candidate in positives] + [(candidate, 0) for candidate in negatives]
            for candidate, label in candidates:
                pair_key = (query_id, candidate)
                if pair_key in query_seen:
                    continue
                query_seen.add(pair_key)
                rows.append(
                    {
                        "dataset": dataset,
                        "query_id": query_id,
                        "old_fragment_smiles": old_fragment,
                        "candidate_smiles": candidate,
                        "label": int(label),
                        "target_key": target_key,
                        "target_name": rep.get("target_name", ""),
                        "endpoint": endpoint,
                        "core_key": core_key,
                        "core_smiles": rep.get("core_smiles", ""),
                        "attachment_signature": attach,
                        "n_active_fragments_in_group": len(active_fragments),
                        "n_support_ligands_for_old_fragment": len(frag_ligands[old_fragment]),
                        "n_positive_candidates": len(positives),
                    }
                )
            queries_written += 1
            if max_queries is not None and queries_written >= max_queries:
                break
        if max_queries is not None and queries_written >= max_queries:
            break

    matrix = pd.DataFrame(rows)
    audit = {
        "input_fragment_records": len(fragment_records),
        "context_groups": len(by_group),
        "eligible_groups": len(eligible),
        "queries": int(matrix["query_id"].nunique()) if not matrix.empty else 0,
        "rows": int(len(matrix)),
        "positive_rows": int(matrix["label"].sum()) if not matrix.empty else 0,
        "negative_rows": int((matrix["label"] == 0).sum()) if not matrix.empty else 0,
        "old_fragments": int(matrix["old_fragment_smiles"].nunique()) if not matrix.empty else 0,
        "candidate_fragments": int(matrix["candidate_smiles"].nunique()) if not matrix.empty else 0,
        "skipped_queries_without_negatives": skipped_without_negatives,
        "random_seed": random_seed,
    }
    return matrix, audit


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path | None, expected_md5_path: Path | None = None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    info: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "md5": md5_file(path),
    }
    if expected_md5_path is not None and expected_md5_path.exists():
        expected = expected_md5_path.read_text(encoding="utf-8").strip().split()[0].lower()
        info["expected_md5"] = expected
        info["md5_matches_expected"] = info["md5"] == expected
    return info


def write_audit_markdown(path: Path, manifest: dict[str, object]) -> None:
    parse_audit = manifest["parse_audit"]
    matrix_audit = manifest["matrix_audit"]
    lines = [
        "# BindingDB MMP Matrix Audit",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        f"Dataset: `{manifest['dataset']}`",
        "",
        "## Source",
        "",
        f"- Download page: {DOWNLOAD_PAGE_URL}",
        f"- Direct zip: {DIRECT_ZIP_URL}",
        f"- Direct md5: {DIRECT_MD5_URL}",
        f"- Zip MD5 match: {manifest.get('source_zip', {}).get('md5_matches_expected')}",
        "",
        "## Parse Counts",
        "",
    ]
    for key in sorted(parse_audit):
        lines.append(f"- `{key}`: {parse_audit[key]}")
    lines.extend(["", "## Matrix Counts", ""])
    for key in sorted(matrix_audit):
        lines.append(f"- `{key}`: {matrix_audit[key]}")
    lines.extend(
        [
            "",
            "## Label Semantics",
            "",
            "Positive labels mean that active BindingDB compounds support an active-active "
            "single-cut BRICS replacement within the same target, endpoint, core, and "
            "attachment signature. Zero labels are in-matrix unlabeled candidates sampled "
            "from other compatible fragment contexts; they are not asserted inactive.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def default_paths() -> dict[str, Path]:
    base = Path(__file__).resolve().parent / "bindingdb_mmp"
    raw = base / "raw"
    return {
        "out_dir": base,
        "tsv": raw / "extracted" / "BindingDB_BindingDB_Articles.tsv",
        "zip": raw / "BindingDB_BindingDB_Articles_202606_tsv.zip",
        "md5": raw / "BindingDB_BindingDB_Articles_202606_tsv.md5",
    }


def parse_args() -> argparse.Namespace:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindingdb-tsv", type=Path, default=paths["tsv"])
    parser.add_argument("--source-zip", type=Path, default=paths["zip"])
    parser.add_argument("--source-md5", type=Path, default=paths["md5"])
    parser.add_argument("--out-dir", type=Path, default=paths["out_dir"])
    parser.add_argument("--dataset", default="bindingdb_mmp_202606_articles")
    parser.add_argument("--matrix-name", default="candidate_matrix.csv.gz")
    parser.add_argument("--active-threshold", type=float, default=6.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--core-min-heavy", type=int, default=8)
    parser.add_argument("--fragment-min-heavy", type=int, default=1)
    parser.add_argument("--fragment-max-heavy", type=int, default=25)
    parser.add_argument("--mol-min-heavy", type=int, default=10)
    parser.add_argument("--mol-max-heavy", type=int, default=80)
    parser.add_argument("--min-active-fragments-per-group", type=int, default=2)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--min-candidates-per-query", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260609)
    parser.add_argument("--write-fragment-records", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fragment_records, parse_audit = load_bindingdb_active_fragments(
        args.bindingdb_tsv,
        active_threshold=args.active_threshold,
        max_rows=args.max_rows,
        core_min_heavy=args.core_min_heavy,
        fragment_min_heavy=args.fragment_min_heavy,
        fragment_max_heavy=args.fragment_max_heavy,
        mol_min_heavy=args.mol_min_heavy,
        mol_max_heavy=args.mol_max_heavy,
    )
    matrix, matrix_audit = build_candidate_matrix_from_fragments(
        fragment_records,
        dataset=args.dataset,
        min_active_fragments_per_group=args.min_active_fragments_per_group,
        negative_ratio=args.negative_ratio,
        min_candidates_per_query=args.min_candidates_per_query,
        max_groups=args.max_groups,
        max_queries=args.max_queries,
        random_seed=args.random_seed,
    )

    matrix_path = args.out_dir / args.matrix_name
    matrix.to_csv(matrix_path, index=False)

    fragment_path = None
    if args.write_fragment_records:
        fragment_path = args.out_dir / "active_fragment_records.csv.gz"
        pd.DataFrame(fragment_records).to_csv(fragment_path, index=False)

    manifest = {
        "dataset": args.dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": "BindingDB curated article TSV; ChEMBL-independent main external source.",
        "download_page_url": DOWNLOAD_PAGE_URL,
        "direct_zip_url": DIRECT_ZIP_URL,
        "direct_md5_url": DIRECT_MD5_URL,
        "source_zip": _file_info(args.source_zip, args.source_md5),
        "source_tsv": _file_info(args.bindingdb_tsv),
        "output_matrix": _file_info(matrix_path),
        "fragment_records": _file_info(fragment_path) if fragment_path else None,
        "parameters": {
            "active_threshold": args.active_threshold,
            "max_rows": args.max_rows,
            "max_groups": args.max_groups,
            "max_queries": args.max_queries,
            "core_min_heavy": args.core_min_heavy,
            "fragment_min_heavy": args.fragment_min_heavy,
            "fragment_max_heavy": args.fragment_max_heavy,
            "mol_min_heavy": args.mol_min_heavy,
            "mol_max_heavy": args.mol_max_heavy,
            "min_active_fragments_per_group": args.min_active_fragments_per_group,
            "negative_ratio": args.negative_ratio,
            "min_candidates_per_query": args.min_candidates_per_query,
            "random_seed": args.random_seed,
        },
        "parse_audit": parse_audit,
        "matrix_audit": matrix_audit,
    }
    manifest_path = args.out_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_audit_markdown(args.out_dir / "matrix_audit.md", manifest)

    print(json.dumps({"matrix": str(matrix_path), "manifest": str(manifest_path), **matrix_audit}, indent=2))


if __name__ == "__main__":
    main()
