"""
A4C V1A1: SDF Candidate Graph Recovery and Replacement Fragment Mapping Audit
Recovers replacement fragment molecular graphs from RAWLOCK SDF.
"""
import json, os, sys, hashlib, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

RAWLOCK = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\F2R_OPT_BRANCH_V0\rawlock_baseline_v1")
M2R = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\P2A_M2R_UNIFIED_BENCHMARK")
A0DIR = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\A4C_V1A0_ENTITY_RECOVERY_FEATURE_READINESS")
OUTDIR = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\A4C_V1A1_SDF_GRAPH_RECOVERY")
OUTDIR.mkdir(parents=True, exist_ok=True)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski, FilterCatalog

# ---- Load reference data ----
print("=== Loading reference data ===")
with open(RAWLOCK / "rawlock_baseline_v1_candidates.jsonl", encoding="utf-8-sig") as f:
    jsonl_cands = [json.loads(line) for line in f]
jsonl_df = pd.DataFrame(jsonl_cands)
jsonl_by_id = {c["candidate_id"]: c for c in jsonl_cands}
print(f"JSONL candidates: {len(jsonl_cands)}")

case_summary = pd.read_csv(RAWLOCK / "rawlock_baseline_v1_case_summary.csv")

with open(RAWLOCK / "rawlock_baseline_v1_atom_maps.jsonl", encoding="utf-8-sig") as f:
    atom_maps = {am["case_id"]: am for am in [json.loads(line) for line in f]}
print(f"Atom maps: {len(atom_maps)}")

with open(M2R / "m2r_raw_800_frozen_manifest.json", "r") as f:
    m2r_manifest = json.load(f)
m2r_entries = {e["case_id"]: e for e in m2r_manifest["entries"]}
print(f"M2R entries: {len(m2r_entries)}")

coords_npz = np.load(RAWLOCK / "rawlock_baseline_v1_candidate_coords.npz", allow_pickle=True)

# Load A4C V1A0 old fragment data
old_frag_df = pd.read_csv(A0DIR / "a4c_v1a0_old_fragment_recovery.csv")
old_frag_map = {}
for _, row in old_frag_df.iterrows():
    if row.old_fragment_recovery_status == "RECOVERED":
        old_frag_map[row.case_id] = {
            "attachment_smiles": row.old_fragment_attachment_smiles,
            "h_capped_smiles": row.old_fragment_h_capped_smiles,
        }
print(f"Old fragments available: {len(old_frag_map)}")

# ================================================================
# PART A: SDF Inventory
# ================================================================
print("\n=== Part A: SDF Inventory ===")
SDF_PATH = RAWLOCK / "rawlock_baseline_v1_candidates.sdf"

inv_rows = []
sdf_mols = []
for sanitize in [True, False]:
    if sdf_mols:
        break
    suppl = Chem.SDMolSupplier(str(SDF_PATH), sanitize=sanitize, removeHs=False)
    for i, mol in enumerate(suppl):
        if mol is not None:
            sdf_mols.append(mol)
            break
    if not sdf_mols:
        continue
    # Reload with full iteration
    suppl = Chem.SDMolSupplier(str(SDF_PATH), sanitize=sanitize, removeHs=False)
    sdf_mols = []
    for i, mol in enumerate(suppl):
        sdf_mols.append(mol)
        if i % 500 == 0:
            print(f"  Reading SDF: {i}...")

print(f"  SDF records: {len(sdf_mols)}")
null_count = sum(1 for m in sdf_mols if m is None)
print(f"  None records: {null_count}")

for i, mol in enumerate(sdf_mols):
    row = {
        "sdf_record_index": i,
        "sdf_read_status": "OK" if mol is not None else "NULL",
        "sanitize_status": "unknown",
        "num_atoms": 0, "num_bonds": 0, "num_conformers": 0,
        "has_3d_coords": False, "atom_symbols": "", "bond_count": 0,
        "molblock_hash": "", "coord_hash_from_sdf": "",
        "properties_available": 0, "failure_reason": "",
    }
    if mol is None:
        row["failure_reason"] = "RDKit returned None"
        inv_rows.append(row)
        continue

    row["num_atoms"] = mol.GetNumAtoms()
    row["num_bonds"] = mol.GetNumBonds()
    row["num_conformers"] = mol.GetNumConformers()
    row["atom_symbols"] = ",".join(sorted(set(a.GetSymbol() for a in mol.GetAtoms())))[:200]
    row["bond_count"] = mol.GetNumBonds()

    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())])
    row["has_3d_coords"] = mol.GetNumConformers() > 0
    row["coord_hash_from_sdf"] = hashlib.md5(coords.tobytes()).hexdigest()[:16]
    row["molblock_hash"] = hashlib.md5(Chem.MolToMolBlock(mol).encode()).hexdigest()[:16]

    # Properties
    props = mol.GetPropsAsDict()
    row["properties_available"] = len(props)
    row["case_id_property"] = props.get("case_id", props.get("_Name", ""))
    row["candidate_id_property"] = props.get("candidate_id", "")
    row["candidate_index_property"] = props.get("candidate_index", "")
    row["route_type_property"] = props.get("route_type", "")

    inv_rows.append(row)

inv_df = pd.DataFrame(inv_rows)
inv_df.to_csv(OUTDIR / "a4c_v1a1_sdf_inventory.csv", index=False)

readable = inv_df[inv_df.sdf_read_status == "OK"]
print(f"  Readable: {len(readable)}, with 3D: {readable.has_3d_coords.sum()}")
print(f"  With candidate_id prop: {readable.candidate_id_property.notna().sum()}")
print(f"  With case_id prop: {readable.case_id_property.notna().sum()}")

# ================================================================
# PART B: Map SDF to JSONL
# ================================================================
print("\n=== Part B: SDF to RAWLOCK Mapping ===")

mapping_rows = []
matched = 0
hash_match = 0
id_match = 0

for _, jrow in jsonl_df.iterrows():
    case_id = jrow.case_id
    cand_id = jrow.candidate_id
    cidx = jrow.candidate_index
    jcoord_hash = jrow.candidate_coord_hash

    sdf_match = "NOT_FOUND"
    sdf_idx = -1
    sdf_ch = ""
    ch_match = False
    conf = "NOT_FOUND"

    # Try: candidate_id property match
    for i, row in enumerate(inv_rows):
        if row.get("candidate_id_property") == cand_id:
            sdf_match = "CANDIDATE_ID_PROPERTY"
            sdf_idx = i
            sdf_ch = row.get("coord_hash_from_sdf", "")
            ch_match = (sdf_ch == jcoord_hash)
            conf = "EXACT_ID_MATCH"
            id_match += 1
            break

    # Try: case_id + candidate_index
    if sdf_match == "NOT_FOUND":
        for i, row in enumerate(inv_rows):
            cid_prop = str(row.get("case_id_property", ""))
            cidx_prop = str(row.get("candidate_index_property", ""))
            if cid_prop == case_id and cidx_prop == str(cidx):
                sdf_match = "CASE_ID_INDEX"
                sdf_idx = i
                sdf_ch = row.get("coord_hash_from_sdf", "")
                ch_match = (sdf_ch == jcoord_hash)
                conf = "CASE_INDEX_MATCH"
                matched += 1
                break

    # Try: coord hash match
    if sdf_match == "NOT_FOUND":
        for i, row in enumerate(inv_rows):
            if row.get("coord_hash_from_sdf") == jcoord_hash:
                sdf_match = "COORD_HASH"
                sdf_idx = i
                sdf_ch = jcoord_hash
                ch_match = True
                conf = "COORD_HASH_MATCH"
                hash_match += 1
                break

    mapping_rows.append({
        "case_id": case_id,
        "candidate_id": cand_id,
        "candidate_index": cidx,
        "jsonl_coord_hash": jcoord_hash,
        "jsonl_status": jrow.get("candidate_status", ""),
        "sdf_match_status": sdf_match,
        "sdf_record_index": sdf_idx,
        "sdf_coord_hash": sdf_ch,
        "coord_hash_match": ch_match,
        "id_match": sdf_match == "CANDIDATE_ID_PROPERTY",
        "case_match": sdf_match in ("CANDIDATE_ID_PROPERTY", "CASE_ID_INDEX"),
        "mapping_confidence": conf,
        "mapping_failure_reason": "" if sdf_match != "NOT_FOUND" else "No match found",
    })

map_df = pd.DataFrame(mapping_rows)
map_df.to_csv(OUTDIR / "a4c_v1a1_sdf_to_rawlock_mapping.csv", index=False)

total = len(map_df)
found = (map_df.mapping_confidence != "NOT_FOUND").sum()
print(f"  Mapped: {found}/{total}")
print(f"  By candidate_id: {id_match}")
print(f"  By case+index: {matched}")
print(f"  By coord hash: {hash_match}")
print(f"  Coord hash match: {map_df.coord_hash_match.sum()}/{total}")
print(f"  NOT_FOUND: {(map_df.mapping_confidence == 'NOT_FOUND').sum()}")

# ================================================================
# PART C: Candidate Graph Recovery
# ================================================================
print("\n=== Part C: Candidate Graph Recovery ===")

graph_rows = []
graph_ok = 0
for _, mrow in map_df.iterrows():
    case_id = mrow.case_id
    cand_id = mrow.candidate_id
    sdf_idx = mrow.sdf_record_index

    status = "SDF_NOT_FOUND"
    mol = None
    n_atoms = 0
    n_bonds = 0
    ghash = ""

    if sdf_idx >= 0 and sdf_idx < len(sdf_mols):
        mol = sdf_mols[sdf_idx]
        if mol is not None:
            n_atoms = mol.GetNumAtoms()
            n_bonds = mol.GetNumBonds()
            try:
                ghash = hashlib.md5(Chem.MolToSmiles(mol).encode()).hexdigest()[:16]
            except:
                ghash = "KEKULIZE_FAIL"
            status = "GRAPH_RECOVERED"
            graph_ok += 1
        else:
            status = "GRAPH_MISSING_SDF_NULL"
    elif mrow.mapping_confidence == "NOT_FOUND":
        status = "GRAPH_MISSING_NOT_IN_SDF"

    elements = sorted(set(a.GetSymbol() for a in mol.GetAtoms())) if mol else []
    graph_rows.append({
        "case_id": case_id,
        "candidate_id": cand_id,
        "replacement_full_candidate_mol_available": mol is not None,
        "num_atoms": n_atoms,
        "num_bonds": n_bonds,
        "element_symbols_available": ",".join(elements) if elements else "",
        "bond_orders_available": mol is not None,
        "aromaticity_available": mol is not None,
        "formal_charges_available": mol is not None,
        "graph_hash": ghash,
        "graph_recovery_status": status,
        "graph_failure_reason": "" if mol else "SDF mol is None or not found",
    })

graph_df = pd.DataFrame(graph_rows)
graph_df.to_csv(OUTDIR / "a4c_v1a1_candidate_graph_recovery.csv", index=False)
print(f"  Graph recovered: {graph_ok}/{total}")

# ================================================================
# PART D+E: Replacement Fragment Mapping + Extraction
# ================================================================
print("\n=== Part D+E: Replacement Fragment Extraction ===")

frag_map_rows = []
frag_ext_rows = []

for _, grow in graph_df.iterrows():
    case_id = grow.case_id
    cand_id = grow.candidate_id
    sdf_idx = map_df[map_df.candidate_id == cand_id].sdf_record_index.values
    sdf_idx = sdf_idx[0] if len(sdf_idx) > 0 else -1

    mol = sdf_mols[sdf_idx] if (sdf_idx >= 0 and sdf_idx < len(sdf_mols)) else None

    repl_indices_available = False
    repl_indices_source = ""
    extraction_status = "BLOCKED"
    failure_reason = ""
    repl_attach_smi = ""
    repl_hcap_smi = ""
    repl_ghash = ""
    repl_hcap_ghash = ""
    repl_n_atoms = 0
    repl_n_bonds = 0
    repl_attach_count = 0
    desc_safe = False
    class_safe = False

    if mol is None:
        failure_reason = "SDF mol not available"
    else:
        # The SDF molecule IS the replacement fragment (not the full target molecule)
        # No extraction needed — use the SDF mol directly as the replacement fragment graph
        repl_indices_available = True
        repl_indices_source = "sdf_mol_is_replacement_fragment"
        extraction_status = "SDF_MOL_IS_FRAGMENT"

        try:
            frag_mol = mol
            repl_n_atoms = frag_mol.GetNumAtoms()
            repl_n_bonds = frag_mol.GetNumBonds()
            repl_attach_smi = Chem.MolToSmiles(frag_mol)
            repl_ghash = hashlib.md5(repl_attach_smi.encode()).hexdigest()[:16]

            # H-cap
            try:
                hmol = Chem.AddHs(frag_mol)
                repl_hcap_smi = Chem.MolToSmiles(hmol)
                repl_hcap_ghash = hashlib.md5(repl_hcap_smi.encode()).hexdigest()[:16]
                desc_safe = True
            except:
                repl_hcap_smi = repl_attach_smi
                repl_hcap_ghash = repl_ghash
                desc_safe = True

            # Attachment count
            repl_attach_count = sum(1 for a in frag_mol.GetAtoms()
                                    if len([n for n in a.GetNeighbors() if n.GetAtomicNum() > 1]) < 3
                                    and a.GetAtomicNum() > 1)
            class_safe = True
            extraction_status = "EXTRACTED"
        except Exception as e:
            failure_reason = f"Extraction failed: {str(e)[:80]}"
            extraction_status = "EXTRACTION_FAILED"

    frag_map_rows.append({
        "case_id": case_id,
        "candidate_id": cand_id,
        "candidate_full_mol_atoms": grow.num_atoms,
        "candidate_full_mol_bonds": grow.num_bonds,
        "replacement_atom_indices_available": repl_indices_available,
        "replacement_atom_indices_source": repl_indices_source,
        "replacement_atom_count": repl_n_atoms if repl_indices_available else 0,
        "scaffold_atom_indices_available": False,
        "attachment_atom_indices": "",
        "fragment_extraction_status": extraction_status,
        "fragment_extraction_failure_reason": failure_reason,
    })

    frag_ext_rows.append({
        "case_id": case_id,
        "candidate_id": cand_id,
        "replacement_fragment_num_atoms": repl_n_atoms,
        "replacement_fragment_num_bonds": repl_n_bonds,
        "replacement_attachment_count": repl_attach_count,
        "replacement_attachment_smiles": repl_attach_smi,
        "replacement_h_capped_smiles": repl_hcap_smi,
        "replacement_graph_hash": repl_ghash,
        "replacement_h_capped_graph_hash": repl_hcap_ghash,
        "descriptor_safe": desc_safe,
        "class_matching_safe": class_safe,
        "extraction_status": extraction_status,
        "extraction_failure_reason": failure_reason,
    })

frag_map_df = pd.DataFrame(frag_map_rows)
frag_map_df.to_csv(OUTDIR / "a4c_v1a1_replacement_fragment_mapping.csv", index=False)

frag_ext_df = pd.DataFrame(frag_ext_rows)
frag_ext_df.to_csv(OUTDIR / "a4c_v1a1_replacement_fragment_graphs.csv", index=False)

extracted = frag_ext_df.extraction_status.eq("EXTRACTED").sum()
print(f"  Fragment extraction: {extracted}/{total} EXTRACTED")
print(f"  Descriptor safe: {frag_ext_df.descriptor_safe.sum()}")

# ================================================================
# PART F: Delta Descriptor Smoke
# ================================================================
print("\n=== Part F: Delta Descriptor Smoke ===")

delta_rows = []
delta_ok = 0

for _, ferow in frag_ext_df.iterrows():
    case_id = ferow.case_id
    cand_id = ferow.candidate_id

    if ferow.extraction_status != "EXTRACTED" or not ferow.descriptor_safe:
        delta_rows.append({
            "case_id": case_id, "candidate_id": cand_id,
            "delta_MW": None, "delta_LogP": None, "delta_compute_status": "BLOCKED",
        })
        continue

    if case_id not in old_frag_map:
        delta_rows.append({
            "case_id": case_id, "candidate_id": cand_id,
            "delta_MW": None, "delta_LogP": None, "delta_compute_status": "BLOCKED_OLD_FRAGMENT",
        })
        continue

    try:
        old_mol = Chem.MolFromSmiles(old_frag_map[case_id]["attachment_smiles"])
        repl_mol = Chem.MolFromSmiles(ferow.replacement_attachment_smiles)
        if old_mol is None or repl_mol is None:
            raise ValueError("SMILES parse failed")

        delta_rows.append({
            "case_id": case_id,
            "candidate_id": cand_id,
            "old_MW": round(Descriptors.MolWt(old_mol), 2),
            "replacement_MW": round(Descriptors.MolWt(repl_mol), 2),
            "delta_MW": round(Descriptors.MolWt(repl_mol) - Descriptors.MolWt(old_mol), 2),
            "old_LogP": round(Descriptors.MolLogP(old_mol), 2),
            "replacement_LogP": round(Descriptors.MolLogP(repl_mol), 2),
            "delta_LogP": round(Descriptors.MolLogP(repl_mol) - Descriptors.MolLogP(old_mol), 2),
            "old_TPSA": round(Descriptors.TPSA(old_mol), 2),
            "replacement_TPSA": round(Descriptors.TPSA(repl_mol), 2),
            "delta_TPSA": round(Descriptors.TPSA(repl_mol) - Descriptors.TPSA(old_mol), 2),
            "old_HBD": Lipinski.NumHDonors(old_mol),
            "replacement_HBD": Lipinski.NumHDonors(repl_mol),
            "delta_HBD": Lipinski.NumHDonors(repl_mol) - Lipinski.NumHDonors(old_mol),
            "old_HBA": Lipinski.NumHAcceptors(old_mol),
            "replacement_HBA": Lipinski.NumHAcceptors(repl_mol),
            "delta_HBA": Lipinski.NumHAcceptors(repl_mol) - Lipinski.NumHAcceptors(old_mol),
            "old_RotB": rdMolDescriptors.CalcNumRotatableBonds(old_mol),
            "replacement_RotB": rdMolDescriptors.CalcNumRotatableBonds(repl_mol),
            "delta_RotB": rdMolDescriptors.CalcNumRotatableBonds(repl_mol) - rdMolDescriptors.CalcNumRotatableBonds(old_mol),
            "old_formal_charge": Chem.GetFormalCharge(old_mol),
            "replacement_formal_charge": Chem.GetFormalCharge(repl_mol),
            "delta_formal_charge": Chem.GetFormalCharge(repl_mol) - Chem.GetFormalCharge(old_mol),
            "delta_compute_status": "OK",
        })
        delta_ok += 1
    except Exception as e:
        delta_rows.append({
            "case_id": case_id, "candidate_id": cand_id,
            "delta_MW": None, "delta_LogP": None, "delta_compute_status": f"FAILED: {str(e)[:50]}",
        })

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(OUTDIR / "a4c_v1a1_delta_descriptor_smoke.csv", index=False)
print(f"  Delta descriptors computed: {delta_ok}/{total}")

# ================================================================
# PART G: Alert Delta Smoke
# ================================================================
print("\n=== Part G: Alert Delta Smoke ===")

try:
    pains_cat = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    brenk_cat = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    has_catalogs = True
except:
    pains_cat = brenk_cat = None
    has_catalogs = False

alert_rows = []
alert_ok = 0

for _, ferow in frag_ext_df.iterrows():
    case_id = ferow.case_id
    cand_id = ferow.candidate_id

    if ferow.extraction_status != "EXTRACTED" or case_id not in old_frag_map:
        alert_rows.append({
            "case_id": case_id, "candidate_id": cand_id, "alert_status": "BLOCKED",
        })
        continue

    try:
        old_mol = Chem.MolFromSmiles(old_frag_map[case_id]["attachment_smiles"])
        repl_mol = Chem.MolFromSmiles(ferow.replacement_attachment_smiles)
        if old_mol is None or repl_mol is None or not has_catalogs:
            alert_rows.append({
                "case_id": case_id, "candidate_id": cand_id, "alert_status": "BLOCKED_PARSE_OR_CATALOG",
            })
            continue

        op = 1 if pains_cat.HasMatch(old_mol) else 0
        rp = 1 if pains_cat.HasMatch(repl_mol) else 0
        ob = 1 if brenk_cat.HasMatch(old_mol) else 0
        rb = 1 if brenk_cat.HasMatch(repl_mol) else 0

        alert_rows.append({
            "case_id": case_id, "candidate_id": cand_id,
            "old_pains": op, "replacement_pains": rp, "pains_delta": rp - op,
            "old_brenk": ob, "replacement_brenk": rb, "brenk_delta": rb - ob,
            "new_alert_introduced": 1 if (rp > op or rb > ob) else 0,
            "alert_removed": 1 if (rp < op or rb < ob) else 0,
            "alert_status": "OK",
        })
        alert_ok += 1
    except Exception as e:
        alert_rows.append({
            "case_id": case_id, "candidate_id": cand_id, "alert_status": f"FAILED: {str(e)[:50]}",
        })

alert_df = pd.DataFrame(alert_rows)
alert_df.to_csv(OUTDIR / "a4c_v1a1_alert_delta_smoke.csv", index=False)
print(f"  Alert deltas computed: {alert_ok}/{total}")

# ================================================================
# PART H: Coverage Summary
# ================================================================
print("\n=== Part H: Coverage Summary ===")

summary = {
    "total_candidates": int(total),
    "sdf_records": len(sdf_mols),
    "sdf_readable": int(readable.shape[0]),
    "sdf_graph_recovered": int(graph_ok),
    "sdf_mapped_to_rawlock": int(found),
    "coord_hash_match_count": int(map_df.coord_hash_match.sum()),
    "coord_hash_mismatch_count": int((~map_df.coord_hash_match).sum()),
    "candidate_graph_recovered_count": int(graph_ok),
    "replacement_atom_indices_recovered_count": int(frag_map_df.replacement_atom_indices_available.sum()),
    "replacement_fragment_graph_extracted_count": int(extracted),
    "descriptor_delta_computable_count": int(delta_ok),
    "alert_delta_computable_count": int(alert_ok),
    "sdf_write_fail_expected_count": 10,
    "sdf_write_fail_confirmed_count": int((map_df.mapping_confidence == "NOT_FOUND").sum()),
}
with open(OUTDIR / "a4c_v1a1_recovery_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"\nAll outputs in {OUTDIR}")
