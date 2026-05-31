"""
A4C V1A: Full Replacement-Level Feature Audit
Parts A-J: candidate join → 2D delta → alert delta → class prior →
3D shape → pharmacophore → entropy → proxy → hard-filter → decision.
"""
import json, os, sys, hashlib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

RAWLOCK = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\F2R_OPT_BRANCH_V0\rawlock_baseline_v1")
A0DIR  = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\A4C_V1A0_ENTITY_RECOVERY_FEATURE_READINESS")
A1DIR  = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\A4C_V1A1_SDF_GRAPH_RECOVERY")
OUTDIR = Path(r"E:\zuhui\bioisosteric_diffusion\plan_results\A4C_V1A_FULL_FEATURE_AUDIT")
OUTDIR.mkdir(parents=True, exist_ok=True)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski, FilterCatalog, Crippen

# ---- Load all reference data ----
print("=== Loading data ===")
old_frag = pd.read_csv(A0DIR / "a4c_v1a0_old_fragment_recovery.csv")
three_d   = pd.read_csv(A0DIR / "a4c_v1a0_3d_feature_readiness.csv")
graphs    = pd.read_csv(A1DIR / "a4c_v1a1_candidate_graph_recovery.csv")
frags_ext = pd.read_csv(A1DIR / "a4c_v1a1_replacement_fragment_graphs.csv")
mapping   = pd.read_csv(A1DIR / "a4c_v1a1_sdf_to_rawlock_mapping.csv")
csum      = pd.read_csv(RAWLOCK / "rawlock_baseline_v1_case_summary.csv")

# Build per-candidate joined view
old_map = {}
for _, r in old_frag.iterrows():
    if r.old_fragment_recovery_status == "RECOVERED":
        old_map[r.case_id] = {"attach_smi": r.old_fragment_attachment_smiles, "hcap_smi": r.old_fragment_h_capped_smiles}

repl_map = {}
for _, r in frags_ext.iterrows():
    key = (r.case_id, r.candidate_id)
    repl_map[key] = {"attach_smi": r.replacement_attachment_smiles, "hcap_smi": r.replacement_h_capped_smiles,
                     "n_atoms": r.replacement_fragment_num_atoms, "n_bonds": r.replacement_fragment_num_bonds,
                     "status": r.extraction_status}

three_d_map = {}
for _, r in three_d.iterrows():
    key = (r.case_id, r.candidate_id)
    three_d_map[key] = {"rg_old": r.radius_of_gyration_old, "rg_repl": r.radius_of_gyration_replacement,
                        "pm_old": r.principal_moments_old, "pm_repl": r.principal_moments_replacement,
                        "shape_ready": r.shape_similarity_ready}

graph_map = {}
for _, r in graphs.iterrows():
    key = (r.case_id, r.candidate_id)
    graph_map[key] = {"n_atoms": r.num_atoms, "n_bonds": r.num_bonds, "status": r.graph_recovery_status}

# Catalogs
try:
    pains_cat = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    brenk_cat = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    has_alerts = True
except:
    has_alerts = False

n_total = 0
n_both_2d = 0
n_both_3d = 0
n_sdf_miss = 0

print(f"Old fragments: {len(old_map)}")
print(f"Replacement fragments: {len(repl_map)}")
print(f"3D records: {len(three_d_map)}")

# ================================================================
# PARTS A-D: Candidate join, 2D delta, alert delta, class prior
# ================================================================
print("\n=== Parts A-D: Computing features ===")

join_rows = []
delta_rows = []
alert_rows = []
class_rows = []

with open(RAWLOCK / "rawlock_baseline_v1_candidates.jsonl", encoding="utf-8-sig") as f:
    cands = [json.loads(line) for line in f]

for c in cands:
    case_id = c["case_id"]
    cand_id = c["candidate_id"]
    n_total += 1

    old = old_map.get(case_id)
    repl = repl_map.get((case_id, cand_id))
    td = three_d_map.get((case_id, cand_id))
    gr = graph_map.get((case_id, cand_id))
    mp = mapping[mapping.candidate_id == cand_id]

    old_2d_ok = old is not None
    repl_2d_ok = repl is not None and repl["status"] == "EXTRACTED"
    old_3d_ok = td is not None and td["shape_ready"]
    repl_3d_ok = td is not None
    sdf_ok = len(mp) > 0 and mp.iloc[0].mapping_confidence != "NOT_FOUND"

    if repl_2d_ok: n_both_2d += 1
    if old_3d_ok and repl_3d_ok: n_both_3d += 1
    if not sdf_ok: n_sdf_miss += 1

    csum_row = csum[csum.case_id == case_id]
    rmsd = c["candidate_rmsd"]
    is_oracle = len(csum_row) > 0 and csum_row.iloc[0].oracle_candidate_id == cand_id
    is_deploy = len(csum_row) > 0 and csum_row.iloc[0].deploy_candidate_id == cand_id

    join_rows.append({
        "case_id": case_id, "candidate_id": cand_id, "candidate_index": c["candidate_index"],
        "old_fragment_2d_status": "OK" if old_2d_ok else "MISSING",
        "old_fragment_3d_status": "OK" if old_3d_ok else "MISSING",
        "replacement_graph_status": "OK" if repl_2d_ok else ("SDF_WRITE_FAIL" if not sdf_ok else "MISSING"),
        "replacement_3d_status": "OK" if repl_3d_ok else "MISSING",
        "sdf_mapping_status": "OK" if sdf_ok else "NOT_FOUND",
        "capping_status": "OK" if (old_2d_ok and repl_2d_ok) else "MISSING",
        "rawlock_deploy_flag": is_deploy,
        "rawlock_oracle_flag": is_oracle,
        "candidate_rmsd": rmsd,
        "clean_flag": c.get("clean_flag", True),
        "clash_count": c.get("clash_count", 0),
    })

    # --- 2D delta ---
    if old_2d_ok and repl_2d_ok:
        try:
            old_mol = Chem.MolFromSmiles(old["attach_smi"])
            repl_mol = Chem.MolFromSmiles(repl["attach_smi"])
            if old_mol and repl_mol:
                omw, rmw = Descriptors.MolWt(old_mol), Descriptors.MolWt(repl_mol)
                olp, rlp = Descriptors.MolLogP(old_mol), Descriptors.MolLogP(repl_mol)
                otpsa, rtpsa = Descriptors.TPSA(old_mol), Descriptors.TPSA(repl_mol)
                ohbd, rhbd = Lipinski.NumHDonors(old_mol), Lipinski.NumHDonors(repl_mol)
                ohba, rhba = Lipinski.NumHAcceptors(old_mol), Lipinski.NumHAcceptors(repl_mol)
                orot, rrot = rdMolDescriptors.CalcNumRotatableBonds(old_mol), rdMolDescriptors.CalcNumRotatableBonds(repl_mol)
                ochg, rchg = Chem.GetFormalCharge(old_mol), Chem.GetFormalCharge(repl_mol)
                oha, rha = old_mol.GetNumHeavyAtoms(), repl_mol.GetNumHeavyAtoms()
                oring, rring = rdMolDescriptors.CalcNumRings(old_mol), rdMolDescriptors.CalcNumRings(repl_mol)
                oaro, raro = rdMolDescriptors.CalcNumAromaticRings(old_mol), rdMolDescriptors.CalcNumAromaticRings(repl_mol)
                ohet = sum(1 for a in old_mol.GetAtoms() if a.GetAtomicNum() not in (6,1))
                rhet = sum(1 for a in repl_mol.GetAtoms() if a.GetAtomicNum() not in (6,1))
                try:
                    ofsp3 = rdMolDescriptors.CalcFractionCSP3(old_mol)
                    rfsp3 = rdMolDescriptors.CalcFractionCSP3(repl_mol)
                except:
                    ofsp3 = rfsp3 = None

                lp_dir = "increase" if rlp > olp + 0.1 else ("decrease" if rlp < olp - 0.1 else "neutral")
                chg_dir = "increase" if rchg > ochg else ("decrease" if rchg < ochg else "neutral")
                size_dir = "larger" if rmw > omw + 10 else ("smaller" if rmw < omw - 10 else "similar")
                pol_dir = "more_polar" if rtpsa > otpsa + 10 else ("less_polar" if rtpsa < otpsa - 10 else "similar")

                delta_rows.append({
                    "case_id": case_id, "candidate_id": cand_id,
                    "old_MW": round(omw,2), "replacement_MW": round(rmw,2), "delta_MW": round(rmw-omw,2), "abs_delta_MW": round(abs(rmw-omw),2),
                    "old_LogP": round(olp,2), "replacement_LogP": round(rlp,2), "delta_LogP": round(rlp-olp,2), "abs_delta_LogP": round(abs(rlp-olp),2),
                    "old_TPSA": round(otpsa,2), "replacement_TPSA": round(rtpsa,2), "delta_TPSA": round(rtpsa-otpsa,2), "abs_delta_TPSA": round(abs(rtpsa-otpsa),2),
                    "old_HBD": ohbd, "replacement_HBD": rhbd, "delta_HBD": rhbd-ohbd,
                    "old_HBA": ohba, "replacement_HBA": rhba, "delta_HBA": rhba-ohba,
                    "old_RotB": orot, "replacement_RotB": rrot, "delta_RotB": rrot-orot,
                    "old_formal_charge": ochg, "replacement_formal_charge": rchg, "delta_formal_charge": rchg-ochg,
                    "old_heavy_atoms": oha, "replacement_heavy_atoms": rha, "delta_heavy_atoms": rha-oha,
                    "old_rings": oring, "replacement_rings": rring, "delta_rings": rring-oring,
                    "old_aromatic_rings": oaro, "replacement_aromatic_rings": raro, "delta_aromatic_rings": raro-oaro,
                    "old_hetero_atoms": ohet, "replacement_hetero_atoms": rhet, "delta_hetero_atoms": rhet-ohet,
                    "old_fCSP3": round(ofsp3,4) if ofsp3 is not None else None, "replacement_fCSP3": round(rfsp3,4) if rfsp3 is not None else None,
                    "logp_direction": lp_dir, "charge_direction": chg_dir, "size_direction": size_dir, "polarity_direction": pol_dir,
                    "compute_status": "OK",
                })
                # alert delta
                if has_alerts:
                    op = 1 if pains_cat.HasMatch(old_mol) else 0; rp = 1 if pains_cat.HasMatch(repl_mol) else 0
                    ob = 1 if brenk_cat.HasMatch(old_mol) else 0; rb = 1 if brenk_cat.HasMatch(repl_mol) else 0
                    new_alert = 1 if (rp>op or rb>ob) else 0; removed = 1 if (rp<op or rb<ob) else 0
                    severity = "severe" if (rp>op or (rb-ob)>=2) else ("warning" if (rb>ob or new_alert) else "none")
                    alert_rows.append({"case_id":case_id,"candidate_id":cand_id,
                        "old_PAINS":op,"replacement_PAINS":rp,"delta_PAINS":rp-op,"new_PAINS":1 if rp>op else 0,
                        "old_Brenk":ob,"replacement_Brenk":rb,"delta_Brenk":rb-ob,"new_Brenk":1 if rb>ob else 0,
                        "new_alert_introduced":new_alert,"alert_removed":removed,"alert_severity":severity})
                # class prior
                old_elems = sorted(set(a.GetSymbol() for a in old_mol.GetAtoms()))
                repl_elems = sorted(set(a.GetSymbol() for a in repl_mol.GetAtoms()))
                old_naro = rdMolDescriptors.CalcNumAromaticRings(old_mol)
                repl_naro = rdMolDescriptors.CalcNumAromaticRings(repl_mol)
                old_n = old_mol.GetNumAtoms(); repl_n = repl_mol.GetNumAtoms()
                cls = "unknown"; cls_conf = 0.0
                # Simple rule-based class detection
                if old_naro>=1 and repl_naro>=1 and "N" not in old_elems and "N" in repl_elems:
                    cls = "aryl_to_N_heteroaryl"; cls_conf = 0.6
                elif old_naro>=1 and repl_naro>=1 and "S" in repl_elems and "S" not in old_elems:
                    cls = "phenyl_to_thiophene_like"; cls_conf = 0.4
                elif "COOH" in Chem.MolToSmiles(old_mol) and "NN" in Chem.MolToSmiles(repl_mol):
                    cls = "acid_to_tetrazole_like"; cls_conf = 0.5
                elif abs(oha-rha)<=2 and abs(omw-rmw)<=20 and abs(olp-rlp)<=1.0:
                    cls = "size_conservative_replacement"; cls_conf = 0.3
                elif abs(rha-oha)>=5:
                    cls = "size_expansion"; cls_conf = 0.4
                class_rows.append({"case_id":case_id,"candidate_id":cand_id,
                    "old_attach_smi":old["attach_smi"],"repl_attach_smi":repl["attach_smi"],
                    "detected_class":cls,"class_confidence":cls_conf})
        except Exception as e:
            delta_rows.append({"case_id":case_id,"candidate_id":cand_id,"compute_status":f"FAILED:{str(e)[:50]}"})
    else:
        delta_rows.append({"case_id":case_id,"candidate_id":cand_id,"compute_status":"BLOCKED"})

# ---- Save Parts A-D ----
join_df = pd.DataFrame(join_rows)
join_df.to_csv(OUTDIR/"a4c_v1a_candidate_join_coverage.csv", index=False)
delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(OUTDIR/"a4c_v1a_2d_delta_features.csv", index=False)
alert_df = pd.DataFrame(alert_rows) if alert_rows else pd.DataFrame()
if len(alert_df) > 0: alert_df.to_csv(OUTDIR/"a4c_v1a_alert_delta_features.csv", index=False)
class_df = pd.DataFrame(class_rows)
class_df.to_csv(OUTDIR/"a4c_v1a_bioisostere_class_prior.csv", index=False)

coverage = {"total_candidates":n_total,"candidates_with_old_2d":sum(1 for r in join_rows if r["old_fragment_2d_status"]=="OK"),
    "candidates_with_replacement_2d":n_both_2d,"candidates_with_both_2d":n_both_2d,"candidates_with_both_3d":n_both_3d,
    "candidates_missing_sdf":n_sdf_miss,"candidates_valid_2d_delta":n_both_2d,"candidates_valid_3d_shape":n_both_3d}
with open(OUTDIR/"a4c_v1a_coverage_summary.json","w") as f: json.dump(coverage,f,indent=2)
print(f"  Coverage: {n_both_2d}/{n_total} 2D delta OK, {n_both_3d}/{n_total} 3D shape OK, {n_sdf_miss} SDF miss")
print(f"  Delta rows: {len(delta_df)}, Alerts: {len(alert_df)}, Classes: {len(class_df)}")

# ================================================================
# PART E: 3D Shape Features
# ================================================================
print("\n=== Part E: 3D Shape Features ===")
shape_rows = []
for _, jr in join_df.iterrows():
    case_id, cand_id = jr["case_id"], jr["candidate_id"]
    td = three_d_map.get((case_id, cand_id))
    if td is None or not td["shape_ready"]:
        shape_rows.append({"case_id":case_id,"candidate_id":cand_id,"shape_status":"BLOCKED"})
        continue
    try:
        pm_old = json.loads(td["pm_old"]) if isinstance(td["pm_old"],str) else td["pm_old"]
        pm_repl = json.loads(td["pm_repl"]) if isinstance(td["pm_repl"],str) else td["pm_repl"]
    except:
        pm_old = pm_repl = None
    rg_o, rg_r = td["rg_old"], td["rg_repl"]
    delta_rg = rg_r - rg_o if (rg_o and rg_r) else None
    pm_dist = np.sqrt(sum((a-b)**2 for a,b in zip(pm_old,pm_repl))) if (pm_old and pm_repl and len(pm_old)==3 and len(pm_repl)==3) else None
    # Asphericity = 1 - (3*I_min/I_sum) where I = principal moments squared (actually from SVD singular values)
    if pm_old and len(pm_old)==3:
        s = sorted(pm_old); asp_o = round(1.5*(s[2]-s[0])/(s[0]+s[1]+s[2]),4) if sum(s)>0 else None
    else: asp_o = None
    if pm_repl and len(pm_repl)==3:
        s = sorted(pm_repl); asp_r = round(1.5*(s[2]-s[0])/(s[0]+s[1]+s[2]),4) if sum(s)>0 else None
    else: asp_r = None
    shape_rows.append({"case_id":case_id,"candidate_id":cand_id,
        "old_Rg":round(rg_o,4) if rg_o else None,"replacement_Rg":round(rg_r,4) if rg_r else None,
        "delta_Rg":round(delta_rg,4) if delta_rg else None,"abs_delta_Rg":round(abs(delta_rg),4) if delta_rg else None,
        "old_PM1":round(pm_old[0],4) if pm_old else None,"old_PM2":round(pm_old[1],4) if pm_old else None,"old_PM3":round(pm_old[2],4) if pm_old else None,
        "replacement_PM1":round(pm_repl[0],4) if pm_repl else None,"replacement_PM2":round(pm_repl[1],4) if pm_repl else None,"replacement_PM3":round(pm_repl[2],4) if pm_repl else None,
        "pm_distance":round(pm_dist,4) if pm_dist else None,
        "old_asphericity":asp_o,"replacement_asphericity":asp_r,"delta_asphericity":round(asp_r-asp_o,4) if (asp_o and asp_r) else None,
        "shape_status":"OK"})
shape_df = pd.DataFrame(shape_rows)
shape_df.to_csv(OUTDIR/"a4c_v1a_3d_shape_features.csv", index=False)
print(f"  Shape features: {len(shape_df)} rows, OK: {(shape_df.shape_status=='OK').sum()}")

# ================================================================
# PART F: Pharmacophore Features
# ================================================================
print("\n=== Part F: Pharmacophore Features ===")
pharm_rows = []
for _, jr in join_df.iterrows():
    case_id, cand_id = jr["case_id"], jr["candidate_id"]
    old = old_map.get(case_id)
    repl = repl_map.get((case_id, cand_id))
    if not (old and repl and repl["status"]=="EXTRACTED"):
        pharm_rows.append({"case_id":case_id,"candidate_id":cand_id,"pharm_status":"BLOCKED"})
        continue
    try:
        om = Chem.MolFromSmiles(old["attach_smi"]); rm = Chem.MolFromSmiles(repl["attach_smi"])
        if not (om and rm):
            pharm_rows.append({"case_id":case_id,"candidate_id":cand_id,"pharm_status":"PARSE_FAIL"})
            continue
        oaro = rdMolDescriptors.CalcNumAromaticRings(om); raro = rdMolDescriptors.CalcNumAromaticRings(rm)
        ohbd = Lipinski.NumHDonors(om); rhbd = Lipinski.NumHDonors(rm)
        ohba = Lipinski.NumHAcceptors(om); rhba = Lipinski.NumHAcceptors(rm)
        # charge proxy: formal charge > 0 or < 0
        ochg = Chem.GetFormalCharge(om); rchg = Chem.GetFormalCharge(rm)
        ohph = rdMolDescriptors.CalcNumAliphaticCarbocycles(om) + rdMolDescriptors.CalcNumAliphaticHeterocycles(om)
        rhph = rdMolDescriptors.CalcNumAliphaticCarbocycles(rm) + rdMolDescriptors.CalcNumAliphaticHeterocycles(rm)
        td = three_d_map.get((case_id, cand_id))
        has_old_3d = td and td["shape_ready"]; has_repl_3d = td is not None
        spatial = "READY" if (has_old_3d and has_repl_3d) else ("PARTIAL" if has_repl_3d else "BLOCKED")
        pharm_rows.append({"case_id":case_id,"candidate_id":cand_id,
            "old_aromatic_count":oaro,"replacement_aromatic_count":raro,"delta_aromatic_count":raro-oaro,
            "old_HBD_count":ohbd,"replacement_HBD_count":rhbd,"delta_HBD_count":rhbd-ohbd,
            "old_HBA_count":ohba,"replacement_HBA_count":rhba,"delta_HBA_count":rhba-ohba,
            "old_charge_sign":1 if ochg>0 else (-1 if ochg<0 else 0),
            "replacement_charge_sign":1 if rchg>0 else (-1 if rchg<0 else 0),"delta_charge_sign":(1 if rchg>0 else (-1 if rchg<0 else 0))-(1 if ochg>0 else (-1 if ochg<0 else 0)),
            "old_hydrophobe_proxy":ohph,"replacement_hydrophobe_proxy":rhph,"delta_hydrophobe_proxy":rhph-ohph,
            "pharmacophore_spatial_status":spatial,"pharm_status":"OK"})
    except Exception as e:
        pharm_rows.append({"case_id":case_id,"candidate_id":cand_id,"pharm_status":f"FAILED:{str(e)[:50]}"})
pharm_df = pd.DataFrame(pharm_rows)
pharm_df.to_csv(OUTDIR/"a4c_v1a_pharmacophore_features.csv", index=False)
print(f"  Pharmacophore: {len(pharm_df)} rows, OK: {(pharm_df.pharm_status=='OK').sum()}")

# ================================================================
# PART G: Feature Entropy and Level Audit
# ================================================================
print("\n=== Part G: Feature Entropy Audit ===")

# Define feature taxonomy
feature_specs = [
    # (name, group, default_level)
    ("delta_MW", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("abs_delta_MW", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_LogP", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("abs_delta_LogP", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_TPSA", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("abs_delta_TPSA", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_HBD", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_HBA", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_RotB", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_formal_charge", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_heavy_atoms", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_rings", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_aromatic_rings", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("delta_hetero_atoms", "2D_delta", "CASE_LEVEL_REPLACEMENT"),
    ("new_PAINS", "alert", "CASE_LEVEL_REPLACEMENT"),
    ("new_Brenk", "alert", "CASE_LEVEL_REPLACEMENT"),
    ("new_alert_introduced", "alert", "CASE_LEVEL_REPLACEMENT"),
    ("delta_Rg", "3D_shape", "CANDIDATE_LEVEL_CONFORMER"),
    ("abs_delta_Rg", "3D_shape", "CANDIDATE_LEVEL_CONFORMER"),
    ("pm_distance", "3D_shape", "CANDIDATE_LEVEL_CONFORMER"),
    ("delta_asphericity", "3D_shape", "CANDIDATE_LEVEL_CONFORMER"),
    ("delta_aromatic_count", "pharm_count", "CASE_LEVEL_REPLACEMENT"),
    ("delta_HBD_count", "pharm_count", "CASE_LEVEL_REPLACEMENT"),
    ("delta_HBA_count", "pharm_count", "CASE_LEVEL_REPLACEMENT"),
    ("delta_charge_sign", "pharm_count", "CASE_LEVEL_REPLACEMENT"),
]

entropy_rows = []
for fname, fgroup, flevel in feature_specs:
    # Find data source
    vals = None
    case_col = None
    if fgroup == "2D_delta" and fname in delta_df.columns:
        df_src = delta_df[delta_df.compute_status == "OK"] if "compute_status" in delta_df.columns else delta_df
        vals = pd.to_numeric(df_src[fname], errors='coerce').dropna()
        case_col = df_src["case_id"]
    elif fgroup == "alert" and fname in alert_df.columns:
        vals = pd.to_numeric(alert_df[fname], errors='coerce').dropna()
        case_col = alert_df["case_id"]
    elif fgroup == "3D_shape" and fname in shape_df.columns:
        ok = shape_df[shape_df.shape_status == "OK"] if "shape_status" in shape_df.columns else shape_df
        vals = pd.to_numeric(ok[fname], errors='coerce').dropna()
        case_col = ok["case_id"]
    elif fgroup == "pharm_count" and fname in pharm_df.columns:
        ok = pharm_df[pharm_df.pharm_status == "OK"] if "pharm_status" in pharm_df.columns else pharm_df
        vals = pd.to_numeric(ok[fname], errors='coerce').dropna()
        case_col = ok["case_id"]

    if vals is None or len(vals) == 0:
        entropy_rows.append({"feature_name":fname,"feature_group":fgroup,"feature_level":flevel,
            "availability_rate":0,"missing_rate":1,"within_case_std_mean":None,"across_case_std":None,"entropy_status":"BLOCKED"})
        continue

    avail = len(vals)/n_total if n_total>0 else 0
    miss = 1-avail
    if case_col is not None and len(vals) > 0:
        tmp = pd.DataFrame({"case_id":case_col.iloc[:len(vals)],"val":vals.values})
        wc_std = tmp.groupby("case_id")["val"].std().dropna()
        wc_mean = float(wc_std.mean()) if len(wc_std)>0 else 0
    else:
        wc_mean = 0
    ac_std = float(vals.std())
    nunique = int(vals.nunique())

    if flevel == "CASE_LEVEL_REPLACEMENT":
        ent_status = "LOW_WITHIN_CASE_EXPECTED"  # Expected for case-level
    elif wc_mean < 0.001 and len(vals) > 100:
        ent_status = "LOW_ENTROPY_PROBLEM"
    elif wc_mean > 0.01:
        ent_status = "HIGH_ENTROPY"
    else:
        ent_status = "LOW_ENTROPY_PROBLEM"

    entropy_rows.append({"feature_name":fname,"feature_group":fgroup,"feature_level":flevel,
        "availability_rate":round(avail,4),"missing_rate":round(miss,4),
        "num_unique_values":nunique,"within_case_std_mean":round(wc_mean,6),
        "across_case_std":round(ac_std,4),"entropy_status":ent_status})

entropy_df = pd.DataFrame(entropy_rows)
entropy_df.to_csv(OUTDIR/"a4c_v1a_feature_entropy_audit.csv", index=False)
print(f"  Entropy: {len(entropy_df)} features audited")
print(f"  CASE_LEVEL: {(entropy_df.feature_level=='CASE_LEVEL_REPLACEMENT').sum()}, CANDIDATE_LEVEL: {(entropy_df.feature_level=='CANDIDATE_LEVEL_CONFORMER').sum()}")

# ================================================================
# PART H: Proxy Discriminability
# ================================================================
print("\n=== Part H: Proxy Discriminability ===")
proxy_rows = []
for _, erow in entropy_df.iterrows():
    fname, fgroup, flevel = erow.feature_name, erow.feature_group, erow.feature_level
    if erow.entropy_status == "BLOCKED":
        proxy_rows.append({"feature_name":fname,"proxy_target":"N/A","AUC":None,"spearman_r":None,"interpretation":"BLOCKED"})
        continue

    # Get values joined with case data
    proxy_rows.append({"feature_name":fname,"proxy_target":"oracle_RMSD_proxy",
        "AUC":None,"spearman_r":None,"interpretation":"geometry proxy only, not bioisostere label"})
proxy_df = pd.DataFrame(proxy_rows)
proxy_df.to_csv(OUTDIR/"a4c_v1a_feature_proxy_discriminability.csv", index=False)

# ================================================================
# PART I: Hard-Filter Threshold Exploration
# ================================================================
print("\n=== Part I: Hard-Filter Threshold Exploration ===")
thresh_rows = []
delta_ok = delta_df[delta_df.compute_status == "OK"] if "compute_status" in delta_df.columns else delta_df

for fname, threshold, desc in [
    ("abs_delta_MW", 100, "|ΔMW| > 100 Da"),
    ("abs_delta_MW", 50, "|ΔMW| > 50 Da"),
    ("abs_delta_LogP", 3, "|ΔLogP| > 3"),
    ("abs_delta_LogP", 2, "|ΔLogP| > 2"),
    ("abs_delta_TPSA", 80, "|ΔTPSA| > 80"),
    ("abs_delta_TPSA", 40, "|ΔTPSA| > 40"),
    ("delta_formal_charge", 2, "|Δcharge| >= 2"),
]:
    if fname in delta_ok.columns:
        vals = pd.to_numeric(delta_ok[fname], errors='coerce')
        if "charge" in fname:
            flagged = vals.abs() >= threshold
        else:
            flagged = vals > threshold
        n_flag = flagged.sum()
        thresh_rows.append({"filter_name":desc,"threshold":threshold,"num_flagged":int(n_flag),
            "flagged_fraction":round(float(n_flag/len(vals)),4) if len(vals)>0 else 0,
            "recommended_use":"V1B_WARNING_ONLY" if n_flag < 50 else "V1B_HARD_FILTER_CANDIDATE"})

if len(alert_df) > 0:
    for fname in ["new_PAINS","new_Brenk","new_alert_introduced"]:
        if fname in alert_df.columns:
            vals = pd.to_numeric(alert_df[fname], errors='coerce')
            n_flag = (vals > 0).sum()
            thresh_rows.append({"filter_name":fname,"threshold":1,"num_flagged":int(n_flag),
                "flagged_fraction":round(float(n_flag/len(vals)),4) if len(vals)>0 else 0,
                "recommended_use":"V1B_HARD_FILTER_CANDIDATE"})

thresh_df = pd.DataFrame(thresh_rows)
thresh_df.to_csv(OUTDIR/"a4c_v1a_hard_filter_threshold_exploration.csv", index=False)
print(f"  Thresholds: {len(thresh_df)} filters explored")

# ================================================================
# PART J: Feature Readiness Decision
# ================================================================
print("\n=== Part J: Feature Readiness Decision ===")
dec_rows = []
for _, erow in entropy_df.iterrows():
    fname, fgroup, flevel, estatus = erow.feature_name, erow.feature_group, erow.feature_level, erow.entropy_status
    if estatus == "BLOCKED":
        ready, use, reason = "BLOCKED", "V1B_DO_NOT_USE", "feature unavailable"
    elif flevel == "CASE_LEVEL_REPLACEMENT":
        if fgroup == "alert" and "new_alert" in fname:
            ready, use, reason = "AVAILABLE_NOW", "V1B_HARD_FILTER", "alert delta is hard filter candidate"
        elif fgroup == "alert":
            ready, use, reason = "AVAILABLE_CASE_LEVEL_ONLY", "V1B_WARNING_RULE", "alert information, case-level"
        elif "abs_delta" in fname and fgroup == "2D_delta":
            ready, use, reason = "AVAILABLE_CASE_LEVEL_ONLY", "V1B_HARD_FILTER", "magnitude of property shift, case-level hard filter candidate"
        else:
            ready, use, reason = "AVAILABLE_CASE_LEVEL_ONLY", "V1B_REVIEW_ANNOTATION", "directional delta, case-level annotation"
    elif flevel == "CANDIDATE_LEVEL_CONFORMER":
        ready, use, reason = "AVAILABLE_CANDIDATE_LEVEL", "V1B_REVIEW_ANNOTATION", "conformer-level shape feature, useful for within-case comparison"
    else:
        ready, use, reason = "AVAILABLE_NOW", "V1B_REVIEW_ANNOTATION", ""
    dec_rows.append({"feature_name":fname,"feature_group":fgroup,"ready_status":ready,"recommended_use":use,"reason":reason})
dec_df = pd.DataFrame(dec_rows)
dec_df.to_csv(OUTDIR/"a4c_v1a_feature_readiness_decision.csv", index=False)
print(f"  Decisions: {len(dec_df)} features")

# ---- Summary ----
print(f"\n=== Summary ===")
print(f"2D delta features: {n_both_2d}/{n_total} candidates")
print(f"3D shape features: {n_both_3d}/{n_total} candidates")
print(f"CASE_LEVEL features: {(entropy_df.feature_level=='CASE_LEVEL_REPLACEMENT').sum()}")
print(f"CANDIDATE_LEVEL features: {(entropy_df.feature_level=='CANDIDATE_LEVEL_CONFORMER').sum()}")
print(f"Hard filter candidates: {(dec_df.recommended_use=='V1B_HARD_FILTER').sum()}")
print(f"Warning/annotation: {(dec_df.recommended_use.str.contains('WARNING|ANNOTATION',na=False)).sum()}")
print(f"\nAll outputs in {OUTDIR}")
