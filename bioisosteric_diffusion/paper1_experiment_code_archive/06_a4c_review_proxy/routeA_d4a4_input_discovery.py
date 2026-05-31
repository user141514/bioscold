#!/usr/bin/env python
"""D4A4 Part A: Input Discovery - scan upstream outputs and report data structures."""

import json
import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"E:\zuhui\bioisosteric_diffusion")
OUTPUT_DIR = PROJECT_ROOT / "plan_results" / "routeA_chembl37k_d4a4_dual_mode_integration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def discover_inputs():
    """Scan upstream directories and report file inventory with structure info."""
    discoveries = []

    # --- D4A2D2 Ensemble outputs ---
    d4a2d2_dir = PROJECT_ROOT / "plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble"

    # Check the large JSONL prediction file
    pred_file = d4a2d2_dir / "d4a2d2_standardized_predictions_test.jsonl"
    if pred_file.exists():
        with open(pred_file, encoding='utf-8') as f:
            line1 = f.readline()
            d = json.loads(line1)

        query_keys = [k for k in d.keys() if k != 'candidates']
        candidate_keys = list(d['candidates'][0].keys()) if d.get('candidates') else []

        discoveries.append({
            'file_path': str(pred_file),
            'role': 'Borda_predictions_test',
            'method': 'Borda(DE+HGB)',
            'group': 'G1_all_Borda_topK',
            'has_query_id': 'query_id' in d or 'query_id' in str(query_keys),
            'has_candidate_id': 'candidate_id' in str(candidate_keys) or 'candidate_norm' in str(candidate_keys),
            'has_replacement_smiles': 'candidate_norm' in str(candidate_keys) or 'replacement_smiles' in str(candidate_keys),
            'has_rank': 'rank_Borda' in str(candidate_keys) or 'rank' in str(candidate_keys),
            'has_score': 'score_Borda' in str(candidate_keys) or 'score' in str(candidate_keys),
            'has_a4c_status': 'a4c_label' in str(candidate_keys),
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': f'query_keys={query_keys}, candidate_keys={candidate_keys}',
        })

    # Hit overlap file
    hit_file = d4a2d2_dir / "d4a2d2_query_hits_test.csv"
    if hit_file.exists():
        df = pd.read_csv(hit_file)
        discoveries.append({
            'file_path': str(hit_file),
            'role': 'query_level_hits',
            'method': 'Borda+HGB+DE',
            'group': 'all',
            'has_query_id': 'query_id' in df.columns or 'qid' in df.columns,
            'has_candidate_id': False,
            'has_replacement_smiles': False,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': f'columns={list(df.columns)}, rows={len(df)}',
        })

    # Bootstrap comparisons
    boot_file = d4a2d2_dir / "d4a2d2_bootstrap_comparisons.csv"
    if boot_file.exists():
        df = pd.read_csv(boot_file)
        discoveries.append({
            'file_path': str(boot_file),
            'role': 'bootstrap_comparisons',
            'method': 'Borda_vs_HGB_vs_DE',
            'group': 'all',
            'has_query_id': False,
            'has_candidate_id': False,
            'has_replacement_smiles': False,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': f'columns={list(df.columns)}',
        })

    # Error analysis
    err_file = d4a2d2_dir / "d4a2d2_ensemble_error_analysis.csv"
    if err_file.exists():
        df = pd.read_csv(err_file, nrows=1)
        discoveries.append({
            'file_path': str(err_file),
            'role': 'error_analysis',
            'method': 'Borda',
            'group': 'all',
            'has_query_id': 'query_id' in df.columns or 'qid' in df.columns,
            'has_candidate_id': True,
            'has_replacement_smiles': True,
            'has_rank': True,
            'has_score': True,
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': f'columns={list(df.columns)}',
        })

    # --- D4A3T Exploration Calibration ---
    d4a3t_dir = PROJECT_ROOT / "plan_results/routeA_chembl37k_d4a3t_exploration_calibration"

    # G1 candidate labels (repaired A4C coverage)
    g1_labels = d4a3t_dir / "d4a3t_candidate_labels_g1.csv"
    if g1_labels.exists():
        df = pd.read_csv(g1_labels)
        discoveries.append({
            'file_path': str(g1_labels),
            'role': 'G1_candidate_labels_repaired',
            'method': 'D4A3T_A4C',
            'group': 'G1',
            'has_query_id': 'qid' in df.columns,
            'has_candidate_id': 'candidate_norm' in df.columns,
            'has_replacement_smiles': 'candidate_norm' in df.columns,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': 'a4c_label' in df.columns,
            'has_alert_rate': False,
            'has_group_label': 'gap_type' in df.columns,
            'status': 'OK',
            'notes': f'columns={list(df.columns)}, rows={len(df)}, a4c_labels={df["a4c_label"].value_counts().to_dict() if "a4c_label" in df.columns else "N/A"}',
        })

    # Risk decomposition
    risk_file = d4a3t_dir / "d4a3t_risk_decomposition.csv"
    if risk_file.exists():
        df = pd.read_csv(risk_file)
        discoveries.append({
            'file_path': str(risk_file),
            'role': 'risk_decomposition',
            'method': 'D4A3T',
            'group': 'G2+G3+G4',
            'has_query_id': False,
            'has_candidate_id': False,
            'has_replacement_smiles': False,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': False,
            'has_alert_rate': True,
            'has_group_label': True,
            'status': 'OK',
            'notes': f'columns={list(df.columns)}',
        })

    # Alert rate by group
    alert_file = d4a3t_dir / "d4a3t_alert_rate_by_group.csv"
    if alert_file.exists():
        df = pd.read_csv(alert_file)
        discoveries.append({
            'file_path': str(alert_file),
            'role': 'alert_rate_summary',
            'method': 'D4A3T',
            'group': 'G1+G4',
            'has_query_id': False,
            'has_candidate_id': False,
            'has_replacement_smiles': False,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': False,
            'has_alert_rate': True,
            'has_group_label': True,
            'status': 'OK',
            'notes': str(df.to_dict('records')),
        })

    # Pre-registered criteria
    pr_file = d4a3t_dir / "d4a3t_pre_registered_criteria.json"
    if pr_file.exists():
        with open(pr_file, encoding='utf-8') as f:
            pr = json.load(f)
        discoveries.append({
            'file_path': str(pr_file),
            'role': 'pre_registered_criteria',
            'method': 'D4A3T',
            'group': 'all',
            'has_query_id': False,
            'has_candidate_id': False,
            'has_replacement_smiles': False,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': json.dumps(pr),
        })

    # Exploration mode policy config
    pol_file = d4a3t_dir / "d4a3t_exploration_mode_policy_config.json"
    if pol_file.exists():
        with open(pol_file, encoding='utf-8') as f:
            pol = json.load(f)
        discoveries.append({
            'file_path': str(pol_file),
            'role': 'policy_config',
            'method': 'D4A3T',
            'group': 'all',
            'has_query_id': False,
            'has_candidate_id': False,
            'has_replacement_smiles': False,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': json.dumps(pol),
        })

    # --- D4A3S A4C Coverage ---
    d4a3s_dir = PROJECT_ROOT / "plan_results/routeA_chembl37k_d4a3s_a4c_coverage_expansion"

    for g in ['G0', 'G1', 'G2', 'G3', 'G4']:
        gfile = d4a3s_dir / f"d4a3s_{g}_candidates.csv"
        if gfile.exists():
            df = pd.read_csv(gfile, nrows=1)
            discoveries.append({
                'file_path': str(gfile),
                'role': f'{g}_candidates_pre_repair',
                'method': 'D4A3S',
                'group': g,
                'has_query_id': 'qid' in df.columns,
                'has_candidate_id': 'candidate_norm' in df.columns,
                'has_replacement_smiles': 'candidate_norm' in df.columns,
                'has_rank': False,
                'has_score': False,
                'has_a4c_status': 'a4c_label' in df.columns,
                'has_alert_rate': False,
                'has_group_label': True,
                'status': 'OK_PRE_REPAIR',
                'notes': f'columns={list(df.columns)}',
            })

    # --- D4A2D1R Dual Encoder Robustness ---
    d4a2d1r_dir = PROJECT_ROOT / "plan_results/routeA_chembl37k_d4a2d1r_dual_encoder_robustness"

    de_pred_file = d4a2d1r_dir / "d4a2d1r_standardized_predictions.jsonl"
    if de_pred_file.exists():
        with open(de_pred_file, encoding='utf-8') as f:
            line1 = f.readline()
            d = json.loads(line1)
        cand_keys = list(d['candidates'][0].keys()) if d.get('candidates') else []
        discoveries.append({
            'file_path': str(de_pred_file),
            'role': 'DE_predictions_test',
            'method': 'DualEncoder',
            'group': 'all_DE',
            'has_query_id': True,
            'has_candidate_id': True,
            'has_replacement_smiles': 'candidate_norm' in str(cand_keys),
            'has_rank': 'rank' in str(cand_keys),
            'has_score': 'score' in str(cand_keys),
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': f'candidate_keys={cand_keys}',
        })

    # D4A2D1R hit overlap
    hit_file2 = d4a2d1r_dir / "d4a2d1r_hit_overlap.csv"
    if hit_file2.exists():
        df = pd.read_csv(hit_file2)
        discoveries.append({
            'file_path': str(hit_file2),
            'role': 'DE_HGB_hit_overlap',
            'method': 'DE+HGB',
            'group': 'all',
            'has_query_id': False,
            'has_candidate_id': False,
            'has_replacement_smiles': False,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': f'columns={list(df.columns)}',
        })

    # --- HGB predictions ---
    hgb_dir = PROJECT_ROOT / "plan_results/routeA_chembl37k_d4a1_learned_ranker"
    hgb_file = hgb_dir / "d4a1_test_predictions.jsonl"
    if hgb_file.exists():
        with open(hgb_file, encoding='utf-8') as f:
            line1 = f.readline()
            d = json.loads(line1)
        cand_keys = list(d['candidates'][0].keys()) if d.get('candidates') else []
        discoveries.append({
            'file_path': str(hgb_file),
            'role': 'HGB_predictions_test',
            'method': 'HGB',
            'group': 'all_HGB',
            'has_query_id': True,
            'has_candidate_id': True,
            'has_replacement_smiles': 'candidate_norm' in str(cand_keys),
            'has_rank': 'rank' in str(cand_keys),
            'has_score': 'score' in str(cand_keys),
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': False,
            'status': 'OK',
            'notes': f'candidate_keys={cand_keys}',
        })

    # --- D4A0 split manifest ---
    d4a0_dir = PROJECT_ROOT / "plan_results/routeA_chembl37k_d0d3_engineering_safe/07_d4a0_matrix_freeze"
    manifest_file = d4a0_dir / "d4a0_query_split_manifest.jsonl"
    if manifest_file.exists():
        with open(manifest_file, encoding='utf-8') as f:
            line1 = f.readline()
            d = json.loads(line1)
        discoveries.append({
            'file_path': str(manifest_file),
            'role': 'query_split_manifest',
            'method': 'D4A0',
            'group': 'all',
            'has_query_id': 'query_id' in d,
            'has_candidate_id': False,
            'has_replacement_smiles': False,
            'has_rank': False,
            'has_score': False,
            'has_a4c_status': False,
            'has_alert_rate': False,
            'has_group_label': 'split' in str(d.keys()),
            'status': 'OK',
            'notes': f'keys={list(d.keys())}',
        })

    return discoveries

def main():
    discoveries = discover_inputs()
    df = pd.DataFrame(discoveries)
    output_path = OUTPUT_DIR / "d4a4_input_discovery.csv"
    df.to_csv(output_path, index=False)
    print(f"Discovered {len(df)} input files")
    print(f"Saved to {output_path}")

    # Key checks
    has_borda = any('Borda_predictions' in str(r) for r in df['role'])
    has_hgb = any('HGB_predictions' in str(r) for r in df['role'])
    has_a4c = any('G1_candidate_labels_repaired' in str(r) for r in df['role'])
    has_groups = any('G2_candidates' in str(r) for r in df['role'])

    print(f"\nRequired inputs:")
    print(f"  Borda top-K predictions: {'FOUND' if has_borda else 'MISSING'}")
    print(f"  HGB top-K predictions: {'FOUND' if has_hgb else 'MISSING'}")
    print(f"  A4C status (D4A3T): {'FOUND' if has_a4c else 'MISSING'}")
    print(f"  G2/G3/G4 group labels: {'FOUND' if has_groups else 'MISSING'}")

    if not has_borda:
        print("\nERROR: MISSING_PROPOSAL_INPUTS - Borda predictions not found")
    if not has_a4c:
        print("\nERROR: MISSING_A4C_STATUS - A4C status not found")

if __name__ == "__main__":
    main()
