#!/usr/bin/env python
"""D4P1-Phase0 Part A: Input Discovery — scan all Route-A stage directories for metric-bearing files."""
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(r"E:\zuhui\bioisosteric_diffusion")
OUTPUT_DIR = PROJECT_ROOT / "plan_results" / "routeA_chembl37k_d4p1_phase0_metric_lock"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target directories
TARGET_DIRS = [
    "plan_results/routeA_chembl37k_d4a0_matrix_freeze",
    "plan_results/routeA_chembl37k_d4a1_learned_ranker",
    "plan_results/routeA_chembl37k_d4a1r_ranker_audit",
    "plan_results/routeA_chembl37k_d4a2d1_dual_encoder_smoke",
    "plan_results/routeA_chembl37k_d4a2d1_full_gate",
    "plan_results/routeA_chembl37k_d4a2d1r_dual_encoder_robustness",
    "plan_results/routeA_chembl37k_d4a2d2_de_hgb_ensemble",
    "plan_results/routeA_chembl37k_d4a3r_a4c_borda_review",
    "plan_results/routeA_chembl37k_d4a3s_a4c_coverage_expansion",
    "plan_results/routeA_chembl37k_d4a3t_exploration_calibration",
    "plan_results/routeA_chembl37k_d4a4_dual_mode_integration",
]

# Keyword patterns for file names (case-insensitive)
KEYWORDS = [
    "metrics", "summary", "verdict", "bootstrap", "comparison",
    "borda", "hgb", "dual", "de", "attach", "frequency",
    "oracle", "fusion", "conservative", "exploration",
    "top10", "top1", "top5", "top20", "top50", "mrr", "hit", "query",
    "test_metrics", "val_metrics", "baseline", "gain", "gate",
    "rank", "score", "canonical", "policy", "risk"
]

STAGES = {
    "d4a0": "D4A0", "d4a1_learned": "D4A1", "d4a1r": "D4A1R",
    "d4a2d1_smoke": "D4A2D1-SMOKE", "d4a2d1_full": "D4A2D1-FULL",
    "d4a2d1r": "D4A2D1R", "d4a2d2": "D4A2D2",
    "d4a3r": "D4A3R", "d4a3s": "D4A3S", "d4a3t": "D4A3T",
    "d4a4": "D4A4"
}

def infer_stage(dirname):
    for key, stage in STAGES.items():
        if key in dirname:
            return stage
    return "UNKNOWN"

def file_has_fields(filepath):
    """Quick check if file has method/metric/query columns."""
    try:
        ext = filepath.suffix.lower()
        if ext == '.csv':
            df = pd.read_csv(filepath, nrows=0)
            cols = [c.lower() for c in df.columns]
            return {
                'has_method_column': any('method' in c for c in cols),
                'has_metric_column': any(k in c for c in cols for k in ['metric', 'value', 'top1', 'top5', 'top10', 'top20', 'top50', 'mrr', 'hit', 'score']),
                'has_query_count': any('query' in c or 'n_' in c or 'num_' in c for c in cols),
                'has_top1': any('top1' in c or 'top_1' in c for c in cols),
                'has_top5': any('top5' in c or 'top_5' in c for c in cols),
                'has_top10': any('top10' in c or 'top_10' in c for c in cols),
                'has_top20': any('top20' in c or 'top_20' in c for c in cols),
                'has_top50': any('top50' in c or 'top_50' in c for c in cols),
                'has_mrr': any('mrr' in c for c in cols),
                'has_ci': any(k in c for k in ['ci_', 'ci95', 'bootstrap', 'ci_low', 'ci_high'] for c in cols),
            }
        elif ext == '.json':
            with open(filepath, encoding='utf-8') as f:
                content = f.read(10000)
            d = json.loads(content) if content.strip().startswith('{') else {}
            keys = [k.lower() for k in d.keys()]
            vals = str(d).lower()
            return {
                'has_method_column': 'method' in keys or 'method' in vals,
                'has_metric_column': any(k in vals for k in ['top1', 'top5', 'top10', 'top20', 'top50', 'mrr', 'hit_rate', 'metric']),
                'has_query_count': 'query' in vals or 'n_queries' in vals,
                'has_top1': 'top1' in vals or 'top_1' in vals,
                'has_top5': 'top5' in vals or 'top_5' in vals,
                'has_top10': 'top10' in vals or 'top_10' in vals,
                'has_top20': 'top20' in vals or 'top_20' in vals,
                'has_top50': 'top50' in vals or 'top_50' in vals,
                'has_mrr': 'mrr' in vals,
                'has_ci': any(k in vals for k in ['ci_', 'ci95', 'bootstrap']),
            }
        elif ext == '.md':
            with open(filepath, encoding='utf-8') as f:
                content = f.read(5000).lower()
            return {
                'has_method_column': False,
                'has_metric_column': any(k in content for k in ['top1', 'top5', 'top10', 'top20', 'top50', 'mrr', 'hit_rate']),
                'has_query_count': 'query' in content or 'n=' in content,
                'has_top1': 'top1' in content,
                'has_top5': 'top5' in content,
                'has_top10': 'top10' in content,
                'has_top20': 'top20' in content,
                'has_top50': 'top50' in content,
                'has_mrr': 'mrr' in content,
                'has_ci': 'ci' in content or 'bootstrap' in content,
            }
    except Exception as e:
        return {k: False for k in ['has_method_column','has_metric_column','has_query_count','has_top1','has_top5','has_top10','has_top20','has_top50','has_mrr','has_ci']}
    return {k: False for k in ['has_method_column','has_metric_column','has_query_count','has_top1','has_top5','has_top10','has_top20','has_top50','has_mrr','has_ci']}

def main():
    discoveries = []
    for tdir_rel in TARGET_DIRS:
        tdir = PROJECT_ROOT / tdir_rel
        if not tdir.exists():
            discoveries.append({
                'file_path': str(tdir),
                'stage': infer_stage(tdir_rel),
                'candidate_role': 'DIRECTORY',
                'file_type': 'MISSING',
                'size_bytes': 0,
                'has_method_column': False,
                'has_metric_column': False,
                'has_query_count': False,
                'has_top1': False, 'has_top5': False, 'has_top10': False,
                'has_top20': False, 'has_top50': False, 'has_mrr': False, 'has_ci': False,
                'status': 'MISSING',
                'notes': 'Directory not found'
            })
            continue

        for f in sorted(tdir.iterdir()):
            if f.is_dir():
                continue
            fname_lower = f.name.lower()
            # Check keyword match
            matched_keywords = [kw for kw in KEYWORDS if kw in fname_lower]
            if not matched_keywords:
                continue

            ext = f.suffix.lower()
            if ext not in ('.csv', '.json', '.jsonl', '.md'):
                continue

            stage = infer_stage(tdir_rel)
            stat = f.stat()
            fields = file_has_fields(f)

            # Infer candidate role
            role = "other"
            if 'verdict' in fname_lower:
                role = 'verdict'
            elif 'bootstrap' in fname_lower:
                role = 'bootstrap'
            elif 'comparison' in fname_lower:
                role = 'comparison'
            elif 'metrics' in fname_lower:
                role = 'metrics'
            elif 'summary' in fname_lower:
                role = 'summary'
            elif 'baseline' in fname_lower:
                role = 'baseline'
            elif 'gate' in fname_lower:
                role = 'gate'
            elif 'policy' in fname_lower:
                role = 'policy_config'
            elif 'error' in fname_lower:
                role = 'error_analysis'
            elif 'prediction' in fname_lower:
                role = 'predictions'

            discoveries.append({
                'file_path': str(f),
                'stage': stage,
                'candidate_role': role,
                'file_type': ext[1:],
                'size_bytes': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                **fields,
                'status': 'FOUND',
                'notes': f'matched_keywords={matched_keywords[:6]}'
            })

    df = pd.DataFrame(discoveries)
    output_path = OUTPUT_DIR / "d4p1_phase0_input_discovery.csv"
    df.to_csv(output_path, index=False)
    print(f"Discovered {len(df)} metric-related files across {len(TARGET_DIRS)} directories")
    print(f"Saved to {output_path}")

    # Summary
    by_stage = df.groupby('stage').size()
    print(f"\nFiles per stage:")
    for stage, count in by_stage.items():
        print(f"  {stage}: {count}")

    by_role = df.groupby('candidate_role').size()
    print(f"\nFiles per role:")
    for role, count in by_role.items():
        print(f"  {role}: {count}")

    # Key files for canonical table
    key_files = df[df['has_metric_column'] | (df['candidate_role'].isin(['verdict', 'bootstrap', 'comparison', 'metrics']))]
    print(f"\n{len(key_files)} key metric-bearing files identified")

if __name__ == "__main__":
    main()
