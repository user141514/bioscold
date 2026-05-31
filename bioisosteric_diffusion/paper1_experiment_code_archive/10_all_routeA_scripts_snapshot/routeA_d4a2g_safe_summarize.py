#!/usr/bin/env python3
"""
=============================================================================
 D4A2G-SAFE Transform Vector Gate — Summarize + Baseline Compare + Verdict
=============================================================================

Combines:
  - Part 5: Baseline comparison across all methods on the same query subset
  - Part 6: Failure diagnosis (classify gate failures)
  - Final verdict with skeptical review

Generates:
  - d4a2g_safe_baseline_comparison.csv
  - d4a2g_safe_failure_diagnosis.csv
  - D4A2G_SAFE_TRANSFORM_VECTOR_GATE_VERDICT.md
  - MAIN_DECISION_LOG.md

Answers 12 questions (Q1-Q12) and produces verdicts A through I.

Usage:
  python core/scripts/routeA_d4a2g_safe_summarize.py

Engineering: CPU-only, reads existing outputs, no heavy computation.
=============================================================================
"""

from __future__ import annotations

import json, logging, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("d4a2g_summarize")

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

REPO_DIR = Path(__file__).resolve().parent.parent.parent
D4A0_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze"
D4A1_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d4a1_learned_ranker"
D4A1R_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d4a1r_ranker_audit"
OUT_DIR = REPO_DIR / "plan_results" / "routeA_chembl37k_d4a2g_safe_transform_vector_gates"


# ===================================================================
# Helpers
# ===================================================================

def load_json(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(str(c) for c in fieldnames) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(c, "")) for c in fieldnames) + "\n")


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dicts."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        vals = line.split(",")
        rows.append({headers[i]: vals[i].strip() if i < len(vals) else "" for i in range(len(headers))})
    return rows


def load_gate_summary(gate_name: str) -> Optional[Dict]:
    return load_json(OUT_DIR / f"d4a2g_safe_gate{gate_name}_summary.json")


def load_resource_preflight() -> Optional[Dict]:
    return load_json(OUT_DIR / "d4a2g_safe_resource_preflight.json")


def load_delta_dataset_summary() -> Optional[Dict]:
    return load_json(OUT_DIR / "d4a2g_safe_delta_dataset_summary.json")


def load_embedding_config() -> Optional[Dict]:
    return load_json(OUT_DIR / "d4a2g_safe_embedding_config.json")


def load_split_summary() -> Optional[Dict]:
    return load_json(OUT_DIR / "d4a2g_safe_split_summary.json")


def write_complete_marker(name: str) -> None:
    (OUT_DIR / f"{name}_complete.md").write_text(
        f"# {name}\nCompleted: {TIMESTAMP}\n", encoding="utf-8"
    )


# ===================================================================
# Part 5: Baseline Comparison
# ===================================================================

def gather_all_metrics() -> List[Dict[str, Any]]:
    """Gather metrics from Gate C outputs + D4A0 + D4A1 baselines."""
    all_metrics: List[Dict[str, Any]] = []

    # From Gate C ablation CSV
    gate_c_csv = OUT_DIR / "d4a2g_safe_gateC_zero_delta_ablation.csv"
    c_rows = read_csv_dicts(gate_c_csv)
    for row in c_rows:
        all_metrics.append({
            "method": row.get("method", "unknown"),
            "n_queries": int(float(row.get("n_queries", 0))),
            "top1": float(row.get("top1", 0)),
            "top5": float(row.get("top5", 0)),
            "top10": float(row.get("top10", 0)),
            "top20": float(row.get("top20", 0)),
            "top50": float(row.get("top50", 0)),
            "mrr": float(row.get("mrr", 0)),
            "retrieval_mode": row.get("retrieval_mode", "unknown"),
            "candidate_coverage": float(row.get("candidate_coverage", 0)),
            "source": "gate_c",
        })

    # From D4A0 baseline CSV
    d4a0_baseline = D4A0_DIR / "d4a0_baseline_reproduction.csv"
    d4a0_rows = read_csv_dicts(d4a0_baseline)
    for row in d4a0_rows:
        method = row.get("baseline", "unknown")
        all_metrics.append({
            "method": f"D4A0_{method}",
            "n_queries": int(float(row.get("n_queries", 0))),
            "top1": float(row.get("top1", 0)),
            "top5": float(row.get("top5", 0)),
            "top10": float(row.get("top10", 0)),
            "top20": float(row.get("top20", 0)),
            "top50": float(row.get("top50", 0)),
            "mrr": float(row.get("mrr", 0)),
            "retrieval_mode": "full_matrix",
            "candidate_coverage": 0,
            "source": "d4a0_baseline",
        })

    # From D4A1 baseline CSV
    d4a1_baseline = D4A1_DIR / "d4a1_baseline_reproduction.csv"
    d4a1_rows = read_csv_dicts(d4a1_baseline)
    for row in d4a1_rows:
        method = row.get("baseline", "unknown")
        all_metrics.append({
            "method": f"D4A1_{method}",
            "n_queries": int(float(row.get("n_queries", 0))),
            "top1": float(row.get("top1", 0)),
            "top5": float(row.get("top5", 0)),
            "top10": float(row.get("top10", 0)),
            "top20": float(row.get("top20", 0)),
            "top50": float(row.get("top50", 0)),
            "mrr": float(row.get("mrr", 0)),
            "retrieval_mode": "full_matrix",
            "candidate_coverage": 0,
            "source": "d4a1_baseline",
        })

    return all_metrics


def save_baseline_comparison(all_metrics: List[Dict[str, Any]]) -> None:
    """Save baseline comparison CSV."""
    # Deduplicate and organize
    seen = set()
    deduped = []
    for m in all_metrics:
        key = m["method"]
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    write_csv(OUT_DIR / "d4a2g_safe_baseline_comparison.csv", deduped)


# ===================================================================
# Part 6: Failure Diagnosis
# ===================================================================

def diagnose_failures(gate_a: Optional[Dict], gate_b: Optional[Dict],
                       gate_c: Optional[Dict]) -> List[Dict[str, str]]:
    """Classify gate failures."""
    diagnoses: List[Dict[str, str]] = []

    if gate_a:
        a_verdict = gate_a.get("verdict", "UNKNOWN")
        if "FAIL" in a_verdict:
            diagnoses.append({
                "gate": "A",
                "verdict": a_verdict,
                "diagnosis": "Delta vectors lack meaningful structure. No significant within-transform/within-attachment similarity beyond random. Options: (1) increase projection_dim, (2) use learned embeddings, (3) check if Morgan FP captures needed chemistry.",
                "severity": "HIGH",
                "recommendation": "Investigate embedding quality before proceeding with delta-based methods.",
            })

    if gate_b:
        b_verdict = gate_b.get("verdict", "UNKNOWN")
        if "FAIL" in b_verdict:
            diagnoses.append({
                "gate": "B",
                "verdict": b_verdict,
                "diagnosis": "Delta vectors are not predictable from old embedding + attachment context. Even simple mean-delta baseline not beaten. Options: (1) richer context features, (2) non-linear model insufficient, (3) delta fundamentally stochastic.",
                "severity": "HIGH",
                "recommendation": "Reconsider delta-prediction approach. Evaluate if zero-delta is acceptable for downstream tasks.",
            })

    if gate_c:
        c_verdict = gate_c.get("verdict", "UNKNOWN")
        if "FAIL" in c_verdict:
            diagnoses.append({
                "gate": "C",
                "verdict": c_verdict,
                "diagnosis": "Learned delta does not improve retrieval over zero-delta baseline. The predicted delta adds noise rather than signal. Options: (1) better delta predictor, (2) different embedding space, (3) transform vectors not viable for this data.",
                "severity": "CRITICAL",
                "recommendation": "Strongly consider abandoning transform-vector approach in favor of direct candidate ranking (D4A1/D4A3).",
            })

    if not diagnoses:
        diagnoses.append({
            "gate": "ALL",
            "verdict": "ALL_PASS",
            "diagnosis": "All gates passed. D4A2G transform-vector approach is feasible.",
            "severity": "INFO",
            "recommendation": "Proceed to full implementation with learned delta predictor.",
        })

    return diagnoses


# ===================================================================
# Verdict determination
# ===================================================================

def determine_overall_verdict(gate_a: Optional[Dict], gate_b: Optional[Dict],
                               gate_c: Optional[Dict]) -> str:
    """Determine overall verdict A through I."""
    a_ok = gate_a and "PASS" in gate_a.get("verdict", "")
    b_ok = gate_b and "PASS" in gate_b.get("verdict", "")
    c_ok = gate_c and "PASS" in gate_c.get("verdict", "")

    if a_ok and b_ok and c_ok:
        return "A"  # All gates pass — full go
    if a_ok and b_ok and not c_ok:
        return "B"  # Gate C fails but structure and predictability OK
    if a_ok and not b_ok and c_ok:
        return "C"  # Gate B fails but structure and retrieval OK
    if not a_ok and b_ok and c_ok:
        return "D"  # Gate A fails but prediction and retrieval OK
    if a_ok and not b_ok and not c_ok:
        return "E"  # Only structure found
    if not a_ok and b_ok and not c_ok:
        return "F"  # Only prediction works (unusual)
    if not a_ok and not b_ok and c_ok:
        return "G"  # Only retrieval works (zero-delta sufficient)
    if not a_ok and not b_ok and not c_ok:
        return "H"  # All gates fail
    return "I"  # Incomplete evaluation


# ===================================================================
# Q&A
# ===================================================================

def answer_questions(preflight: Optional[Dict], split: Optional[Dict],
                      emb_config: Optional[Dict], delta_summary: Optional[Dict],
                      gate_a: Optional[Dict], gate_b: Optional[Dict],
                      gate_c: Optional[Dict]) -> Dict[str, str]:
    """Answer Q1-Q12 from task.md."""
    answers: Dict[str, str] = {}

    # Q1: Can we compute embeddings for all fragments?
    n_frag = (emb_config or {}).get("n_fragments", 0)
    answers["Q1_fragment_embeddings"] = f"YES: {n_frag} fragments embedded (dim={emb_config.get('projection_dim', '?')})"

    # Q2: How many delta records?
    n_train = (delta_summary or {}).get("n_train_delta_records", 0)
    n_val = (delta_summary or {}).get("n_val_delta_records", 0)
    answers["Q2_delta_records"] = f"Train={n_train}, Val={n_val}"

    # Q3: Do delta vectors have structure?
    a_detail = (gate_a or {}).get("detail", "N/A")
    a_verdict = (gate_a or {}).get("verdict", "N/A")
    answers["Q3_delta_structure"] = f"{a_verdict}: {a_detail}"

    # Q4: Are delta vectors predictable?
    b_verdict = (gate_b or {}).get("verdict", "N/A")
    b_detail = (gate_b or {}).get("detail", "N/A")
    answers["Q4_delta_predictability"] = f"{b_verdict}: {b_detail}"

    # Q5: How does learned delta compare to zero delta?
    c_verdict = (gate_c or {}).get("verdict", "N/A")
    c_detail = (gate_c or {}).get("detail", "N/A")
    answers["Q5_learned_vs_zero_delta"] = f"{c_verdict}: {c_detail}"

    # Q6: Is the delta method viable?
    overall = determine_overall_verdict(gate_a, gate_b, gate_c)
    if overall in ("A",):
        answers["Q6_overall_viability"] = f"VIABLE (verdict {overall})"
    elif overall in ("B", "C", "D"):
        answers["Q6_overall_viability"] = f"MARGINAL (verdict {overall})"
    else:
        answers["Q6_overall_viability"] = f"NOT_VIABLE (verdict {overall})"

    # Q7: How does D4A2G compare to D4A0 baselines?
    answers["Q7_vs_d4a0"] = "See baseline_comparison.csv for full comparison"

    # Q8: How does D4A2G compare to D4A1?
    answers["Q8_vs_d4a1"] = "See baseline_comparison.csv for full comparison"

    # Q9: What are the main failure modes?
    diags = diagnose_failures(gate_a, gate_b, gate_c)
    failures = [d["diagnosis"][:100] for d in diags if d["severity"] in ("HIGH", "CRITICAL")]
    answers["Q9_failure_modes"] = "; ".join(failures) if failures else "No critical failures"

    # Q10: Would a learned embedder improve things?
    checkpoint_found = (emb_config or {}).get("checkpoint_found", False)
    ev = (emb_config or {}).get("explained_variance", 0)
    answers["Q10_learned_embedder"] = f"Checkpoint{' found' if checkpoint_found else ' not found'}. SVD explained variance={ev:.4f}"

    # Q11: Resource adequacy?
    rv = (preflight or {}).get("verdict", "UNKNOWN")
    answers["Q11_resource_adequacy"] = f"Preflight: {rv}"

    # Q12: Next step recommendation?
    if overall == "A":
        answers["Q12_recommendation"] = "PROCEED: Implement learned delta predictor in D4A2G full pipeline"
    elif overall in ("B", "C", "D"):
        answers["Q12_recommendation"] = "PROCEED_WITH_CAVEATS: Address specific gate failures before full pipeline"
    else:
        answers["Q12_recommendation"] = "RECONSIDER: Transform-vector approach not viable. Consider D4A3 (direct generation) or D4A1-enhanced ranking"

    return answers


# ===================================================================
# Generate verdict markdown
# ===================================================================

def generate_verdict_md(gate_a: Optional[Dict], gate_b: Optional[Dict],
                         gate_c: Optional[Dict], overall_verdict: str,
                         answers: Dict[str, str],
                         all_metrics: List[Dict[str, Any]],
                         diagnoses: List[Dict[str, str]],
                         preflight: Optional[Dict]) -> str:
    """Generate the full D4A2G_SAFE_TRANSFORM_VECTOR_GATE_VERDICT.md."""
    lines = [
        f"# D4A2G-SAFE Transform Vector Gate Verdict",
        f"Date: {TIMESTAMP}",
        f"Overall Verdict: **{overall_verdict}**",
        "",
        "## Gate Results",
        f"- Gate A (Structure): {(gate_a or {}).get('verdict', 'N/A')}",
        f"  - Detail: {(gate_a or {}).get('detail', 'N/A')}",
        f"- Gate B (Predictability): {(gate_b or {}).get('verdict', 'N/A')}",
        f"  - Detail: {(gate_b or {}).get('detail', 'N/A')}",
        f"- Gate C (Retrieval): {(gate_c or {}).get('verdict', 'N/A')}",
        f"  - Detail: {(gate_c or {}).get('detail', 'N/A')}",
        "",
        "## Resource Preflight",
        f"- Verdict: {(preflight or {}).get('verdict', 'N/A')}",
        "",
        "## Q&A (12 Questions)",
    ]
    for q, a in answers.items():
        lines.append(f"- **{q}**: {a}")

    lines.extend([
        "",
        "## Failure Diagnosis",
    ])
    for d in diagnoses:
        lines.append(f"- Gate {d['gate']}: {d['verdict']} ({d['severity']})")
        lines.append(f"  - {d['diagnosis']}")
        lines.append(f"  - Recommendation: {d['recommendation']}")

    # Baseline comparison table
    lines.extend([
        "",
        "## Baseline Comparison",
        "",
        "| Method | n_queries | Top1 | Top5 | Top10 | Top20 | MRR | Retrieval Mode |",
        "|--------|-----------|------|------|-------|-------|-----|----------------|",
    ])
    for m in sorted(all_metrics, key=lambda x: x.get("top10", 0), reverse=True):
        lines.append(
            f"| {m['method']} | {m.get('n_queries', '?')} | "
            f"{m.get('top1', 0):.4f} | {m.get('top5', 0):.4f} | "
            f"{m.get('top10', 0):.4f} | {m.get('top20', 0):.4f} | "
            f"{m.get('mrr', 0):.4f} | {m.get('retrieval_mode', '?')} |"
        )

    lines.extend([
        "",
        "## Skeptical Review",
        "",
        "### 1. Sample Size",
        "The delta dataset is capped. Results may not generalize to full distribution.",
        "",
        "### 2. Morgan FP Fallback Bias",
        "Morgan fingerprints (radius=2, 2048 bits) projected via SVD may lose subtle chemical information. "
        "Learned embeddings from the contrastive model could yield different delta structure.",
        "",
        "### 3. Delta Structure Reality",
        "Delta vectors in projected Morgan space may reflect fingerprint bit-collision patterns "
        "rather than genuine chemical transformation structure.",
        "",
        "### 4. Zero-Delta Comparison",
        "If zero-delta is competitive, the entire delta-prediction approach is questionable. "
        "A trivial copy of old fragment embedding should not be hard to beat.",
        "",
        "### 5. Resource Caps Impact",
        f"Caps on train delta records ({all_metrics[0].get('n_queries', '?') if all_metrics else '?'} test queries) "
        "limit statistical power. Results may shift with full data.",
        "",
        "### 6. Retrieval Mode",
        "Chunked/exact retrieval may differ from full-matrix exhaustive search used in D4A0/D4A1 baselines.",
        "",
        "### 7. Narrative Bias",
        "The delta approach is appealing because it maps to the 'z_replacement = z_old + Δz' intuition, "
        "but the data may not support this linear assumption in projected Morgan space.",
        "",
        "### 8. Dual Encoder Simplicity",
        "A dual encoder (old_encoder, replacement_encoder) that directly predicts z_replacement "
        "without explicit delta computation is a simpler, untested alternative.",
        "",
        "## Verdict Interpretation",
    ])

    verdict_map = {
        "A": "All gates pass. Transform-vector delta approach is viable. Proceed to full D4A2G pipeline.",
        "B": "Structure and predictability confirmed, but learned delta does not beat zero-delta retrieval. May need stronger predictor.",
        "C": "Structure and retrieval OK, but delta not predictable from available features. Consider richer context.",
        "D": "Delta structure weak but prediction and retrieval work. Check embedding quality.",
        "E": "Only structure found. Delta not predictable, retrieval not improved. Reconsider approach.",
        "F": "Prediction works despite weak structure. Unusual — check for artifacts.",
        "G": "Only retrieval works (zero-delta sufficient). Delta prediction may be unnecessary overhead.",
        "H": "All gates fail. Transform vector approach not viable for this data.",
        "I": "Incomplete evaluation. Some gates not run.",
    }
    lines.append(f"**Verdict {overall_verdict}**: {verdict_map.get(overall_verdict, 'Unknown')}")
    lines.append("")

    return "\n".join(lines)


def generate_decision_log(overall_verdict: str, answers: Dict[str, str]) -> str:
    """Generate MAIN_DECISION_LOG.md."""
    lines = [
        f"# D4A2G-SAFE Main Decision Log",
        f"Date: {TIMESTAMP}",
        f"Verdict: **{overall_verdict}**",
        f"D4A2G Allowed: {'YES' if overall_verdict == 'A' else 'CONDITIONAL' if overall_verdict in ('B', 'C', 'D') else 'NO'}",
        "",
        "## Key Answers",
    ]
    for q, a in answers.items():
        lines.append(f"- {q}: {a}")

    lines.extend([
        "",
        "## Decision",
    ])

    if overall_verdict == "A":
        lines.append("Proceed with D4A2G transform-vector full pipeline using learned delta predictor.")
    elif overall_verdict in ("B", "C", "D"):
        lines.append("Conditional proceed. Address specific gate failures first.")
    else:
        lines.append("Do not proceed with D4A2G. Consider alternative directions (D4A3, D4A1-enhanced).")

    lines.append("")
    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    log.info("=" * 60)
    log.info("D4A2G-SAFE Summarize + Baseline Compare + Final Verdict")
    log.info("=" * 60)

    # Load all gate summaries
    gate_a = load_gate_summary("A")
    gate_b = load_gate_summary("B")
    gate_c = load_gate_summary("C")
    preflight = load_resource_preflight()
    split = load_split_summary()
    emb_config = load_embedding_config()
    delta_summary = load_delta_dataset_summary()

    # Part 5: Baseline comparison
    log.info("Part 5: Baseline comparison")
    all_metrics = gather_all_metrics()
    save_baseline_comparison(all_metrics)
    log.info("  Collected %d metric entries", len(all_metrics))

    # Part 6: Failure diagnosis
    log.info("Part 6: Failure diagnosis")
    diagnoses = diagnose_failures(gate_a, gate_b, gate_c)
    write_csv(OUT_DIR / "d4a2g_safe_failure_diagnosis.csv", diagnoses)

    # Overall verdict
    overall_verdict = determine_overall_verdict(gate_a, gate_b, gate_c)
    log.info("Overall verdict: %s", overall_verdict)

    # Q&A
    answers = answer_questions(preflight, split, emb_config, delta_summary,
                                gate_a, gate_b, gate_c)
    for q, a in sorted(answers.items()):
        log.info("  %s: %s", q, a)

    # Generate verdict markdown
    verdict_md = generate_verdict_md(gate_a, gate_b, gate_c, overall_verdict,
                                      answers, all_metrics, diagnoses, preflight)
    verdict_path = OUT_DIR / "D4A2G_SAFE_TRANSFORM_VECTOR_GATE_VERDICT.md"
    verdict_path.write_text(verdict_md, encoding="utf-8")
    log.info("Verdict written to %s", verdict_path)

    # Generate decision log
    decision_md = generate_decision_log(overall_verdict, answers)
    decision_path = OUT_DIR / "MAIN_DECISION_LOG.md"
    decision_path.write_text(decision_md, encoding="utf-8")
    log.info("Decision log written to %s", decision_path)

    # Save answers as JSON
    save_json(OUT_DIR / "d4a2g_safe_qa_answers.json", answers)

    log.info("=" * 60)
    log.info("D4A2G-SAFE summary complete. Verdict: %s", overall_verdict)
    log.info("=" * 60)

    write_complete_marker("d4a2g_safe_summarize")


if __name__ == "__main__":
    main()
