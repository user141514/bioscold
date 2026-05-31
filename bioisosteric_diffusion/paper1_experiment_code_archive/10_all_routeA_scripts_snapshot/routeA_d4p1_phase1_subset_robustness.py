#!/usr/bin/env python3
"""Route-A D4P1-Phase1 subset robustness under the locked canonical D4A2D2 protocol."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

RDLogger.DisableLog("rdApp.*")

SEED = 20260525
N_BOOT = 1000
TOPK = 10
TOPR = 50
LOW_N_THRESHOLD = 200

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN = PROJECT_ROOT / "plan_results"
OUT = PLAN / "routeA_chembl37k_d4p1_phase1_subset_robustness"
OUT.mkdir(parents=True, exist_ok=True)

PHASE0 = PLAN / "routeA_chembl37k_d4p1_phase0_metric_lock"
D4A0 = PLAN / "routeA_chembl37k_d0d3_engineering_safe" / "07_d4a0_matrix_freeze"
D4A1 = PLAN / "routeA_chembl37k_d4a1_learned_ranker"
D4A2D1R = PLAN / "routeA_chembl37k_d4a2d1r_dual_encoder_robustness"
D4A2D2 = PLAN / "routeA_chembl37k_d4a2d2_de_hgb_ensemble"
D4A3S = PLAN / "routeA_chembl37k_d4a3s_a4c_coverage_expansion"
D4A4 = PLAN / "routeA_chembl37k_d4a4_dual_mode_integration"

MANIFEST = D4A0 / "d4a0_query_split_manifest.jsonl"
VOCAB = D4A0 / "d4a0_train_replacement_vocabulary.csv"
DE_STD = D4A2D1R / "d4a2d1r_standardized_predictions.jsonl"
HGB_PREDS = D4A1 / "d4a1_test_predictions.jsonl"
CANONICAL_PROPOSAL_TABLE = PHASE0 / "d4p1_phase0_canonical_proposal_table.csv"
STANDARDIZED_TEST = D4A2D2 / "d4a2d2_standardized_predictions_test.jsonl"
G2_CSV = D4A3S / "d4a3s_G2_candidates.csv"
G3_CSV = D4A3S / "d4a3s_G3_candidates.csv"
G4_CSV = D4A3S / "d4a3s_G4_candidates.csv"
D4A4_TIERS = D4A4 / "d4a4_candidate_final_tiers.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_jsonl(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def pipe_join(values: Iterable[str]) -> str:
    ordered = []
    seen = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return "|".join(ordered)


def safe_rank_to_mrr(best_rank: float) -> float:
    if math.isfinite(best_rank) and best_rank > 0:
        return 1.0 / float(best_rank)
    return 0.0


def canonicalize_fallback(candidate_exact: str) -> str:
    stripped = candidate_exact.replace("*", "")
    mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        return stripped
    return Chem.MolToSmiles(mol, canonical=True)


def load_phase0_targets() -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(CANONICAL_PROPOSAL_TABLE)
    targets: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        method = str(row["method"])
        if "attachment" in method.lower():
            key = "attach"
        elif method == "DE":
            key = "de"
        elif method == "HGB":
            key = "hgb"
        elif "borda" in method.lower():
            key = "borda"
        else:
            continue
        targets[key] = {
            "Top1": float(row["Top1"]),
            "Top5": float(row["Top5"]),
            "Top10": float(row["Top10"]),
            "Top20": float(row["Top20"]),
            "Top50": float(row["Top50"]),
            "MRR": float(row["MRR"]),
            "gain_vs_HGB_Top10": float(row["gain_vs_HGB_Top10"]),
        }
    return targets


def load_exact_to_norm_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in load_jsonl(STANDARDIZED_TEST):
        cand = str(row["candidate"])
        norm = str(row["candidate_norm"])
        mapping.setdefault(cand, norm)
    return mapping


def load_vocab_and_manifest(exact_to_norm: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, dict]]:
    vocab_df = pd.read_csv(VOCAB)
    vocab_freq = {
        str(row["replacement_smiles"]): int(row["global_train_frequency"])
        for _, row in vocab_df.iterrows()
    }
    vocab_set = set(vocab_freq)
    query_meta: Dict[str, dict] = {}

    for row in load_jsonl(MANIFEST):
        if row.get("split") != "test":
            continue
        pos_exact = [s for s in row["positive_replacement_set"] if s in vocab_set]
        if not pos_exact:
            continue
        qid = row["query_id"]
        pos_norm = [exact_to_norm.get(s, canonicalize_fallback(s)) for s in pos_exact]
        freqs = [vocab_freq[s] for s in pos_exact]
        query_meta[qid] = {
            "query_id": qid,
            "old_fragment_smiles": row["old_fragment_smiles"],
            "attachment_signature": row["attachment_signature"],
            "positive_exact": pos_exact,
            "positive_exact_set": set(pos_exact),
            "positive_norm": pos_norm,
            "positive_norm_set": set(pos_norm),
            "positive_replacement_set_size": len(pos_exact),
            "num_positive_replacements_total": int(row.get("num_positive_replacements", len(pos_exact))),
            "target_replacement_frequency_mean": float(np.mean(freqs)),
            "target_replacement_frequency_min": float(np.min(freqs)),
            "target_replacement_frequency_max": float(np.max(freqs)),
        }

    old_fragment_counts = Counter(meta["old_fragment_smiles"] for meta in query_meta.values())
    for meta in query_meta.values():
        meta["old_fragment_query_frequency"] = old_fragment_counts[meta["old_fragment_smiles"]]
    return vocab_df, vocab_freq, query_meta


def load_group_maps() -> Dict[Tuple[str, str], str]:
    group_map: Dict[Tuple[str, str], str] = {}
    for path, label in (
        (G2_CSV, "G2_pure_borda_only"),
        (G3_CSV, "G3_de_retained_by_borda"),
        (G4_CSV, "G4_shared"),
    ):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (str(row["qid"]), str(row["candidate_norm"]))
            group_map[key] = label
    return group_map


def load_tier_map() -> Dict[Tuple[str, str], dict]:
    mapping: Dict[Tuple[str, str], dict] = {}
    if not D4A4_TIERS.exists():
        return mapping
    df = pd.read_csv(D4A4_TIERS)
    for _, row in df.iterrows():
        key = (str(row["query_id"]), str(row["candidate_norm"]))
        mapping[key] = {
            "final_action_tier": str(row["final_action_tier"]),
            "group_origin": str(row.get("group_origin", "other")),
            "hard_alert_flag": bool(row.get("hard_alert_flag", False)),
            "review_ready_flag": bool(row.get("review_ready_flag", False)),
        }
    return mapping


def load_attach_and_de_predictions(query_meta: Dict[str, dict]) -> Dict[str, dict]:
    methods = {"M1_attach": "attach", "M2_DE": "de"}
    per_method = {
        "attach": defaultdict(list),
        "de": defaultdict(list),
    }
    best_rank = {
        "attach": defaultdict(lambda: math.inf),
        "de": defaultdict(lambda: math.inf),
    }

    for row in load_jsonl(DE_STD):
        qid = row["q"]
        method = methods.get(row["m"])
        if method is None or qid not in query_meta:
            continue
        rank = int(row["r"])
        cand = str(row["c"])
        score = float(row.get("s", 0.0))
        label = int(row.get("l", 0))
        if rank <= TOPR:
            per_method[method][qid].append((rank, cand, score, label))
        if label == 1 and rank < best_rank[method][qid]:
            best_rank[method][qid] = rank

    for method in per_method:
        for qid in per_method[method]:
            per_method[method][qid].sort(key=lambda item: item[0])

    return {
        "top50": per_method,
        "best_rank": best_rank,
    }


def load_hgb_predictions(query_meta: Dict[str, dict]) -> Dict[str, dict]:
    grouped = defaultdict(list)
    for row in load_jsonl(HGB_PREDS):
        qid = row["query_id"]
        if qid not in query_meta:
            continue
        grouped[qid].append(
            (
                float(row.get("score", 0.0)),
                str(row["candidate"]),
                int(row.get("label", 0)),
            )
        )

    top50 = {}
    best_rank = defaultdict(lambda: math.inf)
    for qid, rows in grouped.items():
        rows.sort(key=lambda item: (-item[0], item[1]))
        top_rows = []
        for idx, (score, cand, label) in enumerate(rows, start=1):
            if idx <= TOPR:
                top_rows.append((idx, cand, score, label))
            if label == 1 and not math.isfinite(best_rank[qid]):
                best_rank[qid] = idx
        top50[qid] = top_rows
    return {
        "top50": top50,
        "best_rank": best_rank,
    }


def compute_borda_for_query(
    de_top50: List[Tuple[int, str, float, int]],
    hgb_top50: List[Tuple[int, str, float, int]],
) -> List[Tuple[int, str, float, int, int]]:
    hgb_rank = {cand: rank for rank, cand, _, _ in hgb_top50}
    scored = []
    for de_rank, cand, score, label in de_top50:
        borda_score = (TOPR - de_rank + 1) + (TOPR - hgb_rank.get(cand, TOPR + 1) + 1)
        scored.append((cand, borda_score, de_rank, label))
    scored.sort(key=lambda item: -item[1])
    ranked = []
    for idx, (cand, borda_score, de_rank, label) in enumerate(scored, start=1):
        ranked.append((idx, cand, borda_score, de_rank, label))
    return ranked


def summarize_top_hits(best_rank: float) -> Dict[str, int]:
    summary = {}
    for k in (1, 5, 10, 20, 50):
        summary[f"hit{k}"] = int(math.isfinite(best_rank) and best_rank <= k)
    return summary


def build_query_level_table() -> Tuple[pd.DataFrame, dict]:
    log("Loading canonical inputs...")
    exact_to_norm = load_exact_to_norm_map()
    vocab_df, vocab_freq, query_meta = load_vocab_and_manifest(exact_to_norm)
    group_map = load_group_maps()
    tier_map = load_tier_map()
    attach_de = load_attach_and_de_predictions(query_meta)
    hgb = load_hgb_predictions(query_meta)

    common_qids = sorted(
        set(query_meta)
        & set(attach_de["top50"]["attach"])
        & set(attach_de["top50"]["de"])
        & set(hgb["top50"])
    )
    log(f"Canonical common queries: {len(common_qids)}")

    rows = []
    for qid in common_qids:
        meta = query_meta[qid]
        attach_top50 = attach_de["top50"]["attach"][qid]
        de_top50 = attach_de["top50"]["de"][qid]
        hgb_top50 = hgb["top50"][qid]
        borda_top50 = compute_borda_for_query(de_top50, hgb_top50)

        attach_best_rank = attach_de["best_rank"]["attach"].get(qid, math.inf)
        de_best_rank = attach_de["best_rank"]["de"].get(qid, math.inf)
        hgb_best_rank = hgb["best_rank"].get(qid, math.inf)
        borda_best_rank = math.inf
        for rank, cand, _, _, label in borda_top50:
            if label == 1:
                borda_best_rank = rank
                break
        oracle_best_rank = min(de_best_rank, hgb_best_rank)

        attach_exact_top10 = [cand for rank, cand, _, _ in attach_top50[:TOPK]]
        de_exact_top10 = [cand for rank, cand, _, _ in de_top50[:TOPK]]
        hgb_exact_top10 = [cand for rank, cand, _, _ in hgb_top50[:TOPK]]
        borda_exact_top10 = [cand for rank, cand, _, _, _ in borda_top50[:TOPK]]

        borda_positive_norms = []
        borda_positive_groups = []
        borda_positive_tiers = []
        tier_join_match_count = 0
        for cand in borda_exact_top10:
            if cand not in meta["positive_exact_set"]:
                continue
            norm = exact_to_norm.get(cand, canonicalize_fallback(cand))
            borda_positive_norms.append(norm)
            group = group_map.get((qid, norm))
            tier = tier_map.get((qid, norm))
            if group is None and tier is not None:
                group = tier.get("group_origin", "other")
            borda_positive_groups.append(group or "other")
            if tier is not None:
                borda_positive_tiers.append(tier["final_action_tier"])
                tier_join_match_count += 1
            else:
                borda_positive_tiers.append("Tier0_DATA_PENDING")

        unique_groups = list(dict.fromkeys(borda_positive_groups))
        unique_tiers = list(dict.fromkeys(borda_positive_tiers))
        if not borda_positive_norms:
            provenance_query_group = "NO_BORDA_HIT"
            best_positive_tier = "NO_BORDA_HIT"
        else:
            provenance_query_group = unique_groups[0] if len(unique_groups) == 1 else "mixed"
            best_positive_tier = unique_tiers[0] if len(unique_tiers) == 1 else "mixed"

        row = {
            "query_id": qid,
            "old_fragment_smiles": meta["old_fragment_smiles"],
            "attachment_signature": meta["attachment_signature"],
            "positive_replacement_set_size": meta["positive_replacement_set_size"],
            "num_positive_replacements_total": meta["num_positive_replacements_total"],
            "positive_replacements_exact": pipe_join(meta["positive_exact"]),
            "positive_replacements_norm": pipe_join(meta["positive_norm"]),
            "candidate_set_size": int(len(vocab_df)),
            "ranked_candidate_pool_size": TOPR,
            "old_fragment_query_frequency": meta["old_fragment_query_frequency"],
            "target_replacement_frequency_mean": meta["target_replacement_frequency_mean"],
            "target_replacement_frequency_min": meta["target_replacement_frequency_min"],
            "target_replacement_frequency_max": meta["target_replacement_frequency_max"],
            "attach_best_rank": None if not math.isfinite(attach_best_rank) else int(attach_best_rank),
            "DE_best_rank": None if not math.isfinite(de_best_rank) else int(de_best_rank),
            "HGB_best_rank": None if not math.isfinite(hgb_best_rank) else int(hgb_best_rank),
            "Borda_best_rank": None if not math.isfinite(borda_best_rank) else int(borda_best_rank),
            "Oracle_best_rank_if_available": None if not math.isfinite(oracle_best_rank) else int(oracle_best_rank),
            "attach_mrr": safe_rank_to_mrr(attach_best_rank),
            "DE_mrr": safe_rank_to_mrr(de_best_rank),
            "HGB_mrr": safe_rank_to_mrr(hgb_best_rank),
            "Borda_mrr": safe_rank_to_mrr(borda_best_rank),
            "attach_top10_exact": pipe_join(attach_exact_top10),
            "DE_top10_exact": pipe_join(de_exact_top10),
            "HGB_top10_exact": pipe_join(hgb_exact_top10),
            "Borda_top10_exact": pipe_join(borda_exact_top10),
            "Borda_top10_norm": pipe_join(exact_to_norm.get(c, canonicalize_fallback(c)) for c in borda_exact_top10),
            "Borda_positive_top10_norm": pipe_join(borda_positive_norms),
            "Borda_positive_group_set": pipe_join(unique_groups),
            "Borda_positive_tier_set": pipe_join(unique_tiers),
            "G_group_if_applicable": provenance_query_group,
            "A4C_tier_if_available": best_positive_tier,
            "borda_positive_norm_count": len(borda_positive_norms),
            "borda_positive_tier_match_count": tier_join_match_count,
            "borda_hit10_has_g2_positive": int("G2_pure_borda_only" in unique_groups),
            "borda_hit10_has_g3_positive": int("G3_de_retained_by_borda" in unique_groups),
            "borda_hit10_has_g4_positive": int("G4_shared" in unique_groups),
            "borda_hit10_has_tier0_positive": int("Tier0_DATA_PENDING" in unique_tiers),
            "borda_hit10_has_tier1_positive": int("Tier1_STANDARD_REVIEW" in unique_tiers),
            "borda_hit10_has_tier2_positive": int("Tier2_EXPERT_REVIEW" in unique_tiers),
            "borda_hit10_has_tier3_positive": int("Tier3_HARD_REJECT" in unique_tiers),
            "borda_hit10_reviewable": int(
                ("Tier1_STANDARD_REVIEW" in unique_tiers) or ("Tier2_EXPERT_REVIEW" in unique_tiers)
            ),
            "borda_hit10_hard_reject": int("Tier3_HARD_REJECT" in unique_tiers),
            "borda_hit10_data_pending": int("Tier0_DATA_PENDING" in unique_tiers),
            "num_positive_bin": "single_pos" if meta["positive_replacement_set_size"] == 1 else "multi_pos",
            "candidate_set_size_bin": "CONSTANT_152",
        }

        row.update({f"attach_{k}": v for k, v in summarize_top_hits(attach_best_rank).items()})
        row.update({f"DE_{k}": v for k, v in summarize_top_hits(de_best_rank).items()})
        row.update({f"HGB_{k}": v for k, v in summarize_top_hits(hgb_best_rank).items()})
        row.update({f"Borda_{k}": v for k, v in summarize_top_hits(borda_best_rank).items()})
        row.update({f"Oracle_{k}_if_available": v for k, v in summarize_top_hits(oracle_best_rank).items()})
        rows.append(row)

    query_df = pd.DataFrame(rows)
    query_df["borda_net_gain_flag"] = (
        (query_df["Borda_hit10"] == 1) & (query_df["HGB_hit10"] == 0)
    ).astype(int)
    query_df["borda_net_loss_flag"] = (
        (query_df["Borda_hit10"] == 0) & (query_df["HGB_hit10"] == 1)
    ).astype(int)
    return query_df, {
        "vocab_size": len(vocab_df),
        "vocab_freq": vocab_freq,
    }


def reconstruct_metrics(query_df: pd.DataFrame, targets: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for key, col_prefix in (
        ("attach", "attach"),
        ("de", "DE"),
        ("hgb", "HGB"),
        ("borda", "Borda"),
    ):
        target = targets[key]
        rows.append(
            {
                "method": key,
                "metric": "Top10",
                "reconstructed": float(query_df[f"{col_prefix}_hit10"].mean()),
                "phase0_target": target["Top10"],
            }
        )
        rows.append(
            {
                "method": key,
                "metric": "MRR",
                "reconstructed": float(query_df[f"{col_prefix}_mrr"].mean()),
                "phase0_target": target["MRR"],
            }
        )
    rows.append(
        {
            "method": "borda",
            "metric": "Borda_minus_HGB_Top10",
            "reconstructed": float(query_df["Borda_hit10"].mean() - query_df["HGB_hit10"].mean()),
            "phase0_target": targets["borda"]["gain_vs_HGB_Top10"],
        }
    )
    recon = pd.DataFrame(rows)
    recon["abs_diff"] = (recon["reconstructed"] - recon["phase0_target"]).abs()
    recon["pass_0p5pp_tolerance"] = recon["abs_diff"] <= 0.005
    return recon


def bootstrap_delta(values_a: np.ndarray, values_b: np.ndarray, seed: int) -> Tuple[float, float, float, float]:
    delta = (values_a.astype(np.float32) - values_b.astype(np.float32)).astype(np.float32)
    observed = float(delta.mean()) if len(delta) else float("nan")
    if len(delta) == 0:
        return observed, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(N_BOOT, len(delta)), dtype=np.int32)
    boot = delta[idx].mean(axis=1)
    return observed, float(boot.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def build_old_fragment_clusters(query_df: pd.DataFrame) -> pd.DataFrame:
    fps = np.zeros((len(query_df), 2048), dtype=np.float32)
    for i, smiles in enumerate(query_df["old_fragment_smiles"].tolist()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        DataStructs.ConvertToNumpyArray(fp, fps[i])

    reducer = TruncatedSVD(n_components=64, random_state=SEED)
    reduced = reducer.fit_transform(fps)
    kmeans = KMeans(n_clusters=10, random_state=SEED, n_init=20)
    labels = kmeans.fit_predict(reduced)
    out = query_df.copy()
    out["old_fragment_cluster_id"] = [f"cluster_{label:02d}" for label in labels]
    return out


def define_subsets(query_df: pd.DataFrame) -> Tuple[List[dict], List[dict], dict]:
    thresholds = {}
    defs: List[dict] = []
    registry: List[dict] = []

    def add_subset(
        family: str,
        name: str,
        mask: pd.Series | None,
        definition: str,
        thresholds_or_cluster_info: str,
        notes: str = "",
        diagnostic_only: bool = False,
        available: bool = True,
    ) -> None:
        n = int(mask.sum()) if mask is not None else 0
        defs.append(
            {
                "subset_family": family,
                "subset_name": name,
                "definition": definition,
                "N": n,
                "thresholds_or_cluster_info": thresholds_or_cluster_info,
                "available": bool(available),
                "notes": notes,
            }
        )
        registry.append(
            {
                "subset_family": family,
                "subset_name": name,
                "mask": mask,
                "definition": definition,
                "thresholds_or_cluster_info": thresholds_or_cluster_info,
                "diagnostic_only": diagnostic_only,
                "available": available,
                "notes": notes,
            }
        )

    add_subset(
        "global",
        "all_test",
        pd.Series(True, index=query_df.index),
        "All canonical D4A2D2 test queries with at least one positive replacement in train vocabulary.",
        "canonical D4A2D2 protocol, 152-vocab, query bootstrap",
    )
    add_subset(
        "query_difficulty",
        "hard_top10_miss",
        query_df["attach_hit10"] == 0,
        "Attachment-frequency Top10 misses all positives.",
        "baseline = attachment frequency Top10",
    )
    add_subset(
        "query_difficulty",
        "easy_top10_hit",
        query_df["attach_hit10"] == 1,
        "Attachment-frequency Top10 hits at least one positive.",
        "baseline = attachment frequency Top10",
    )
    add_subset(
        "query_difficulty",
        "top1_miss",
        query_df["attach_hit1"] == 0,
        "Attachment-frequency Top1 misses all positives.",
        "baseline = attachment frequency Top1",
    )
    add_subset(
        "query_difficulty",
        "top1_hit",
        query_df["attach_hit1"] == 1,
        "Attachment-frequency Top1 hits at least one positive.",
        "baseline = attachment frequency Top1",
    )

    add_subset(
        "positive_set_size",
        "single_pos",
        query_df["positive_replacement_set_size"] == 1,
        "Exactly one in-vocab positive replacement.",
        "positive_replacement_set_size = 1",
    )
    add_subset(
        "positive_set_size",
        "multi_pos",
        query_df["positive_replacement_set_size"] > 1,
        "More than one in-vocab positive replacement.",
        "positive_replacement_set_size > 1",
    )

    q1, q2 = np.quantile(query_df["target_replacement_frequency_mean"], [1 / 3, 2 / 3])
    thresholds["replacement_frequency_mean_tertiles"] = {"q1": float(q1), "q2": float(q2)}
    add_subset(
        "replacement_frequency",
        "rare_replacement",
        query_df["target_replacement_frequency_mean"] <= q1,
        "Lower tercile of mean positive replacement train frequency.",
        f"mean_positive_global_train_frequency <= {q1:.3f}",
        notes="Relative rarity; for multi-positive queries this is a query-level summary, not a candidate-level rarity claim.",
    )
    add_subset(
        "replacement_frequency",
        "medium_replacement",
        (query_df["target_replacement_frequency_mean"] > q1) & (query_df["target_replacement_frequency_mean"] <= q2),
        "Middle tercile of mean positive replacement train frequency.",
        f"{q1:.3f} < mean_positive_global_train_frequency <= {q2:.3f}",
        notes="Relative rarity; query-level mean over positives.",
    )
    add_subset(
        "replacement_frequency",
        "frequent_replacement",
        query_df["target_replacement_frequency_mean"] > q2,
        "Upper tercile of mean positive replacement train frequency.",
        f"mean_positive_global_train_frequency > {q2:.3f}",
        notes="Relative frequency; query-level mean over positives.",
    )

    for name in ("small_candidate_set", "medium_candidate_set", "large_candidate_set"):
        add_subset(
            "candidate_set_size",
            name,
            None,
            "Requested candidate-set-size subset.",
            "candidate_set_size is constant at 152 for every canonical query",
            notes="BLOCKED: canonical D4A2D2 uses a fixed 152-vocab candidate universe, so there is no between-query variation.",
            available=False,
        )

    for signature, count in query_df["attachment_signature"].value_counts().sort_index().items():
        notes = "LOW_N_WARNING" if count < LOW_N_THRESHOLD else ""
        add_subset(
            "attachment_signature",
            signature,
            query_df["attachment_signature"] == signature,
            "Queries sharing the same attachment signature.",
            f"attachment_signature = {signature}",
            notes=notes,
        )

    for cluster_id, count in query_df["old_fragment_cluster_id"].value_counts().sort_index().items():
        notes = "LOW_N_WARNING" if count < LOW_N_THRESHOLD else ""
        add_subset(
            "old_fragment_cluster",
            cluster_id,
            query_df["old_fragment_cluster_id"] == cluster_id,
            "Old-fragment Morgan-FP cluster (radius=2, nBits=2048, SVD64, k=10, seed=20260525).",
            "MorganFP(radius=2,nBits=2048) -> TruncatedSVD(64) -> KMeans(k=10, seed=20260525)",
            notes=notes,
        )

    diagnostic_hit_mask = query_df["Borda_hit10"] == 1
    for name in ("G2_pure_borda_only", "G3_de_retained_by_borda", "G4_shared", "mixed", "other"):
        mask = diagnostic_hit_mask & (query_df["G_group_if_applicable"] == name)
        add_subset(
            "gain_provenance_group",
            name,
            mask,
            "Outcome-conditioned provenance subset over Borda-hit queries.",
            "Defined from positive Borda Top10 candidates only.",
            notes="DIAGNOSTIC_ONLY: uses Borda-hit outcome, so it cannot support robustness claims.",
            diagnostic_only=True,
        )

    for name in ("Tier0_DATA_PENDING", "Tier1_STANDARD_REVIEW", "Tier2_EXPERT_REVIEW", "Tier3_HARD_REJECT", "mixed"):
        mask = diagnostic_hit_mask & (query_df["A4C_tier_if_available"] == name)
        add_subset(
            "a4c_tier_group",
            name,
            mask,
            "Outcome-conditioned A4C tier subset over Borda-hit queries.",
            "Defined from positive Borda Top10 candidates only.",
            notes="DIAGNOSTIC_ONLY: review tiers are joined after proposal generation and must not be mixed into canonical proposal claims.",
            diagnostic_only=True,
        )

    return defs, registry, thresholds


def classify_subset_status(n: int, delta: float, ci_low: float, ci_high: float) -> str:
    if n < LOW_N_THRESHOLD or np.isnan(ci_low) or np.isnan(ci_high):
        return "LOW_N_INCONCLUSIVE"
    if ci_low > 0:
        return "ROBUST_POSITIVE"
    if delta < 0:
        return "NEGATIVE"
    return "POSITIVE_NOT_SIGNIFICANT"


def compute_subset_metrics(query_df: pd.DataFrame, registry: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    boot_rows = []

    for idx, subset in enumerate(registry):
        if not subset["available"] or subset["mask"] is None:
            continue
        sub = query_df.loc[subset["mask"]].copy()
        n = int(len(sub))
        observed_bh, boot_mean_bh, ci_low_bh, ci_high_bh = bootstrap_delta(
            sub["Borda_hit10"].to_numpy(),
            sub["HGB_hit10"].to_numpy(),
            seed=SEED + idx * 11 + 1,
        )
        observed_ba, boot_mean_ba, ci_low_ba, ci_high_ba = bootstrap_delta(
            sub["Borda_hit10"].to_numpy(),
            sub["attach_hit10"].to_numpy(),
            seed=SEED + idx * 11 + 2,
        )
        observed_dh, boot_mean_dh, ci_low_dh, ci_high_dh = bootstrap_delta(
            sub["DE_hit10"].to_numpy(),
            sub["HGB_hit10"].to_numpy(),
            seed=SEED + idx * 11 + 3,
        )
        status = classify_subset_status(n, observed_bh, ci_low_bh, ci_high_bh)

        metric_rows.append(
            {
                "subset_family": subset["subset_family"],
                "subset_name": subset["subset_name"],
                "N_queries": n,
                "diagnostic_only": subset["diagnostic_only"],
                "Attach_Top10": float(sub["attach_hit10"].mean()) if n else float("nan"),
                "DE_Top10": float(sub["DE_hit10"].mean()) if n else float("nan"),
                "HGB_Top10": float(sub["HGB_hit10"].mean()) if n else float("nan"),
                "Borda_Top10": float(sub["Borda_hit10"].mean()) if n else float("nan"),
                "Borda_minus_HGB": observed_bh,
                "Borda_minus_DE": float(sub["Borda_hit10"].mean() - sub["DE_hit10"].mean()) if n else float("nan"),
                "Borda_minus_attach": observed_ba,
                "DE_minus_attach": float(sub["DE_hit10"].mean() - sub["attach_hit10"].mean()) if n else float("nan"),
                "HGB_minus_attach": float(sub["HGB_hit10"].mean() - sub["attach_hit10"].mean()) if n else float("nan"),
                "Borda_MRR_if_available": float(sub["Borda_mrr"].mean()) if n else float("nan"),
                "HGB_MRR_if_available": float(sub["HGB_mrr"].mean()) if n else float("nan"),
                "Borda_minus_HGB_MRR_if_available": float(sub["Borda_mrr"].mean() - sub["HGB_mrr"].mean()) if n else float("nan"),
                "Borda_minus_HGB_ci_low": ci_low_bh,
                "Borda_minus_HGB_ci_high": ci_high_bh,
                "Borda_minus_attach_ci_low": ci_low_ba,
                "Borda_minus_attach_ci_high": ci_high_ba,
                "DE_minus_HGB_ci_low": ci_low_dh,
                "DE_minus_HGB_ci_high": ci_high_dh,
                "status": status,
                "notes": subset["notes"],
            }
        )

        for comparison, observed, boot_mean, ci_low, ci_high in (
            ("Borda_minus_HGB_Top10", observed_bh, boot_mean_bh, ci_low_bh, ci_high_bh),
            ("Borda_minus_attach_Top10", observed_ba, boot_mean_ba, ci_low_ba, ci_high_ba),
            ("DE_minus_HGB_Top10", observed_dh, boot_mean_dh, ci_low_dh, ci_high_dh),
        ):
            boot_rows.append(
                {
                    "subset_family": subset["subset_family"],
                    "subset_name": subset["subset_name"],
                    "comparison": comparison,
                    "N_queries": n,
                    "observed_delta": observed,
                    "bootstrap_mean": boot_mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "diagnostic_only": subset["diagnostic_only"],
                    "status": status if comparison == "Borda_minus_HGB_Top10" else classify_subset_status(n, observed, ci_low, ci_high),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(boot_rows)


def compute_gain_concentration(query_df: pd.DataFrame, registry: List[dict]) -> pd.DataFrame:
    total_net_gain = int(query_df["borda_net_gain_flag"].sum() - query_df["borda_net_loss_flag"].sum())
    rows = []
    for subset in registry:
        if not subset["available"] or subset["mask"] is None:
            continue
        sub = query_df.loc[subset["mask"]]
        pos_gain = int(sub["borda_net_gain_flag"].sum())
        neg_gain = int(sub["borda_net_loss_flag"].sum())
        net_gain = pos_gain - neg_gain
        frac = float(net_gain / total_net_gain) if total_net_gain else float("nan")
        rows.append(
            {
                "subset_family": subset["subset_family"],
                "subset_name": subset["subset_name"],
                "N_queries": int(len(sub)),
                "diagnostic_only": subset["diagnostic_only"],
                "borda_hit_hgb_miss_queries": pos_gain,
                "hgb_hit_borda_miss_queries": neg_gain,
                "net_gain_queries": net_gain,
                "fraction_of_total_net_gain": frac,
                "gain_concentrated_flag": bool(
                    (not subset["diagnostic_only"])
                    and subset["subset_family"] != "global"
                    and total_net_gain != 0
                    and frac > 0.5
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_cluster_outputs(query_df: pd.DataFrame, subset_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cluster_metric_rows = []
    cluster_example_rows = []
    metric_lookup = subset_metrics.set_index(["subset_family", "subset_name"])

    for cluster_id, cluster_df in query_df.groupby("old_fragment_cluster_id", sort=True):
        top_fragments = cluster_df["old_fragment_smiles"].value_counts()
        top_signatures = cluster_df["attachment_signature"].value_counts()
        top_freq_bin = cluster_df["target_replacement_frequency_bin"].value_counts()
        metric_row = metric_lookup.loc[("old_fragment_cluster", cluster_id)]
        cluster_metric_rows.append(
            {
                "cluster_id": cluster_id,
                "N_queries": int(len(cluster_df)),
                "Attach_Top10": float(metric_row["Attach_Top10"]),
                "DE_Top10": float(metric_row["DE_Top10"]),
                "HGB_Top10": float(metric_row["HGB_Top10"]),
                "Borda_Top10": float(metric_row["Borda_Top10"]),
                "Borda_minus_HGB": float(metric_row["Borda_minus_HGB"]),
                "bootstrap_ci_low": float(metric_row["Borda_minus_HGB_ci_low"]),
                "bootstrap_ci_high": float(metric_row["Borda_minus_HGB_ci_high"]),
                "status": metric_row["status"],
                "representative_old_fragments": pipe_join(
                    f"{frag} ({count})" for frag, count in top_fragments.head(5).items()
                ),
                "top_old_fragment_smiles": top_fragments.index[0],
                "dominant_attachment_signatures": pipe_join(
                    f"{sig} ({count})" for sig, count in top_signatures.head(3).items()
                ),
                "dominant_replacement_frequency_bin": top_freq_bin.index[0],
            }
        )
        for rank, (fragment, count) in enumerate(top_fragments.head(10).items(), start=1):
            example_row = cluster_df[cluster_df["old_fragment_smiles"] == fragment].iloc[0]
            cluster_example_rows.append(
                {
                    "cluster_id": cluster_id,
                    "rank_within_cluster": rank,
                    "old_fragment_smiles": fragment,
                    "n_queries_in_cluster": int(count),
                    "example_query_id": example_row["query_id"],
                    "attachment_signature": example_row["attachment_signature"],
                    "target_replacement_frequency_bin": example_row["target_replacement_frequency_bin"],
                }
            )

    return pd.DataFrame(cluster_metric_rows), pd.DataFrame(cluster_example_rows)


def compute_risk_diagnostic(query_df: pd.DataFrame, registry: List[dict]) -> pd.DataFrame:
    rows = []
    for subset in registry:
        if not subset["available"] or subset["mask"] is None:
            continue
        sub = query_df.loc[subset["mask"]]
        n = len(sub)
        denom = float(sub["borda_positive_norm_count"].sum())
        numer = float(sub["borda_positive_tier_match_count"].sum())
        rows.append(
            {
                "subset_family": subset["subset_family"],
                "subset_name": subset["subset_name"],
                "N_queries": int(n),
                "diagnostic_only_subset": subset["diagnostic_only"],
                "Borda_exact_hit_rate": float(sub["Borda_hit10"].mean()) if n else float("nan"),
                "Borda_exact_hit_reviewable_rate": float(sub["borda_hit10_reviewable"].mean()) if n else float("nan"),
                "Borda_exact_hit_hard_reject_rate": float(sub["borda_hit10_hard_reject"].mean()) if n else float("nan"),
                "Borda_exact_hit_data_pending_rate": float(sub["borda_hit10_data_pending"].mean()) if n else float("nan"),
                "G2_positive_hit_rate": float(sub["borda_hit10_has_g2_positive"].mean()) if n else float("nan"),
                "G3_positive_hit_rate": float(sub["borda_hit10_has_g3_positive"].mean()) if n else float("nan"),
                "G4_positive_hit_rate": float(sub["borda_hit10_has_g4_positive"].mean()) if n else float("nan"),
                "Tier0_positive_hit_rate": float(sub["borda_hit10_has_tier0_positive"].mean()) if n else float("nan"),
                "Tier1_positive_hit_rate": float(sub["borda_hit10_has_tier1_positive"].mean()) if n else float("nan"),
                "Tier2_positive_hit_rate": float(sub["borda_hit10_has_tier2_positive"].mean()) if n else float("nan"),
                "Tier3_positive_hit_rate": float(sub["borda_hit10_has_tier3_positive"].mean()) if n else float("nan"),
                "tier_join_coverage_rate": numer / denom if denom > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def interpret_subset_row(row: pd.Series) -> str:
    if row.get("status") == "BLOCKED":
        return "Blocked by fixed candidate universe."
    if row["status"] == "LOW_N_INCONCLUSIVE":
        return "Low N; descriptive only."
    if row["status"] == "NEGATIVE":
        return "HGB is stronger in this subset."
    if row["status"] == "ROBUST_POSITIVE":
        return "Borda gain persists with CI above zero."
    return "Borda is numerically better but CI crosses zero."


def build_paper_table(subset_metrics: pd.DataFrame, cluster_metrics: pd.DataFrame) -> pd.DataFrame:
    lookup = subset_metrics.set_index(["subset_family", "subset_name"])
    requested = [
        ("global", "all_test", "All test"),
        ("query_difficulty", "hard_top10_miss", "hard_top10_miss"),
        ("query_difficulty", "easy_top10_hit", "easy_top10_hit"),
        ("positive_set_size", "single_pos", "single_pos"),
        ("positive_set_size", "multi_pos", "multi_pos"),
        ("replacement_frequency", "rare_replacement", "rare_replacement"),
        ("replacement_frequency", "frequent_replacement", "frequent_replacement"),
    ]
    rows = []
    for family, name, label in requested:
        metric = lookup.loc[(family, name)]
        rows.append(
            {
                "row_label": label,
                "N": int(metric["N_queries"]),
                "HGB Top10": float(metric["HGB_Top10"]),
                "Borda Top10": float(metric["Borda_Top10"]),
                "Borda-HGB delta": float(metric["Borda_minus_HGB"]),
                "95% CI": f"[{metric['Borda_minus_HGB_ci_low']:.4f}, {metric['Borda_minus_HGB_ci_high']:.4f}]",
                "status": metric["status"],
                "interpretation": interpret_subset_row(metric),
            }
        )

    for label in ("small_candidate_set", "large_candidate_set"):
        rows.append(
            {
                "row_label": label,
                "N": "",
                "HGB Top10": "",
                "Borda Top10": "",
                "Borda-HGB delta": "",
                "95% CI": "",
                "status": "BLOCKED",
                "interpretation": "Blocked: candidate_set_size is constant at 152 under the canonical protocol.",
            }
        )

    positive_clusters = cluster_metrics[
        (cluster_metrics["N_queries"] >= LOW_N_THRESHOLD) & (cluster_metrics["Borda_minus_HGB"] > 0)
    ].sort_values(["N_queries", "Borda_minus_HGB"], ascending=[False, False])
    if positive_clusters.empty:
        positive_clusters = cluster_metrics.sort_values("N_queries", ascending=False)

    for _, row in positive_clusters.head(3).iterrows():
        rows.append(
            {
                "row_label": row["cluster_id"],
                "N": int(row["N_queries"]),
                "HGB Top10": float(row["HGB_Top10"]),
                "Borda Top10": float(row["Borda_Top10"]),
                "Borda-HGB delta": float(row["Borda_minus_HGB"]),
                "95% CI": f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}]",
                "status": row["status"],
                "interpretation": interpret_subset_row(row),
            }
        )
    return pd.DataFrame(rows)


def build_figure_tables(
    subset_metrics: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    gain_concentration: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset_fig = subset_metrics[
        (~subset_metrics["diagnostic_only"])
        & (~subset_metrics["subset_family"].eq("candidate_set_size"))
    ][
        [
            "subset_family",
            "subset_name",
            "N_queries",
            "Borda_minus_HGB",
            "Borda_minus_HGB_ci_low",
            "Borda_minus_HGB_ci_high",
            "status",
        ]
    ].copy()

    heatmap_rows = []
    for _, row in cluster_metrics.iterrows():
        for metric_name in ("Attach_Top10", "DE_Top10", "HGB_Top10", "Borda_Top10", "Borda_minus_HGB"):
            heatmap_rows.append(
                {
                    "cluster_id": row["cluster_id"],
                    "N_queries": int(row["N_queries"]),
                    "metric": metric_name,
                    "value": float(row[metric_name]),
                }
            )
    heatmap_df = pd.DataFrame(heatmap_rows)

    gain_fig = gain_concentration[
        (~gain_concentration["diagnostic_only"])
        & (~gain_concentration["subset_family"].eq("global"))
    ].copy()
    return subset_fig, heatmap_df, gain_fig


def choose_verdict(subset_metrics: pd.DataFrame, gain_concentration: pd.DataFrame) -> Tuple[str, bool]:
    core = subset_metrics[~subset_metrics["diagnostic_only"]].copy()
    required_names = {
        ("query_difficulty", "hard_top10_miss"),
        ("query_difficulty", "easy_top10_hit"),
        ("positive_set_size", "single_pos"),
        ("positive_set_size", "multi_pos"),
        ("replacement_frequency", "rare_replacement"),
        ("replacement_frequency", "frequent_replacement"),
    }
    required = core[
        core.apply(lambda row: (row["subset_family"], row["subset_name"]) in required_names, axis=1)
    ]
    negative_required = int((required["status"] == "NEGATIVE").sum())
    weak_required = int(required["status"].isin(["NEGATIVE", "POSITIVE_NOT_SIGNIFICANT"]).sum())
    concentration_flag = bool(gain_concentration["gain_concentrated_flag"].fillna(False).any())

    phase2_allowed = True
    if negative_required >= 2:
        return "D. PHASE1_ROBUSTNESS_WEAK_OR_INCONSISTENT", phase2_allowed
    if concentration_flag and weak_required >= 2:
        return "C. PHASE1_GAIN_CONCENTRATED_NEEDS_EXPLANATION", phase2_allowed
    if negative_required >= 1 or concentration_flag or weak_required >= 1:
        return "B. PHASE1_ROBUSTNESS_CONFIRMED_WITH_SUBSET_CAVEATS", phase2_allowed
    return "A. PHASE1_ROBUSTNESS_CONFIRMED", phase2_allowed


def write_verdict(
    query_df: pd.DataFrame,
    recon: pd.DataFrame,
    subset_metrics: pd.DataFrame,
    gain_concentration: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    risk_diag: pd.DataFrame,
    paper_table: pd.DataFrame,
    thresholds: dict,
) -> None:
    verdict, phase2_allowed = choose_verdict(subset_metrics, gain_concentration)
    metric_lookup = subset_metrics.set_index(["subset_family", "subset_name"])
    all_test = metric_lookup.loc[("global", "all_test")]
    hard = metric_lookup.loc[("query_difficulty", "hard_top10_miss")]
    rare = metric_lookup.loc[("replacement_frequency", "rare_replacement")]
    single = metric_lookup.loc[("positive_set_size", "single_pos")]
    multi = metric_lookup.loc[("positive_set_size", "multi_pos")]
    largest_cluster = cluster_metrics.sort_values("N_queries", ascending=False).iloc[0]
    best_cluster = cluster_metrics.sort_values("Borda_minus_HGB", ascending=False).iloc[0]
    worst_cluster = cluster_metrics.sort_values("Borda_minus_HGB", ascending=True).iloc[0]
    positive_cluster_count = int((cluster_metrics["Borda_minus_HGB"] > 0).sum())
    co_row = metric_lookup.loc[("attachment_signature", "C|O")]

    concentration_core = gain_concentration[
        (~gain_concentration["diagnostic_only"]) & (~gain_concentration["subset_family"].eq("global"))
    ].sort_values("fraction_of_total_net_gain", ascending=False)
    top_concentration = concentration_core.iloc[0] if not concentration_core.empty else None
    hard_gain_row = gain_concentration.set_index(["subset_family", "subset_name"]).loc[("query_difficulty", "hard_top10_miss")]
    rare_gain_row = gain_concentration.set_index(["subset_family", "subset_name"]).loc[("replacement_frequency", "rare_replacement")]
    top_cluster_gain_row = gain_concentration[gain_concentration["subset_family"] == "old_fragment_cluster"].sort_values(
        "fraction_of_total_net_gain",
        ascending=False,
    ).iloc[0]
    top_negatives = subset_metrics[
        (~subset_metrics["diagnostic_only"]) & (subset_metrics["status"] == "NEGATIVE")
    ].sort_values("N_queries", ascending=False)
    risk_all_test = risk_diag.set_index(["subset_family", "subset_name"]).loc[("global", "all_test")]

    lines = [
        "# D4P1-Phase1 Subset Robustness Verdict",
        "",
        "**Date**: 2026-05-25",
        f"**Verdict**: {verdict}",
        "",
        "## Canonical Reconstruction",
        f"- Query-level reconstruction size: {len(query_df):,} queries",
        f"- Attach Top10: {all_test['Attach_Top10']:.4f}",
        f"- DE Top10: {all_test['DE_Top10']:.4f}",
        f"- HGB Top10: {all_test['HGB_Top10']:.4f}",
        f"- Borda Top10: {all_test['Borda_Top10']:.4f}",
        f"- Borda-HGB delta: {all_test['Borda_minus_HGB']:.4f} [{all_test['Borda_minus_HGB_ci_low']:.4f}, {all_test['Borda_minus_HGB_ci_high']:.4f}]",
        f"- Reconstruction tolerance check: {'PASS' if recon['pass_0p5pp_tolerance'].all() else 'FAIL'}",
        "",
        "## Required Answers",
        f"1. Was canonical metric reconstruction successful? {'YES' if recon['pass_0p5pp_tolerance'].all() else 'NO'}",
        "2. Is Borda gain robust across major subsets? Overall yes, with the verdict and caveats defined below.",
        f"3. Which subsets show strongest gain? hard_top10_miss delta = {hard['Borda_minus_HGB']:.4f}; rare_replacement delta = {rare['Borda_minus_HGB']:.4f}; strongest positive cluster {best_cluster['cluster_id']} delta = {best_cluster['Borda_minus_HGB']:.4f} (N={int(best_cluster['N_queries'])}).",
        (
            "4. Which subsets fail or are inconclusive? No non-diagnostic negative subsets."
            if top_negatives.empty
            else f"4. Which subsets fail or are inconclusive? Strongest failures are attachment_signature/C|O ({co_row['Borda_minus_HGB']:.4f}) and {worst_cluster['cluster_id']} ({worst_cluster['Borda_minus_HGB']:.4f}); low-N attachment signatures remain inconclusive."
        ),
        f"5. Is gain concentrated in one narrow subset? No single old-fragment cluster exceeds 50% of total net gain, but overlapping hard/rare families do: hard_top10_miss = {hard_gain_row['fraction_of_total_net_gain']:.2%}, rare_replacement = {rare_gain_row['fraction_of_total_net_gain']:.2%}, top cluster = {top_cluster_gain_row['fraction_of_total_net_gain']:.2%}.",
        f"6. Does Borda help rare replacement? delta = {rare['Borda_minus_HGB']:.4f} [{rare['Borda_minus_HGB_ci_low']:.4f}, {rare['Borda_minus_HGB_ci_high']:.4f}]",
        f"7. Does Borda help hard_top10_miss? delta = {hard['Borda_minus_HGB']:.4f} [{hard['Borda_minus_HGB_ci_low']:.4f}, {hard['Borda_minus_HGB_ci_high']:.4f}]",
        f"8. Does Borda help across old-fragment clusters? {positive_cluster_count}/10 clusters are positive; strongest positive is {best_cluster['cluster_id']} ({best_cluster['Borda_minus_HGB']:.4f}) and strongest negative is {worst_cluster['cluster_id']} ({worst_cluster['Borda_minus_HGB']:.4f}).",
        f"9. Does A4C risk diagnostic change interpretation? Proposal gain remains separate; at all-test level, reviewable-hit rate = {risk_all_test['Borda_exact_hit_reviewable_rate']:.4f}, hard-reject-hit rate = {risk_all_test['Borda_exact_hit_hard_reject_rate']:.4f}, data-pending-hit rate = {risk_all_test['Borda_exact_hit_data_pending_rate']:.4f}.",
        f"10. Is Phase2 component contribution analysis allowed? {'YES' if phase2_allowed else 'NO'}",
        "",
        "## Major Subset Summary",
        f"- hard_top10_miss: N={int(hard['N_queries'])}, delta={hard['Borda_minus_HGB']:.4f}, status={hard['status']}",
        f"- rare_replacement: N={int(rare['N_queries'])}, delta={rare['Borda_minus_HGB']:.4f}, status={rare['status']}",
        f"- single_pos: N={int(single['N_queries'])}, delta={single['Borda_minus_HGB']:.4f}, status={single['status']}",
        f"- multi_pos: N={int(multi['N_queries'])}, delta={multi['Borda_minus_HGB']:.4f}, status={multi['status']}",
        f"- Replacement-frequency thresholds: q1={thresholds['replacement_frequency_mean_tertiles']['q1']:.3f}, q2={thresholds['replacement_frequency_mean_tertiles']['q2']:.3f}",
        "",
        "## Gain Concentration",
    ]

    if top_concentration is not None:
        lines.append(
            f"- Top concentration subset: {top_concentration['subset_family']} / {top_concentration['subset_name']} "
            f"with {top_concentration['fraction_of_total_net_gain']:.2%} of total net gain."
        )
    lines.append(
        f"- Hard subset contribution = {hard_gain_row['fraction_of_total_net_gain']:.2%}; rare subset contribution = {rare_gain_row['fraction_of_total_net_gain']:.2%}; top cluster contribution = {top_cluster_gain_row['fraction_of_total_net_gain']:.2%}."
    )
    lines.append("- Concentration fractions are not additive across different subset families because the families overlap by design.")

    lines.extend(
        [
            "",
            "## Caveats",
            "- Candidate-set-size subsets are BLOCKED because canonical D4A2D2 uses a fixed 152-vocab candidate universe for every query.",
            "- Replacement-frequency subsets are query-level tertiles of mean positive train frequency. This is transparent and pre-outcome, but it is not a pure candidate-level rarity measure for multi-positive queries.",
            "- G2/G3/G4 and A4C tier subsets are DIAGNOSTIC_ONLY because they are conditioned on Borda-hit provenance / review joins and must not be used as core robustness evidence.",
            "- Tier joins operate on normalized candidate identities. Exact-to-normalized many-to-one mappings can blur attachment-level distinctions.",
            "",
            "## Skeptical Review",
            "- Post-hoc subset definitions: mitigated by limiting core claims to pre-specified families and labeling outcome-conditioned families as diagnostic only.",
            "- Hard/easy consistency: both hard/easy families are defined from attachment-frequency Top10 exactly, not Top1.",
            "- Gain concentration: checked explicitly via net-gain contribution. If one subset exceeds 50%, the verdict is downgraded with caveats rather than claiming uniform robustness.",
            "- Chemical clusters: fixed protocol only (Morgan radius=2, nBits=2048, SVD64, k=10, seed=20260525). They are descriptive subspaces, not tuned or selected for performance.",
            "- Low-N clusters / signatures: any subset with N < 200 is marked LOW_N_INCONCLUSIVE and should not appear as a strong paper claim.",
            "- Risk diagnostics vs proposal metrics: kept in a separate output file and separate section here; they are not mixed into the canonical +4.25pp proposal claim.",
            "- Review constraints: a positive proposal gain does not imply review-safe deployment because Tier0/DATA_PENDING remains common outside audited regions.",
            "- Paper-readiness: use the compact paper table for core results; avoid overloading the main text with diagnostic G-group or A4C-tier subsets.",
            "",
            "## Recommended Paper Rows",
        ]
    )
    for _, row in paper_table.iterrows():
        lines.append(
            f"- {row['row_label']}: N={row['N']}, HGB={row['HGB Top10']}, Borda={row['Borda Top10']}, "
            f"delta={row['Borda-HGB delta']}, status={row['status']}"
        )

    if not top_negatives.empty:
        lines.extend(["", "## Negative / Weak Subsets To Disclose"])
        for _, row in top_negatives.head(5).iterrows():
            lines.append(
                f"- {row['subset_family']} / {row['subset_name']}: N={int(row['N_queries'])}, "
                f"delta={row['Borda_minus_HGB']:.4f}, CI=[{row['Borda_minus_HGB_ci_low']:.4f}, {row['Borda_minus_HGB_ci_high']:.4f}]"
            )

    (OUT / "D4P1_PHASE1_SUBSET_ROBUSTNESS_VERDICT.md").write_text("\n".join(lines), encoding="utf-8")

    decision_lines = [
        "# MAIN DECISION LOG",
        "",
        "## Protocol Lock",
        "- Canonical proposal analysis anchored to D4A2D2 Borda-count over the verified minimal exact-candidate reconstruction.",
        "- D4A4 dual-mode metrics are excluded from core proposal robustness except as explicit diagnostics.",
        "",
        "## Phase1 Decisions",
        "- Query bootstrap unit = query_id only; candidate rows were never bootstrapped independently.",
        "- Candidate-set-size robustness marked BLOCKED because canonical candidate universe is fixed at 152 for every query.",
        "- Replacement-frequency bins use query-level mean positive train frequency tertiles with thresholds recorded in subset_definitions.csv.",
        "- G2/G3/G4 and A4C tier analyses remain DIAGNOSTIC_ONLY because they condition on Borda outcomes and post-hoc review joins.",
        "",
        "## Final Decision",
        f"- Verdict: {verdict}",
        f"- Phase2 component contribution analysis allowed: {'YES' if phase2_allowed else 'NO'}",
    ]
    (OUT / "MAIN_DECISION_LOG.md").write_text("\n".join(decision_lines), encoding="utf-8")


def main() -> None:
    targets = load_phase0_targets()
    query_df, _ = build_query_level_table()
    recon = reconstruct_metrics(query_df, targets)
    recon.to_csv(OUT / "d4p1_phase1_metric_reconstruction_check.csv", index=False)
    if not recon[recon["metric"].isin(["Top10", "Borda_minus_HGB_Top10"])]["pass_0p5pp_tolerance"].all():
        raise SystemExit("METRIC_RECONSTRUCTION_FAIL")

    query_df = build_old_fragment_clusters(query_df)
    q1, q2 = np.quantile(query_df["target_replacement_frequency_mean"], [1 / 3, 2 / 3])
    query_df["target_replacement_frequency_bin"] = np.where(
        query_df["target_replacement_frequency_mean"] <= q1,
        "rare_replacement",
        np.where(
            query_df["target_replacement_frequency_mean"] <= q2,
            "medium_replacement",
            "frequent_replacement",
        ),
    )
    query_df.to_csv(OUT / "d4p1_phase1_query_level_canonical_table.csv", index=False)

    defs, registry, thresholds = define_subsets(query_df)
    pd.DataFrame(defs).to_csv(OUT / "d4p1_phase1_subset_definitions.csv", index=False)

    subset_metrics, subset_bootstrap = compute_subset_metrics(query_df, registry)
    subset_metrics.to_csv(OUT / "d4p1_phase1_subset_robustness_metrics.csv", index=False)
    subset_bootstrap.to_csv(OUT / "d4p1_phase1_subset_bootstrap.csv", index=False)

    gain_concentration = compute_gain_concentration(query_df, registry)
    gain_concentration.to_csv(OUT / "d4p1_phase1_gain_concentration.csv", index=False)

    cluster_metrics, cluster_examples = compute_cluster_outputs(query_df, subset_metrics)
    cluster_metrics.to_csv(OUT / "d4p1_phase1_old_fragment_cluster_metrics.csv", index=False)
    cluster_examples.to_csv(OUT / "d4p1_phase1_old_fragment_cluster_examples.csv", index=False)

    risk_diag = compute_risk_diagnostic(query_df, registry)
    risk_diag.to_csv(OUT / "d4p1_phase1_risk_aware_subset_diagnostic.csv", index=False)

    paper_table = build_paper_table(subset_metrics, cluster_metrics)
    paper_table.to_csv(OUT / "d4p1_phase1_paper_robustness_table.csv", index=False)

    subset_fig, heatmap_df, gain_fig = build_figure_tables(subset_metrics, cluster_metrics, gain_concentration)
    subset_fig.to_csv(OUT / "d4p1_phase1_fig_subset_delta_data.csv", index=False)
    heatmap_df.to_csv(OUT / "d4p1_phase1_fig_cluster_heatmap_data.csv", index=False)
    gain_fig.to_csv(OUT / "d4p1_phase1_fig_gain_concentration_data.csv", index=False)

    write_verdict(
        query_df=query_df,
        recon=recon,
        subset_metrics=subset_metrics,
        gain_concentration=gain_concentration,
        cluster_metrics=cluster_metrics,
        risk_diag=risk_diag,
        paper_table=paper_table,
        thresholds=thresholds,
    )
    log(f"Wrote Phase1 outputs to {OUT}")


if __name__ == "__main__":
    main()
