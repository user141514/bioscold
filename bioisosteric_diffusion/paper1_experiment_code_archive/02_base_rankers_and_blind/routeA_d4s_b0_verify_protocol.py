#!/usr/bin/env python3
"""Route-A D4S-B0 verify leakage, lock blind metrics, and compare splits."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

RDLogger.DisableLog("rdApp.*")

SEED = 20260525
TOPR = 50
np.random.seed(SEED)

BASE = Path("E:/zuhui/bioisosteric_diffusion")
PLAN = BASE / "plan_results"
OUT = PLAN / "routeA_chembl37k_d4s_b0_blind_split_baseline"
PHASE0 = PLAN / "routeA_chembl37k_d4p1_phase0_metric_lock/d4p1_phase0_canonical_proposal_table.csv"
PHASE1 = PLAN / "routeA_chembl37k_d4p1_phase1_subset_robustness/d4p1_phase1_query_level_canonical_table.csv"
PHASE2 = PLAN / "routeA_chembl37k_d4p1_phase2_component_contribution/d4p1_phase2_component_contribution_metrics.csv"


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def bootstrap_paired(a: np.ndarray, b: np.ndarray, n_boot=1000):
    rng = np.random.RandomState(SEED)
    n = len(a)
    diffs = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = float(a[idx].mean() - b[idx].mean())
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compute_clusters(old_smiles: list[str]):
    uniq = sorted(set(old_smiles))
    fps = []
    kept = []
    for smi in uniq:
        mol = Chem.MolFromSmiles(smi.replace("*", ""))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        kept.append(smi)
    X = np.stack(fps)
    X64 = TruncatedSVD(n_components=64, random_state=SEED).fit_transform(X)
    kmeans = KMeans(n_clusters=10, random_state=SEED, n_init=20)
    labels = kmeans.fit_predict(X64)
    return {smi: f"cluster_{int(label):02d}" for smi, label in zip(kept, labels)}


def group_predictions(path: Path):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in load_jsonl(path):
        grouped[row["query_id"]][row["method"]].append((int(row["rank"]), row["candidate"], float(row["score"]), int(row["is_pos"])))
    out = defaultdict(dict)
    for qid, methods in grouped.items():
        for method, rows in methods.items():
            out[qid][method] = [(cand, score, label) for _rank, cand, score, label in sorted(rows, key=lambda x: x[0])]
    return out


def pool_table(preds, manifest_df, split_name: str):
    split_df = manifest_df.loc[(manifest_df["split_new"] == split_name) & (manifest_df["target_any_seen_vocab"])].copy()
    positives = {row.query_id: set(row.positive_replacement_set) for row in split_df.itertuples(index=False)}
    rows = []
    for pool_name, builder in [
        ("P0_Borda_top10", lambda q: {cand for cand, _score, _label in preds[q]["Borda(DE,HGB)"][:10]}),
        ("P1_union_DE10_HGB10", lambda q: {cand for cand, _score, _label in preds[q]["DE"][:10]} | {cand for cand, _score, _label in preds[q]["HGB"][:10]}),
        ("P2_union_DE20_HGB20_attach20", lambda q: {cand for cand, _score, _label in preds[q]["DE"][:20]} | {cand for cand, _score, _label in preds[q]["HGB"][:20]} | {cand for cand, _score, _label in preds[q]["Attachment_frequency"][:20]}),
        ("P3_union_DE50_HGB50_attach50", lambda q: {cand for cand, _score, _label in preds[q]["DE"][:50]} | {cand for cand, _score, _label in preds[q]["HGB"][:50]} | {cand for cand, _score, _label in preds[q]["Attachment_frequency"][:50]}),
        ("P4_full_vocabulary", lambda q: set()),
    ]:
        sizes = []
        hits = []
        for qid in split_df["query_id"]:
            if pool_name == "P4_full_vocabulary":
                sizes.append(np.nan)
                hits.append(1)
                continue
            pool = builder(qid)
            sizes.append(len(pool))
            hits.append(int(len(pool & positives[qid]) > 0))
        rows.append(
            {
                "split": split_name,
                "pool_name": pool_name,
                "n_queries": int(len(split_df)),
                "pool_size_mean": float(np.nanmean(sizes)) if pool_name != "P4_full_vocabulary" else np.nan,
                "oracle_possible_hit_rate": float(np.mean(hits)),
            }
        )
    return rows


def main():
    manifest = pd.read_json(OUT / "d4s_b0_secondary_split_manifest.jsonl", lines=True)
    val_metrics = pd.read_csv(OUT / "d4s_b0_baseline_replay_metrics_val.csv")
    blind_metrics = pd.read_csv(OUT / "d4s_b0_baseline_replay_metrics_blind.csv")
    blind_query = pd.read_csv(OUT / "d4s_b0_query_level_metrics_blind.csv")
    train_vocab = pd.read_csv(OUT / "d4s_b0_train_vocab.csv")
    phase0 = pd.read_csv(PHASE0)
    phase1 = pd.read_csv(PHASE1)
    phase2 = pd.read_csv(PHASE2)

    blind_eval_n = int(blind_metrics["n_queries_eval_seen_vocab"].iloc[0])
    hgb_top10 = float(blind_metrics.loc[blind_metrics["method"] == "HGB", "Top10"].iloc[0])
    borda_top10 = float(blind_metrics.loc[blind_metrics["method"] == "Borda(DE,HGB)", "Top10"].iloc[0])
    oracle_top10 = float(blind_metrics.loc[blind_metrics["method"] == "Oracle(DE,HGB)", "Top10"].iloc[0])

    borda_hits = blind_query["Borda_DE_HGB_hit10"].astype(int).to_numpy()
    hgb_hits = blind_query["HGB_hit10"].astype(int).to_numpy()
    oracle_hits = blind_query["Oracle_DE_HGB_hit10"].astype(int).to_numpy()
    b_mean, b_lo, b_hi = bootstrap_paired(borda_hits, hgb_hits)
    o_mean, o_lo, o_hi = bootstrap_paired(oracle_hits, borda_hits)

    table_rows = []
    for row in blind_metrics.itertuples(index=False):
        table_rows.append(
            {
                "method": row.method,
                "N_queries": int(row.n_queries_eval_seen_vocab),
                "candidate_vocab_size": int(len(train_vocab)),
                "Top1": row.Top1,
                "Top5": row.Top5,
                "Top10": row.Top10,
                "Top20": row.Top20,
                "Top50": row.Top50,
                "MRR": row.MRR,
                "Borda_HGB_delta": b_mean if row.method == "Borda(DE,HGB)" else np.nan,
                "Oracle_Borda_gap": o_mean if row.method == "Oracle(DE,HGB)" else np.nan,
                "bootstrap_CI_low": b_lo if row.method == "Borda(DE,HGB)" else (o_lo if row.method == "Oracle(DE,HGB)" else np.nan),
                "bootstrap_CI_high": b_hi if row.method == "Borda(DE,HGB)" else (o_hi if row.method == "Oracle(DE,HGB)" else np.nan),
            }
        )
    pd.DataFrame(table_rows).to_csv(OUT / "d4s_b0_blind_canonical_metric_table.csv", index=False)
    pd.DataFrame(
        [
            {"comparison": "Borda_vs_HGB_Top10", "delta_mean": b_mean, "ci_low": b_lo, "ci_high": b_hi, "n_queries": blind_eval_n},
            {"comparison": "Oracle_vs_Borda_Top10", "delta_mean": o_mean, "ci_low": o_lo, "ci_high": o_hi, "n_queries": blind_eval_n},
        ]
    ).to_csv(OUT / "d4s_b0_blind_bootstrap.csv", index=False)

    old_top10 = {
        "Attachment_frequency": float(phase0.loc[phase0["method"] == "Attachment_frequency", "Top10"].iloc[0]),
        "HGB": float(phase0.loc[phase0["method"] == "HGB", "Top10"].iloc[0]),
        "Borda(DE,HGB)": float(phase0.loc[phase0["method"] == "Borda(DE,HGB)", "Top10"].iloc[0]),
        "Oracle(DE,HGB)": float(phase2.loc[phase2["method_id"] == "M8", "Top10"].iloc[0]),
    }
    all_repl_counts = Counter()
    for repls in manifest["positive_replacement_set"]:
        for repl in repls:
            all_repl_counts[repl] += 1
    def freq_score(repls): return float(np.mean([all_repl_counts[r] for r in repls]))
    manifest["repl_freq_score"] = manifest["positive_replacement_set"].apply(freq_score)
    q1, q2 = manifest["repl_freq_score"].quantile([1 / 3, 2 / 3]).tolist()
    def freq_bin(score):
        if score <= q1:
            return "rare"
        if score <= q2:
            return "medium"
        return "frequent"
    manifest["freq_bin_global"] = manifest["repl_freq_score"].apply(freq_bin)
    old_eval = phase1[["query_id", "old_fragment_smiles", "attachment_signature", "num_positive_replacements_total", "target_replacement_frequency_bin"]].copy()
    old_eval["query_id"] = old_eval["query_id"].astype(str)
    old_eval = old_eval.rename(columns={"target_replacement_frequency_bin": "freq_bin_old"})
    new_eval = manifest.loc[(manifest["split_new"] == "blind_test") & (manifest["target_any_seen_vocab"])].copy()
    cluster_map = compute_clusters(old_eval["old_fragment_smiles"].tolist() + new_eval["old_fragment_smiles"].tolist())
    old_eval["cluster"] = old_eval["old_fragment_smiles"].map(cluster_map)
    new_eval["cluster"] = new_eval["old_fragment_smiles"].map(cluster_map)
    comparison_rows = [
        {"metric_family": "count", "metric_name": "query_count", "old_value": int(len(old_eval)), "new_value": int(len(new_eval)), "delta": int(len(new_eval) - len(old_eval))},
        {"metric_family": "count", "metric_name": "transform_key_count", "old_value": int(load_json(OUT / "d4s_b0_old_test_quarantine_summary.json")["n_old_test_transform_keys"]), "new_value": int(len({k for cell in new_eval["transform_key"] for k in str(cell).split('|') if k})), "delta": np.nan},
    ]
    for sig in sorted(set(old_eval["attachment_signature"]) | set(new_eval["attachment_signature"])):
        comparison_rows.append({"metric_family": "attachment_signature", "metric_name": sig, "old_value": float((old_eval["attachment_signature"] == sig).mean()), "new_value": float((new_eval["attachment_signature"] == sig).mean()), "delta": float((new_eval["attachment_signature"] == sig).mean() - (old_eval["attachment_signature"] == sig).mean())})
    comparison_rows.append({"metric_family": "positive_set_size", "metric_name": "single_pos", "old_value": float((old_eval["num_positive_replacements_total"] == 1).mean()), "new_value": float((new_eval["num_positives"] == 1).mean()), "delta": float((new_eval["num_positives"] == 1).mean() - (old_eval["num_positive_replacements_total"] == 1).mean())})
    comparison_rows.append({"metric_family": "positive_set_size", "metric_name": "multi_pos", "old_value": float((old_eval["num_positive_replacements_total"] > 1).mean()), "new_value": float((new_eval["num_positives"] > 1).mean()), "delta": float((new_eval["num_positives"] > 1).mean() - (old_eval["num_positive_replacements_total"] > 1).mean())})
    for label in ["rare", "medium", "frequent"]:
        comparison_rows.append({"metric_family": "replacement_frequency", "metric_name": label, "old_value": float((old_eval["freq_bin_old"].str.contains(label.split()[0]) if False else (old_eval["freq_bin_old"] == {"rare":"rare_replacement","medium":"medium_replacement","frequent":"frequent_replacement"}[label])).mean()), "new_value": float((new_eval["freq_bin_global"] == label).mean()), "delta": np.nan})
    for cluster in sorted(set(old_eval["cluster"]) | set(new_eval["cluster"])):
        comparison_rows.append({"metric_family": "old_fragment_cluster", "metric_name": cluster, "old_value": float((old_eval["cluster"] == cluster).mean()), "new_value": float((new_eval["cluster"] == cluster).mean()), "delta": float((new_eval["cluster"] == cluster).mean() - (old_eval["cluster"] == cluster).mean())})
    for method in ["Attachment_frequency", "HGB", "Borda(DE,HGB)", "Oracle(DE,HGB)"]:
        comparison_rows.append({"metric_family": "performance", "metric_name": method, "old_value": old_top10[method], "new_value": float(blind_metrics.loc[blind_metrics["method"] == method, "Top10"].iloc[0]), "delta": float(blind_metrics.loc[blind_metrics["method"] == method, "Top10"].iloc[0] - old_top10[method])})
    pd.DataFrame(comparison_rows).to_csv(OUT / "d4s_b0_old_vs_new_split_comparison.csv", index=False)

    preds_val = group_predictions(OUT / "d4s_b0_baseline_replay_predictions_val.jsonl")
    preds_blind = group_predictions(OUT / "d4s_b0_baseline_replay_predictions_blind.jsonl")
    pool_rows = pool_table(preds_val, manifest, "val") + pool_table(preds_blind, manifest, "blind_test")
    pd.DataFrame(pool_rows).to_csv(OUT / "d4s_b0_reranker_pool_opportunity.csv", index=False)


if __name__ == "__main__":
    main()
