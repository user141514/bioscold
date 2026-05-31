#!/usr/bin/env python3
"""Shared utilities for Route-A D4S2 listwise reranker."""

from __future__ import annotations

import csv
import gzip
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from routeA_d4s_b0_replay_baselines import train_de

RDLogger.DisableLog("rdApp.*")

SEED = 20260525
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path("E:/zuhui/bioisosteric_diffusion")
PLAN = BASE / "plan_results"
B0 = PLAN / "routeA_chembl37k_d4s_b0_blind_split_baseline"
OUT = PLAN / "routeA_chembl37k_d4s2_listwise_reranker"
ART = OUT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = B0 / "d4s_b0_secondary_split_manifest.jsonl"
LEAKAGE_AUDIT_PATH = B0 / "d4s_b0_split_leakage_audit.csv"
TRAIN_VOCAB_PATH = B0 / "d4s_b0_train_vocab.csv"
B0_VAL_METRICS_PATH = B0 / "d4s_b0_baseline_replay_metrics_val.csv"
B0_BLIND_METRICS_PATH = B0 / "d4s_b0_baseline_replay_metrics_blind.csv"
B0_VAL_QUERY_PATH = B0 / "d4s_b0_query_level_metrics_val.csv"
B0_BLIND_QUERY_PATH = B0 / "d4s_b0_query_level_metrics_blind.csv"
B0_BLIND_TABLE_PATH = B0 / "d4s_b0_blind_canonical_metric_table.csv"
B0_POOL_PATH = B0 / "d4s_b0_reranker_pool_opportunity.csv"
B0_VERDICT_PATH = B0 / "D4S_B0_BLIND_SPLIT_BASELINE_VERDICT.md"

VOCAB_N = 150
SIG_ORDER = ["C|C", "C|N", "C|O", "C|S", "N|S"]
SIG_TO_IDX = {s: i for i, s in enumerate(SIG_ORDER)}
TOPK_LIST = [1, 5, 10, 20, 50]

MATRIX_DIR = OUT / "matrices"
MATRIX_DIR.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_atts(cell: str | float) -> set[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return set()
    return {part for part in str(cell).split("|") if part}


def normalize_name(name: str) -> str:
    return (
        name.replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


@dataclass
class BaselineArtifacts:
    hgb_model: HistGradientBoostingClassifier
    hgb_scaler: StandardScaler
    de_model: torch.nn.Module
    de_candidate_tensor: torch.Tensor
    de_log_af: torch.Tensor
    de_alpha: float
    vocab: list[str]
    vocab_df: pd.DataFrame
    global_freq: np.ndarray
    candidate_signatures: list[set[str]]
    attach_counts: dict[str, Counter]
    global_counts: Counter


class DescriptorCache:
    def __init__(self):
        self.mol_cache: dict[str, Chem.Mol | None] = {}
        self.fp1024: dict[str, np.ndarray] = {}
        self.fp2048: dict[str, np.ndarray] = {}
        self.props: dict[str, dict[str, float]] = {}

    def _mol(self, smiles: str):
        if smiles not in self.mol_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol = Chem.MolFromSmiles(smiles.replace("*", ""))
            self.mol_cache[smiles] = mol
        return self.mol_cache[smiles]

    def fp(self, smiles: str, bits: int):
        cache = self.fp1024 if bits == 1024 else self.fp2048
        if smiles not in cache:
            mol = self._mol(smiles)
            if mol is None:
                cache[smiles] = np.zeros(bits, dtype=np.float32)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits)
                arr = np.zeros(bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                cache[smiles] = arr
        return cache[smiles]

    def props_for(self, smiles: str):
        if smiles not in self.props:
            mol = self._mol(smiles)
            if mol is None:
                self.props[smiles] = {
                    "heavy_atoms": 0.0,
                    "ring_count": 0.0,
                    "hetero_count": 0.0,
                    "mw": 0.0,
                    "logp": 0.0,
                    "tpsa": 0.0,
                }
            else:
                self.props[smiles] = {
                    "heavy_atoms": float(mol.GetNumHeavyAtoms()),
                    "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
                    "hetero_count": float(Lipinski.NumHeteroatoms(mol)),
                    "mw": float(Descriptors.MolWt(mol)),
                    "logp": float(Crippen.MolLogP(mol)),
                    "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                }
        return self.props[smiles]


DESC = DescriptorCache()


def load_manifest() -> pd.DataFrame:
    df = pd.read_json(MANIFEST_PATH, lines=True)
    df["query_id"] = df["query_id"].astype(str)
    df["positive_replacement_set"] = df["positive_replacement_set"].apply(list)
    return df


def load_vocab_df() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_VOCAB_PATH)
    df["replacement_smiles"] = df["replacement_smiles"].astype(str)
    return df


def split_query_dicts(manifest: pd.DataFrame):
    out = defaultdict(dict)
    for row in manifest.itertuples(index=False):
        out[row.split_new][str(row.query_id)] = {
            "query_id": str(row.query_id),
            "split_new": row.split_new,
            "old_fragment_smiles": row.old_fragment_smiles,
            "attachment_signature": row.attachment_signature,
            "positive_replacement_set": set(row.positive_replacement_set),
            "num_positives": int(row.num_positives),
            "target_any_seen_vocab": bool(row.target_any_seen_vocab),
            "target_all_seen_vocab": bool(row.target_all_seen_vocab),
        }
    return out


def preflight_checks() -> tuple[pd.DataFrame, dict]:
    verdict_text = B0_VERDICT_PATH.read_text(encoding="utf-8")
    leakage = pd.read_csv(LEAKAGE_AUDIT_PATH)
    vocab_df = pd.read_csv(TRAIN_VOCAB_PATH)
    blind_table = pd.read_csv(B0_BLIND_TABLE_PATH)
    val_metrics = pd.read_csv(B0_VAL_METRICS_PATH)
    blind_metrics = pd.read_csv(B0_BLIND_METRICS_PATH)
    blind_pool = pd.read_csv(B0_POOL_PATH)

    def fetch(check_name: str):
        sub = leakage.loc[leakage["check"] == check_name]
        return float(sub["value"].iloc[0]) if not sub.empty else np.nan

    blind_borda = float(blind_metrics.loc[blind_metrics["method"] == "Borda(DE,HGB)", "Top10"].iloc[0])
    blind_pool_p3 = float(
        blind_pool.loc[
            (blind_pool["split"] == "blind_test")
            & (blind_pool["pool_name"] == "P3_union_DE50_HGB50_attach50"),
            "oracle_possible_hit_rate",
        ].iloc[0]
    )
    rows = [
        {
            "check": "b0_verdict_pass",
            "value": int("A. D4S_B0_PASS_NEW_BLIND_READY_FOR_RERANKER" in verdict_text),
            "status": "PASS" if "A. D4S_B0_PASS_NEW_BLIND_READY_FOR_RERANKER" in verdict_text else "FAIL",
        },
        {
            "check": "old_test_overlap_blind",
            "value": fetch("old_canonical_test_in_new_blind"),
            "status": "PASS" if fetch("old_canonical_test_in_new_blind") == 0 else "FAIL",
        },
        {
            "check": "transform_overlap_train_blind",
            "value": fetch("transform_overlap_train_blind"),
            "status": "PASS" if fetch("transform_overlap_train_blind") == 0 else "FAIL",
        },
        {
            "check": "blind_seen_vocab_rate",
            "value": fetch("blind_target_any_seen_vocab_rate"),
            "status": "PASS" if fetch("blind_target_any_seen_vocab_rate") >= 0.90 else "FAIL",
        },
        {
            "check": "train_vocab_size",
            "value": int(len(vocab_df)),
            "status": "PASS" if 140 <= len(vocab_df) <= 160 else "FAIL",
        },
        {
            "check": "val_metrics_present",
            "value": int(len(val_metrics)),
            "status": "PASS" if len(val_metrics) >= 4 else "FAIL",
        },
        {
            "check": "blind_metrics_present",
            "value": int(len(blind_metrics)),
            "status": "PASS" if len(blind_metrics) >= 5 else "FAIL",
        },
        {
            "check": "blind_borda_top10_expected",
            "value": blind_borda,
            "status": "PASS" if abs(blind_borda - 0.8384) <= 0.01 else "WARN",
        },
        {"check": "blind_p3_pool_expected", "value": blind_pool_p3, "status": "PASS"},
    ]
    report = pd.DataFrame(rows)
    summary = {
        "seed": SEED,
        "b0_verdict": "A. D4S_B0_PASS_NEW_BLIND_READY_FOR_RERANKER"
        if "A. D4S_B0_PASS_NEW_BLIND_READY_FOR_RERANKER" in verdict_text
        else "OTHER",
        "old_test_overlap_blind": int(fetch("old_canonical_test_in_new_blind")),
        "transform_overlap_train_blind": int(fetch("transform_overlap_train_blind")),
        "blind_seen_vocab_rate": float(fetch("blind_target_any_seen_vocab_rate")),
        "train_vocab_size": int(len(vocab_df)),
        "blind_borda_top10": float(blind_borda),
        "blind_oracle_top10": float(
            blind_table.loc[blind_table["method"] == "Oracle(DE,HGB)", "Top10"].iloc[0]
        ),
    }
    return report, summary


def build_frequency_counts(manifest: pd.DataFrame):
    train = manifest.loc[manifest["split_new"] == "train"].copy()
    global_counts: Counter = Counter()
    attach_counts: dict[str, Counter] = defaultdict(Counter)
    for row in train.itertuples(index=False):
        for repl in row.positive_replacement_set:
            global_counts[repl] += 1
            attach_counts[row.attachment_signature][repl] += 1
    return global_counts, attach_counts


def fit_old_fragment_clusterer(train_old_fragments: list[str]):
    uniq = sorted(set(train_old_fragments))
    fps = []
    kept = []
    for smi in uniq:
        fp = DESC.fp(smi, 2048)
        if float(fp.sum()) <= 0:
            continue
        fps.append(fp)
        kept.append(smi)
    X = np.stack(fps)
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    X64 = svd.fit_transform(X)
    kmeans = KMeans(n_clusters=10, random_state=SEED, n_init=20)
    labels = kmeans.fit_predict(X64)
    mapping = {smi: f"cluster_{int(label):02d}" for smi, label in zip(kept, labels)}
    return {"svd": svd, "kmeans": kmeans, "mapping": mapping}


def assign_cluster(smiles: str, clusterer) -> str:
    mapping = clusterer["mapping"]
    if smiles in mapping:
        return mapping[smiles]
    fp = DESC.fp(smiles, 2048).reshape(1, -1)
    x64 = clusterer["svd"].transform(fp)
    label = int(clusterer["kmeans"].predict(x64)[0])
    return f"cluster_{label:02d}"


def compute_query_meta(
    manifest: pd.DataFrame,
    global_counts: Counter,
    clusterer,
    hard_query_csv: Path | None,
):
    train = manifest.loc[manifest["split_new"] == "train"].copy()
    train_scores = []
    for row in train.itertuples(index=False):
        vals = [global_counts.get(cand, 0) for cand in row.positive_replacement_set]
        train_scores.append(float(np.mean(vals)) if vals else 0.0)
    q1, q2 = np.quantile(train_scores, [1 / 3, 2 / 3]).tolist()
    hard_map = {}
    if hard_query_csv is not None and hard_query_csv.exists():
        hard_df = pd.read_csv(hard_query_csv)
        if "Attachment_frequency_hit10" in hard_df.columns:
            hard_map = {
                str(r.query_id): int(r.Attachment_frequency_hit10 == 0)
                for r in hard_df.itertuples(index=False)
            }
    meta_rows = []
    for row in manifest.itertuples(index=False):
        pos_freqs = [global_counts.get(cand, 0) for cand in row.positive_replacement_set]
        mean_freq = float(np.mean(pos_freqs)) if pos_freqs else 0.0
        if mean_freq <= q1:
            freq_bin = "rare_replacement"
        elif mean_freq <= q2:
            freq_bin = "medium_replacement"
        else:
            freq_bin = "frequent_replacement"
        qid = str(row.query_id)
        meta_rows.append(
            {
                "query_id": qid,
                "split": row.split_new,
                "old_fragment_smiles": row.old_fragment_smiles,
                "attachment_signature": row.attachment_signature,
                "num_positives": int(row.num_positives),
                "positive_set_size": int(row.num_positives),
                "single_pos_flag": int(row.num_positives == 1),
                "multi_pos_flag": int(row.num_positives > 1),
                "replacement_frequency_mean": mean_freq,
                "replacement_frequency_bin": freq_bin,
                "hard_top10_miss_flag": int(hard_map.get(qid, 0)),
                "old_fragment_cluster_id": assign_cluster(row.old_fragment_smiles, clusterer),
                "target_any_seen_vocab": int(bool(row.target_any_seen_vocab)),
                "target_all_seen_vocab": int(bool(row.target_all_seen_vocab)),
            }
        )
    meta_df = pd.DataFrame(meta_rows)
    thresholds = {"rare_upper": float(q1), "medium_upper": float(q2)}
    return meta_df, thresholds


def candidate_static_arrays(vocab_df: pd.DataFrame):
    vocab = vocab_df["replacement_smiles"].astype(str).tolist()
    global_freq = vocab_df["global_train_frequency"].astype(np.float32).to_numpy()
    num_att = vocab_df["num_attachments_train"].astype(np.float32).to_numpy()
    candidate_signatures = [parse_atts(cell) for cell in vocab_df["attachment_signatures"]]
    props = [DESC.props_for(cand) for cand in vocab]
    return {
        "vocab": vocab,
        "global_freq": global_freq,
        "num_att": num_att,
        "candidate_signatures": candidate_signatures,
        "cand_heavy_atoms": np.array([p["heavy_atoms"] for p in props], dtype=np.float32),
        "cand_ring_count": np.array([p["ring_count"] for p in props], dtype=np.float32),
        "cand_hetero_count": np.array([p["hetero_count"] for p in props], dtype=np.float32),
        "cand_mw": np.array([p["mw"] for p in props], dtype=np.float32),
        "cand_logp": np.array([p["logp"] for p in props], dtype=np.float32),
        "cand_tpsa": np.array([p["tpsa"] for p in props], dtype=np.float32),
    }


def hgb_feature_rows(
    old_fragment: str,
    candidates: list[str],
    global_freq: np.ndarray,
    attach_freq: np.ndarray,
):
    old_fp = DESC.fp(old_fragment, 1024)
    old_props = DESC.props_for(old_fragment)
    X = np.zeros((len(candidates), 8), dtype=np.float32)
    for i, cand in enumerate(candidates):
        cand_fp = DESC.fp(cand, 1024)
        inter = float(np.dot(old_fp, cand_fp))
        denom = float(old_fp.sum() + cand_fp.sum() - inter + 1e-10)
        tanimoto = inter / denom if denom > 0 else 0.0
        cand_props = DESC.props_for(cand)
        X[i] = [
            float(np.log1p(global_freq[i])),
            float(np.log1p(attach_freq[i])),
            float(tanimoto),
            abs(cand_props["heavy_atoms"] - old_props["heavy_atoms"]),
            cand_props["heavy_atoms"],
            old_props["heavy_atoms"],
            abs(cand_props["ring_count"] - old_props["ring_count"]),
            abs(cand_props["tpsa"] - old_props["tpsa"]),
        ]
    return X


def reservoir_train_rows(
    manifest: pd.DataFrame,
    vocab: list[str],
    global_freq: np.ndarray,
    attach_counts: dict[str, Counter],
    sample_size=300000,
):
    rng = np.random.RandomState(SEED)
    sample = []
    total = 0
    global_map = {cand: float(freq) for cand, freq in zip(vocab, global_freq)}
    for row in manifest.loc[manifest["split_new"] == "train"].itertuples(index=False):
        att_counter = attach_counts.get(row.attachment_signature, Counter())
        compat = [cand for cand in vocab if cand in att_counter]
        positives = set(row.positive_replacement_set)
        for cand in compat:
            total += 1
            item = {
                "old_fragment_smiles": row.old_fragment_smiles,
                "candidate": cand,
                "label": int(cand in positives),
                "global_freq": global_map[cand],
                "attach_freq": float(att_counter.get(cand, 0)),
            }
            if len(sample) < sample_size:
                sample.append(item)
            else:
                j = rng.randint(0, total)
                if j < sample_size:
                    sample[j] = item
    return sample, total


def train_hgb_signal(
    manifest: pd.DataFrame,
    vocab: list[str],
    global_freq: np.ndarray,
    attach_counts: dict[str, Counter],
):
    sample_rows, total_rows = reservoir_train_rows(
        manifest, vocab, global_freq, attach_counts
    )
    X = []
    y = []
    for row in sample_rows:
        feats = hgb_feature_rows(
            row["old_fragment_smiles"],
            [row["candidate"]],
            np.array([row["global_freq"]], dtype=np.float32),
            np.array([row["attach_freq"]], dtype=np.float32),
        )
        X.append(feats[0])
        y.append(row["label"])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=200,
        learning_rate=0.08,
        early_stopping=False,
        class_weight="balanced",
        random_state=SEED,
    )
    t0 = time.time()
    model.fit(Xs, y)
    return model, scaler, {
        "method": "HGB_signal",
        "phase": "train",
        "rows_seen_total": int(total_rows),
        "rows_used": int(len(sample_rows)),
        "positives_used": int(y.sum()),
        "negatives_used": int(len(y) - y.sum()),
        "runtime_sec": round(time.time() - t0, 2),
        "config": "HistGradientBoostingClassifier(max_depth=6,max_iter=200,learning_rate=0.08,class_weight=balanced)",
        "status": "PASS",
    }


def fit_baselines(manifest: pd.DataFrame, vocab_df: pd.DataFrame):
    global_counts, attach_counts = build_frequency_counts(manifest)
    split_queries = split_query_dicts(manifest)
    vocab = vocab_df["replacement_smiles"].astype(str).tolist()
    global_freq = vocab_df["global_train_frequency"].astype(np.float32).to_numpy()
    log("Training D4S2 HGB signal.")
    hgb_model, hgb_scaler, hgb_train_row = train_hgb_signal(
        manifest, vocab, global_freq, attach_counts
    )
    log("Training D4S2 DE signal.")
    de_model, cand_t, log_af, de_train_row, de_alpha = train_de(
        split_queries["train"],
        split_queries["val"],
        vocab,
        global_freq.astype(np.float32),
    )
    artifacts = BaselineArtifacts(
        hgb_model=hgb_model,
        hgb_scaler=hgb_scaler,
        de_model=de_model,
        de_candidate_tensor=cand_t,
        de_log_af=log_af,
        de_alpha=de_alpha,
        vocab=vocab,
        vocab_df=vocab_df,
        global_freq=global_freq,
        candidate_signatures=[parse_atts(cell) for cell in vocab_df["attachment_signatures"]],
        attach_counts=attach_counts,
        global_counts=global_counts,
    )
    with open(ART / "baseline_artifacts.pkl", "wb") as handle:
        pickle.dump(artifacts, handle)
    pd.DataFrame([hgb_train_row, de_train_row]).to_csv(
        OUT / "d4s2_baseline_refit_training_log.csv", index=False
    )
    return artifacts


def load_baseline_artifacts() -> BaselineArtifacts:
    with open(ART / "baseline_artifacts.pkl", "rb") as handle:
        return pickle.load(handle)


def score_matrix_to_ranks(scores: np.ndarray, secondary: np.ndarray | None = None):
    n_rows, n_cols = scores.shape
    ranks = np.zeros((n_rows, n_cols), dtype=np.int16)
    base_secondary = np.arange(n_cols) if secondary is None else np.asarray(secondary)
    for i in range(n_rows):
        order = np.lexsort((np.arange(n_cols), base_secondary, -scores[i]))
        ranks[i, order] = np.arange(1, n_cols + 1, dtype=np.int16)
    return ranks


def full_attach_arrays(query_df: pd.DataFrame, artifacts: BaselineArtifacts):
    vocab = artifacts.vocab
    global_freq = artifacts.global_freq
    n_vocab = len(vocab)
    scores = np.zeros((len(query_df), n_vocab), dtype=np.float32)
    for i, row in enumerate(query_df.itertuples(index=False)):
        counter = artifacts.attach_counts.get(row.attachment_signature, Counter())
        scores[i] = np.array([float(counter.get(cand, 0)) for cand in vocab], dtype=np.float32)
    ranks = score_matrix_to_ranks(scores, secondary=-global_freq)
    return scores, ranks


def score_de_full(query_df: pd.DataFrame, artifacts: BaselineArtifacts, batch_size=512):
    q_vectors = []
    for row in query_df.itertuples(index=False):
        onehot = np.zeros(len(SIG_ORDER), dtype=np.float32)
        onehot[SIG_TO_IDX[row.attachment_signature]] = 1.0
        old_fp = DESC.fp(row.old_fragment_smiles, 2048)
        q_vectors.append(np.concatenate([old_fp, onehot], axis=0))
    q_tensor = torch.tensor(np.asarray(q_vectors, dtype=np.float32))
    scores = np.zeros((len(query_df), len(artifacts.vocab)), dtype=np.float32)
    model = artifacts.de_model
    cand_t = artifacts.de_candidate_tensor
    log_af = artifacts.de_log_af
    model.eval()
    with torch.no_grad():
        for start in range(0, q_tensor.shape[0], batch_size):
            end = min(start + batch_size, q_tensor.shape[0])
            chunk = model(q_tensor[start:end], cand_t) + artifacts.de_alpha * log_af
            scores[start:end] = chunk.detach().cpu().numpy().astype(np.float32)
    ranks = score_matrix_to_ranks(scores)
    return scores, ranks


def score_hgb_full(query_df: pd.DataFrame, artifacts: BaselineArtifacts):
    vocab = artifacts.vocab
    n_vocab = len(vocab)
    global_map = {cand: float(freq) for cand, freq in zip(vocab, artifacts.global_freq)}
    scores = np.full((len(query_df), n_vocab), -1.0, dtype=np.float32)
    for i, row in enumerate(query_df.itertuples(index=False)):
        compat = [
            idx
            for idx, sigs in enumerate(artifacts.candidate_signatures)
            if row.attachment_signature in sigs
        ]
        if not compat:
            compat = list(range(n_vocab))
        cand_list = [vocab[idx] for idx in compat]
        attach_arr = np.array(
            [float(artifacts.attach_counts[row.attachment_signature].get(cand, 0)) for cand in cand_list],
            dtype=np.float32,
        )
        global_arr = np.array([global_map[cand] for cand in cand_list], dtype=np.float32)
        X = hgb_feature_rows(row.old_fragment_smiles, cand_list, global_arr, attach_arr)
        probs = artifacts.hgb_model.predict_proba(
            artifacts.hgb_scaler.transform(X)
        )[:, 1]
        scores[i, compat] = probs.astype(np.float32)
    ranks = score_matrix_to_ranks(scores, secondary=-artifacts.global_freq)
    return scores, ranks


def full_borda(de_ranks: np.ndarray, hgb_ranks: np.ndarray):
    n_vocab = de_ranks.shape[1]
    scores = (n_vocab + 1 - de_ranks.astype(np.float32)) + (
        n_vocab + 1 - hgb_ranks.astype(np.float32)
    )
    ranks = score_matrix_to_ranks(scores)
    return scores.astype(np.float32), ranks


def feature_columns_for_sets():
    base_rank = ["rank_attach", "rank_de", "rank_hgb", "rank_borda"]
    base_score = ["score_attach", "score_de", "score_hgb", "score_borda"]
    freq = [
        "replacement_frequency",
        "attachment_frequency",
        "candidate_num_attachments_train",
        "compatibility_flag",
    ]
    chem = [
        "morgan_similarity_old_candidate",
        "cand_heavy_atoms",
        "delta_heavy_atoms",
        "cand_ring_count",
        "delta_ring_count",
        "cand_hetero_count",
        "delta_hetero_count",
        "cand_mw",
        "delta_mw",
        "cand_logp",
        "delta_logp",
        "cand_tpsa",
        "delta_tpsa",
        "sig_C|C",
        "sig_C|N",
        "sig_C|O",
        "sig_C|S",
        "sig_N|S",
    ]
    return {
        "F0_rank_only": base_rank,
        "F1_rank_score": base_rank + base_score,
        "F2_rank_score_frequency": base_rank + base_score + freq,
        "F3_rank_score_frequency_chemistry": base_rank + base_score + freq + chem,
    }


def build_feature_tensor_rows(
    query_df: pd.DataFrame,
    query_meta_df: pd.DataFrame,
    artifacts: BaselineArtifacts,
    de_scores: np.ndarray,
    de_ranks: np.ndarray,
    hgb_scores: np.ndarray,
    hgb_ranks: np.ndarray,
    attach_scores: np.ndarray,
    attach_ranks: np.ndarray,
    split_name: str,
    include_labels: bool,
    csv_prefix: str,
    shard_size_queries=2500,
):
    vocab = artifacts.vocab
    n_queries = len(query_df)
    n_vocab = len(vocab)
    candidate_arrays = candidate_static_arrays(artifacts.vocab_df)
    borda_scores, borda_ranks = full_borda(de_ranks, hgb_ranks)
    feature_sets = feature_columns_for_sets()
    all_numeric_cols = feature_sets["F3_rank_score_frequency_chemistry"]
    feature_dim = len(all_numeric_cols)

    query_meta_df = query_meta_df.set_index("query_id")
    split_dir = MATRIX_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    feat_path = ART / f"d4s2_group_features_{split_name}.npy"
    label_path = ART / f"d4s2_group_labels_{split_name}.npy"
    feat_mem = np.lib.format.open_memmap(
        feat_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_queries, n_vocab, feature_dim),
    )
    label_mem = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n_queries, n_vocab),
    )
    query_id_path = ART / f"d4s2_group_query_ids_{split_name}.json"

    csv_cols = [
        "query_id",
        "split",
        "candidate_id",
        "candidate_smiles",
        "is_positive",
        "positive_set_size",
        "rank_attach",
        "rank_DE",
        "rank_HGB",
        "rank_Borda",
        "score_attach",
        "score_DE",
        "score_HGB",
        "score_Borda",
        "candidate_in_attach_top10",
        "candidate_in_DE_top10",
        "candidate_in_HGB_top10",
        "candidate_in_Borda_top10",
        "replacement_frequency",
        "attachment_frequency",
        "old_fragment_smiles",
        "attachment_signature",
        "old_fragment_cluster_if_available",
        "morgan_similarity_old_candidate_if_available",
        "candidate_heavy_atoms",
        "delta_heavy_atoms",
        "candidate_ring_count",
        "delta_ring_count",
        "candidate_hetero_count",
        "delta_hetero_count",
        "candidate_mw",
        "delta_mw",
        "candidate_logp",
        "delta_logp",
        "candidate_tpsa",
        "delta_tpsa",
        "compatibility_flag",
        "a4c_tier_if_available_diagnostic_only",
    ]

    file_idx = 0
    query_ids = []
    csv_handle = None
    csv_writer = None

    def open_writer():
        nonlocal file_idx, csv_handle, csv_writer
        suffix = (
            f"{csv_prefix}_shard_{file_idx:03d}.csv.gz"
            if split_name == "train"
            else f"{csv_prefix}.csv.gz"
        )
        path = split_dir / suffix
        csv_handle = gzip.open(path, "wt", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_handle, fieldnames=csv_cols)
        csv_writer.writeheader()
        file_idx += 1
        return path

    current_rows = 0
    written_paths = [open_writer()]

    for i, row in enumerate(query_df.itertuples(index=False)):
        qid = str(row.query_id)
        query_ids.append(qid)
        positives = set(row.positive_replacement_set) if include_labels else set()
        meta = query_meta_df.loc[qid]
        old_props = DESC.props_for(row.old_fragment_smiles)
        old_fp = DESC.fp(row.old_fragment_smiles, 1024)
        sig_onehot = np.zeros(len(SIG_ORDER), dtype=np.float32)
        sig_onehot[SIG_TO_IDX[row.attachment_signature]] = 1.0
        numeric = np.zeros((n_vocab, feature_dim), dtype=np.float32)
        labels = np.zeros(n_vocab, dtype=np.uint8)
        for j, cand in enumerate(vocab):
            cand_fp = DESC.fp(cand, 1024)
            inter = float(np.dot(old_fp, cand_fp))
            denom = float(old_fp.sum() + cand_fp.sum() - inter + 1e-10)
            tanimoto = inter / denom if denom > 0 else 0.0
            cand_props = {
                "heavy_atoms": candidate_arrays["cand_heavy_atoms"][j],
                "ring_count": candidate_arrays["cand_ring_count"][j],
                "hetero_count": candidate_arrays["cand_hetero_count"][j],
                "mw": candidate_arrays["cand_mw"][j],
                "logp": candidate_arrays["cand_logp"][j],
                "tpsa": candidate_arrays["cand_tpsa"][j],
            }
            compat = float(row.attachment_signature in artifacts.candidate_signatures[j])
            numeric[j] = np.array(
                [
                    float(attach_ranks[i, j]),
                    float(de_ranks[i, j]),
                    float(hgb_ranks[i, j]),
                    float(borda_ranks[i, j]),
                    float(attach_scores[i, j]),
                    float(de_scores[i, j]),
                    float(hgb_scores[i, j]),
                    float(borda_scores[i, j]),
                    float(candidate_arrays["global_freq"][j]),
                    float(attach_scores[i, j]),
                    float(candidate_arrays["num_att"][j]),
                    compat,
                    float(tanimoto),
                    float(cand_props["heavy_atoms"]),
                    float(cand_props["heavy_atoms"] - old_props["heavy_atoms"]),
                    float(cand_props["ring_count"]),
                    float(cand_props["ring_count"] - old_props["ring_count"]),
                    float(cand_props["hetero_count"]),
                    float(cand_props["hetero_count"] - old_props["hetero_count"]),
                    float(cand_props["mw"]),
                    float(cand_props["mw"] - old_props["mw"]),
                    float(cand_props["logp"]),
                    float(cand_props["logp"] - old_props["logp"]),
                    float(cand_props["tpsa"]),
                    float(cand_props["tpsa"] - old_props["tpsa"]),
                    *sig_onehot.tolist(),
                ],
                dtype=np.float32,
            )
            if include_labels and cand in positives:
                labels[j] = 1
            csv_writer.writerow(
                {
                    "query_id": qid,
                    "split": split_name,
                    "candidate_id": j,
                    "candidate_smiles": cand,
                    "is_positive": int(labels[j]),
                    "positive_set_size": int(meta["positive_set_size"]),
                    "rank_attach": int(attach_ranks[i, j]),
                    "rank_DE": int(de_ranks[i, j]),
                    "rank_HGB": int(hgb_ranks[i, j]),
                    "rank_Borda": int(borda_ranks[i, j]),
                    "score_attach": float(attach_scores[i, j]),
                    "score_DE": float(de_scores[i, j]),
                    "score_HGB": float(hgb_scores[i, j]),
                    "score_Borda": float(borda_scores[i, j]),
                    "candidate_in_attach_top10": int(attach_ranks[i, j] <= 10),
                    "candidate_in_DE_top10": int(de_ranks[i, j] <= 10),
                    "candidate_in_HGB_top10": int(hgb_ranks[i, j] <= 10),
                    "candidate_in_Borda_top10": int(borda_ranks[i, j] <= 10),
                    "replacement_frequency": float(candidate_arrays["global_freq"][j]),
                    "attachment_frequency": float(attach_scores[i, j]),
                    "old_fragment_smiles": row.old_fragment_smiles,
                    "attachment_signature": row.attachment_signature,
                    "old_fragment_cluster_if_available": meta["old_fragment_cluster_id"],
                    "morgan_similarity_old_candidate_if_available": float(tanimoto),
                    "candidate_heavy_atoms": float(cand_props["heavy_atoms"]),
                    "delta_heavy_atoms": float(cand_props["heavy_atoms"] - old_props["heavy_atoms"]),
                    "candidate_ring_count": float(cand_props["ring_count"]),
                    "delta_ring_count": float(cand_props["ring_count"] - old_props["ring_count"]),
                    "candidate_hetero_count": float(cand_props["hetero_count"]),
                    "delta_hetero_count": float(cand_props["hetero_count"] - old_props["hetero_count"]),
                    "candidate_mw": float(cand_props["mw"]),
                    "delta_mw": float(cand_props["mw"] - old_props["mw"]),
                    "candidate_logp": float(cand_props["logp"]),
                    "delta_logp": float(cand_props["logp"] - old_props["logp"]),
                    "candidate_tpsa": float(cand_props["tpsa"]),
                    "delta_tpsa": float(cand_props["tpsa"] - old_props["tpsa"]),
                    "compatibility_flag": int(compat),
                    "a4c_tier_if_available_diagnostic_only": "",
                }
            )
        feat_mem[i] = numeric
        label_mem[i] = labels
        current_rows += 1
        if split_name == "train" and current_rows >= shard_size_queries:
            csv_handle.close()
            written_paths.append(open_writer())
            current_rows = 0
    if csv_handle is not None:
        csv_handle.close()

    with open(query_id_path, "w", encoding="utf-8") as handle:
        json.dump(query_ids, handle, ensure_ascii=False)
    meta_out = split_dir / f"d4s2_query_meta_{split_name}.csv"
    query_meta_df.loc[query_ids].reset_index().to_csv(meta_out, index=False)
    return {
        "feature_path": str(feat_path),
        "label_path": str(label_path),
        "query_id_path": str(query_id_path),
        "csv_paths": [str(p) for p in written_paths],
        "query_meta_path": str(meta_out),
        "n_queries": int(n_queries),
        "n_candidates": int(n_vocab),
        "feature_dim": int(feature_dim),
    }


def write_feature_schema(query_meta_thresholds: dict):
    schema = {
        "seed": SEED,
        "candidate_vocab_size": VOCAB_N,
        "feature_sets": feature_columns_for_sets(),
        "frequency_bin_thresholds": query_meta_thresholds,
        "notes": {
            "A4C": "Diagnostic only. Not included in model feature sets because coverage is incomplete on the blind split.",
            "candidate_pool": "Full 150 train-vocabulary replacements per query. HGB/attachment signals are derived from compatible train-signature support and filled with low/missing defaults elsewhere.",
        },
    }
    dump_json(OUT / "d4s2_reranker_feature_schema.json", schema)
    rows = [
        {
            "feature_set": name,
            "definition": ",".join(cols),
            "n_features": len(cols),
            "uses_a4c": 0,
        }
        for name, cols in feature_columns_for_sets().items()
    ]
    pd.DataFrame(rows).to_csv(OUT / "d4s2_feature_set_definitions.csv", index=False)
    return schema


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor):
    pos_mask = (labels > 0.5).bool()
    pos_lse = torch.logsumexp(scores.masked_fill(~pos_mask, -1e9), dim=1)
    return (torch.logsumexp(scores, dim=1) - pos_lse).mean()


def compute_best_ranks_from_scores(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores, axis=1, kind="mergesort")
    sorted_labels = np.take_along_axis(labels, order, axis=1)
    best = np.full(scores.shape[0], np.inf, dtype=np.float32)
    for i in range(scores.shape[0]):
        pos = np.where(sorted_labels[i] > 0)[0]
        if len(pos):
            best[i] = float(pos[0] + 1)
    return best


def compute_ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k=10):
    order = np.argsort(-scores, axis=1, kind="mergesort")[:, :k]
    gains = np.take_along_axis(labels, order, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * discounts).sum(axis=1)
    ideal = np.sort(labels, axis=1)[:, ::-1][:, :k]
    idcg = (ideal * discounts).sum(axis=1)
    ndcg = np.divide(dcg, np.where(idcg > 0, idcg, 1.0))
    return float(np.mean(ndcg))


def metrics_from_score_matrix(scores: np.ndarray, labels: np.ndarray):
    best = compute_best_ranks_from_scores(scores, labels)
    return {
        "Top1": float(np.mean(best <= 1)),
        "Top5": float(np.mean(best <= 5)),
        "Top10": float(np.mean(best <= 10)),
        "Top20": float(np.mean(best <= 20)),
        "Top50": float(np.mean(best <= 50)),
        "MRR": float(np.mean(np.where(np.isfinite(best), 1.0 / best, 0.0))),
        "NDCG@10": compute_ndcg_at_k(scores, labels, k=10),
        "best_rank": best,
    }


def load_group_arrays(split_name: str):
    X = np.load(ART / f"d4s2_group_features_{split_name}.npy", mmap_mode="r")
    y = np.load(ART / f"d4s2_group_labels_{split_name}.npy", mmap_mode="r")
    with open(ART / f"d4s2_group_query_ids_{split_name}.json", encoding="utf-8") as handle:
        qids = json.load(handle)
    meta = pd.read_csv(MATRIX_DIR / split_name / f"d4s2_query_meta_{split_name}.csv")
    return X, y, qids, meta


def train_standardizer_from_memmap(X: np.ndarray, cols: list[int]):
    flat = X[:, :, cols].reshape(-1, len(cols))
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


class LinearListwise(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class MLPListwise(nn.Module):
    def __init__(self, dim: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict_torch_group_model(
    model: nn.Module,
    X: np.ndarray,
    cols: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    batch_queries=512,
):
    model.eval()
    out = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for st in range(0, X.shape[0], batch_queries):
            xb = torch.tensor(
                ((X[st : st + batch_queries][:, :, cols] - mean) / std).astype(np.float32)
            )
            scores = model(xb).detach().cpu().numpy().astype(np.float32)
            out[st : st + batch_queries] = scores
    return out


def fit_torch_group_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cols: list[int],
    lr=1e-3,
    weight_decay=1e-4,
    epochs=8,
    batch_queries=512,
):
    mean, std = train_standardizer_from_memmap(X_train, cols)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_metrics = None
    patience = 2
    no_improve = 0
    for _epoch in range(epochs):
        perm = np.random.permutation(X_train.shape[0])
        model.train()
        for st in range(0, len(perm), batch_queries):
            idx = perm[st : st + batch_queries]
            xb = torch.tensor(
                ((X_train[idx][:, :, cols] - mean) / std).astype(np.float32)
            )
            yb = torch.tensor(y_train[idx].astype(np.float32))
            scores = model(xb)
            loss = listwise_loss(scores, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        val_scores = predict_torch_group_model(
            model, X_val, cols, mean, std, batch_queries=batch_queries
        )
        metrics = metrics_from_score_matrix(val_scores, np.asarray(y_val))
        if best_metrics is None or metrics["Top10"] > best_metrics["Top10"] + 1e-12 or (
            abs(metrics["Top10"] - best_metrics["Top10"]) <= 1e-12
            and metrics["MRR"] > best_metrics["MRR"]
        ):
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, best_metrics


def fit_histgb_candidate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cols: list[int],
    sample_rows=500000,
):
    flat_x = X_train[:, :, cols].reshape(-1, len(cols))
    flat_y = y_train.reshape(-1)
    rng = np.random.RandomState(SEED)
    if len(flat_y) > sample_rows:
        idx = rng.choice(len(flat_y), size=sample_rows, replace=False)
        flat_x = flat_x[idx]
        flat_y = flat_y[idx]
    model = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=200,
        learning_rate=0.08,
        early_stopping=False,
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(flat_x.astype(np.float32), flat_y.astype(np.int32))
    return model


def predict_histgb_group_model(model, X: np.ndarray, cols: list[int], batch_queries=512):
    out = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
    for st in range(0, X.shape[0], batch_queries):
        flat = X[st : st + batch_queries][:, :, cols].reshape(-1, len(cols))
        probs = model.predict_proba(flat.astype(np.float32))[:, 1].astype(np.float32)
        out[st : st + batch_queries] = probs.reshape(-1, X.shape[1])
    return out


def query_level_from_scores(query_ids: list[str], scores: np.ndarray, labels: np.ndarray, method_name: str):
    best = compute_best_ranks_from_scores(scores, labels)
    rows = []
    prefix = normalize_name(method_name)
    for qid, rank in zip(query_ids, best):
        rows.append(
            {
                "query_id": qid,
                f"{prefix}_best_rank": "" if not math.isfinite(rank) else float(rank),
                f"{prefix}_hit1": int(rank <= 1) if math.isfinite(rank) else 0,
                f"{prefix}_hit5": int(rank <= 5) if math.isfinite(rank) else 0,
                f"{prefix}_hit10": int(rank <= 10) if math.isfinite(rank) else 0,
                f"{prefix}_hit20": int(rank <= 20) if math.isfinite(rank) else 0,
                f"{prefix}_hit50": int(rank <= 50) if math.isfinite(rank) else 0,
                f"{prefix}_mrr": 0.0 if not math.isfinite(rank) else float(1.0 / rank),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_paired(a: np.ndarray, b: np.ndarray, n_boot=1000):
    rng = np.random.RandomState(SEED)
    n = len(a)
    diffs = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = float(a[idx].mean() - b[idx].mean())
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def write_a4c_blocked_note():
    text = """# D4S2 A4C Diagnostic Blocked

A4C / G2-G3-G4 diagnostic joins are not complete enough for a blind-wide metric table.

- Joinable surface exists through `query_id` plus candidate normalization (`candidate` / `candidate_norm`) and `old_fragment`.
- Coverage remains partial and mixed-source.
- `d4a4_candidate_final_tiers.csv` still contains pervasive `A4C_UNKNOWN` / `Tier0_DATA_PENDING`.
- D4S2 therefore does not use A4C for model selection or paper-main claims on the blind split.

Action: report A4C as blocked for this stage rather than emitting an incomplete comparative metric table.
"""
    (OUT / "d4s2_a4c_diagnostic_blocked.md").write_text(text, encoding="utf-8")
