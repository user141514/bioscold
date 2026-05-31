#!/usr/bin/env python3
"""Route-A D4S-B0 replay baselines on the secondary blind split."""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

RDLogger.DisableLog("rdApp.*")

SEED = 20260525
TOPR = 50
FP_BITS_HGB = 1024
FP_BITS_DE = 2048
SIG_ORDER = ["C|C", "C|N", "C|O", "C|S", "N|S"]
SIG_TO_IDX = {s: i for i, s in enumerate(SIG_ORDER)}
HIDDEN = 256
OUT_DIM = 128
TEMPS = [0.03, 0.05, 0.07, 0.1]
ALPHAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path("E:/zuhui/bioisosteric_diffusion")
OUT = BASE / "plan_results/routeA_chembl37k_d4s_b0_blind_split_baseline"
MANIFEST = OUT / "d4s_b0_secondary_split_manifest.jsonl"
VOCAB_CSV = OUT / "d4s_b0_train_vocab.csv"
MATRIX_DIR = OUT / "matrices"


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FeatureCache:
    def __init__(self):
        self.fp1024 = {}
        self.fp2048 = {}
        self.ha = {}

    def _mol(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        stripped = smiles.replace("*", "")
        return Chem.MolFromSmiles(stripped)

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

    def heavy_atoms(self, smiles: str) -> int:
        if smiles not in self.ha:
            mol = self._mol(smiles)
            self.ha[smiles] = 0 if mol is None else int(mol.GetNumHeavyAtoms())
        return self.ha[smiles]


CACHE = FeatureCache()


class DualEncoder(nn.Module):
    def __init__(self, qdim=FP_BITS_DE + len(SIG_ORDER), cdim=FP_BITS_DE, hidden=HIDDEN, out_dim=OUT_DIM, temp=0.07):
        super().__init__()
        self.q_enc = nn.Sequential(nn.Linear(qdim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.c_enc = nn.Sequential(nn.Linear(cdim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.log_t = nn.Parameter(torch.tensor(np.log(temp), dtype=torch.float32))

    @property
    def temp(self):
        return torch.exp(self.log_t)

    def forward(self, q, c):
        return (self.q_enc(q) @ self.c_enc(c).T) / self.temp


def mp_softmax_loss(scores, pos_mask):
    pos_lse = torch.logsumexp(scores.masked_fill(~pos_mask, -1e9), dim=1)
    return (torch.logsumexp(scores, dim=1) - pos_lse).mean()


def parse_manifest():
    rows = list(load_jsonl(MANIFEST))
    queries = {}
    for row in rows:
        row["positive_replacement_set"] = set(row["positive_replacement_set"])
        row["query_id"] = str(row["query_id"])
        queries[row["query_id"]] = row
    return queries


def load_vocab():
    vocab_df = pd.read_csv(VOCAB_CSV)
    vocab = vocab_df["replacement_smiles"].astype(str).tolist()
    freqs = vocab_df["global_train_frequency"].astype(int).to_numpy()
    return vocab_df, vocab, freqs


def shard_paths(split_name: str):
    return sorted((MATRIX_DIR / split_name).glob(f"{split_name}_features_shard_*.jsonl"))


def query_feature_vector(old_smiles: str, attachment_signature: str) -> np.ndarray | None:
    if attachment_signature not in SIG_TO_IDX:
        return None
    old_fp = CACHE.fp(old_smiles, FP_BITS_DE)
    onehot = np.zeros(len(SIG_ORDER), dtype=np.float32)
    onehot[SIG_TO_IDX[attachment_signature]] = 1.0
    return np.concatenate([old_fp, onehot], axis=0)


def batch_hgb_features(rows, queries):
    X = np.zeros((len(rows), 7), dtype=np.float32)
    meta = []
    for i, row in enumerate(rows):
        qinfo = queries[row["query_id"]]
        old_smiles = qinfo["old_fragment_smiles"]
        cand = row["candidate"]
        old_fp = CACHE.fp(old_smiles, FP_BITS_HGB)
        cand_fp = CACHE.fp(cand, FP_BITS_HGB)
        inter = float(np.dot(old_fp, cand_fp))
        denom = float(old_fp.sum() + cand_fp.sum() - inter + 1e-10)
        tanimoto = inter / denom if denom > 0 else 0.0
        old_ha = CACHE.heavy_atoms(old_smiles)
        cand_ha = CACHE.heavy_atoms(cand)
        X[i] = [
            np.log1p(row["global_freq"]),
            np.log1p(row["attach_freq"]),
            tanimoto,
            abs(cand_ha - old_ha),
            cand_ha,
            old_ha,
            0.0,
        ]
        meta.append((row["query_id"], cand, int(row["label"]), int(row["attach_freq"]), int(row["global_freq"])))
    return X, meta


def reservoir_sample_train(paths, sample_size=200000):
    sample = []
    total = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                total += 1
                if len(sample) < sample_size:
                    sample.append(row)
                else:
                    j = np.random.randint(0, total)
                    if j < sample_size:
                        sample[j] = row
    return sample, total


def compute_topk_metrics(ranked_by_query, positives_by_query):
    ks = [1, 5, 10, 20, 50]
    hits = {k: [] for k in ks}
    mrr = []
    best_rank = {}
    for qid, ranked in ranked_by_query.items():
        positives = positives_by_query[qid]
        rank = math.inf
        for idx, (cand, _score, _label) in enumerate(ranked, start=1):
            if cand in positives:
                rank = float(idx)
                break
        best_rank[qid] = rank
        for k in ks:
            hits[k].append(int(rank <= k))
        mrr.append(0.0 if not math.isfinite(rank) else 1.0 / rank)
    return {
        "Top1": float(np.mean(hits[1])),
        "Top5": float(np.mean(hits[5])),
        "Top10": float(np.mean(hits[10])),
        "Top20": float(np.mean(hits[20])),
        "Top50": float(np.mean(hits[50])),
        "MRR": float(np.mean(mrr)),
        "hit_vectors": {k: np.array(v, dtype=np.int8) for k, v in hits.items()},
        "best_rank": best_rank,
    }


def attach_rankings(paths, queries, eval_qids):
    per_q = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                qid = row["query_id"]
                if qid not in eval_qids:
                    continue
                per_q[qid].append((row["candidate"], float(row["attach_freq"]), int(row["label"])))
    ranked = {}
    for qid, rows in per_q.items():
        ranked[qid] = sorted(rows, key=lambda x: (-x[1], x[0]))[:TOPR]
    positives = {qid: queries[qid]["positive_replacement_set"] for qid in eval_qids}
    return ranked, compute_topk_metrics(ranked, positives)


def train_hgb(train_paths, queries):
    sample_rows, total_rows = reservoir_sample_train(train_paths, sample_size=200000)
    X, meta = batch_hgb_features(sample_rows, queries)
    y = np.array([label for *_rest, label, _af, _gf in meta], dtype=np.int32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=100,
        learning_rate=0.1,
        early_stopping=False,
        class_weight="balanced",
        random_state=SEED,
    )
    t0 = time.time()
    model.fit(Xs, y)
    runtime = time.time() - t0
    return model, scaler, {
        "method": "HGB",
        "phase": "train",
        "rows_seen_total": int(total_rows),
        "rows_used": int(len(sample_rows)),
        "positives_used": int(y.sum()),
        "negatives_used": int(len(y) - y.sum()),
        "runtime_sec": round(runtime, 2),
        "config": "HistGradientBoostingClassifier(max_depth=5,max_iter=100,learning_rate=0.1,class_weight=balanced)",
        "status": "PASS",
    }


def score_hgb(paths, queries, eval_qids, model, scaler):
    per_q = defaultdict(list)
    batch_rows = []

    def flush():
        if not batch_rows:
            return
        X, meta = batch_hgb_features(batch_rows, queries)
        scores = model.predict_proba(scaler.transform(X))[:, 1]
        for score, (qid, cand, label, af, gf) in zip(scores, meta):
            per_q[qid].append((cand, float(score), int(label), af, gf))
        batch_rows.clear()

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["query_id"] not in eval_qids:
                    continue
                batch_rows.append(row)
                if len(batch_rows) >= 4096:
                    flush()
    flush()
    ranked = {}
    for qid, rows in per_q.items():
        ranked[qid] = [(cand, score, label) for cand, score, label, _af, _gf in sorted(rows, key=lambda x: (-x[1], x[0]))[:TOPR]]
    positives = {qid: queries[qid]["positive_replacement_set"] for qid in eval_qids}
    return ranked, compute_topk_metrics(ranked, positives)


def build_de_tensors(queries, vocab):
    fps = []
    sigs = []
    pos_lists = []
    qids = []
    vocab_index = {cand: idx for idx, cand in enumerate(vocab)}
    for qid, row in queries.items():
        qvec = query_feature_vector(row["old_fragment_smiles"], row["attachment_signature"])
        if qvec is None:
            continue
        pos_idx = [vocab_index[c] for c in row["positive_replacement_set"] if c in vocab_index]
        if not pos_idx:
            continue
        fps.append(qvec[:FP_BITS_DE])
        sigs.append(qvec[FP_BITS_DE:])
        pos_lists.append(pos_idx)
        qids.append(qid)
    ft = torch.tensor(np.array(fps, dtype=np.float32))
    oh = torch.tensor(np.array(sigs, dtype=np.float32))
    qt = torch.cat([ft, oh], dim=1)
    pm = torch.zeros((qt.shape[0], len(vocab)), dtype=torch.bool)
    for i, idxs in enumerate(pos_lists):
        pm[i, idxs] = True
    return qids, qt, pm


def train_de(train_queries, val_queries, vocab, freq_array):
    cand_fps = np.stack([CACHE.fp(cand, FP_BITS_DE) for cand in vocab]).astype(np.float32)
    cand_t = torch.tensor(cand_fps)
    train_qids, train_q, train_pm = build_de_tensors(train_queries, vocab)
    val_qids, val_q, val_pm = build_de_tensors(val_queries, vocab)
    log_af = torch.tensor(np.log(freq_array + 1.0), dtype=torch.float32)
    best_temp = TEMPS[0]
    best_t10 = -1.0
    for temp in TEMPS:
        model = DualEncoder(temp=temp)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ep in range(3):
            perm = torch.randperm(train_q.shape[0])
            for st in range(0, train_q.shape[0], 256):
                batch = perm[st : st + 256]
                scores = model(train_q[batch], cand_t)
                loss = mp_softmax_loss(scores, train_pm[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            scores = model(val_q, cand_t)
        val_ranked = scores.argsort(dim=1, descending=True)[:, :TOPR].numpy()
        hit10 = np.mean([int(any(int(idx) in set(torch.where(val_pm[i])[0].tolist()) for idx in val_ranked[i, :10])) for i in range(val_ranked.shape[0])])
        if hit10 > best_t10:
            best_t10 = hit10
            best_temp = temp
    model = DualEncoder(temp=best_temp)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_alpha = 0.0
    best_val_t10 = -1.0
    patience = 5
    no_improve = 0
    t0 = time.time()
    for _ep in range(20):
        perm = torch.randperm(train_q.shape[0])
        for st in range(0, train_q.shape[0], 256):
            batch = perm[st : st + 256]
            scores = model(train_q[batch], cand_t)
            loss = mp_softmax_loss(scores, train_pm[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores = model(val_q, cand_t)
        for alpha in ALPHAS:
            fused = scores + alpha * log_af
            ranks = fused.argsort(dim=1, descending=True)[:, :10].numpy()
            t10 = np.mean([int(any(int(idx) in set(torch.where(val_pm[i])[0].tolist()) for idx in ranks[i])) for i in range(ranks.shape[0])])
            if t10 > best_val_t10:
                best_val_t10 = t10
                best_alpha = alpha
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
        no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - t0
    return model, cand_t, log_af, {
        "method": "DE",
        "phase": "train",
        "rows_seen_total": int(train_q.shape[0]),
        "rows_used": int(train_q.shape[0]),
        "positives_used": int(train_pm.sum().item()),
        "negatives_used": int(train_q.shape[0] * len(vocab) - train_pm.sum().item()),
        "runtime_sec": round(runtime, 2),
        "config": f"DualEncoder(temp={best_temp},alpha={best_alpha},hidden={HIDDEN},out_dim={OUT_DIM})",
        "status": "PASS",
        "best_alpha": float(best_alpha),
    }, best_alpha


def score_de(split_queries, vocab, model, cand_t, log_af, alpha):
    qids, q_tensor, pos_mask = build_de_tensors(split_queries, vocab)
    with torch.no_grad():
        scores = model(q_tensor, cand_t) + alpha * log_af
    top_idx = scores.argsort(dim=1, descending=True)[:, :TOPR].numpy()
    ranked = {}
    positives = {}
    for i, qid in enumerate(qids):
        positives[qid] = split_queries[qid]["positive_replacement_set"]
        ranked[qid] = []
        for idx in top_idx[i]:
            cand = vocab[int(idx)]
            label = int(cand in positives[qid])
            ranked[qid].append((cand, float(scores[i, int(idx)].item()), label))
    return ranked, compute_topk_metrics(ranked, positives)


def borda_rankings(de_ranked, hgb_ranked, positives_by_qid):
    ranked = {}
    for qid, de_rows in de_ranked.items():
        hgb_ranks = {cand: idx + 1 for idx, (cand, _score, _label) in enumerate(hgb_ranked[qid])}
        scored = []
        for idx, (cand, _score, label) in enumerate(de_rows, start=1):
            score = (TOPR - idx + 1) + (TOPR - hgb_ranks.get(cand, TOPR + 1) + 1)
            scored.append((cand, float(score), label))
        ranked[qid] = sorted(scored, key=lambda x: (-x[1], x[0]))[:TOPR]
    return ranked, compute_topk_metrics(ranked, positives_by_qid)


def oracle_metrics(de_metrics, hgb_metrics, positives_by_qid):
    ranked = {}
    for qid in positives_by_qid:
        de_best = de_metrics["best_rank"].get(qid, math.inf)
        hgb_best = hgb_metrics["best_rank"].get(qid, math.inf)
        best = min(de_best, hgb_best)
        ranked[qid] = []
        if math.isfinite(best):
            ranked[qid] = [("__oracle__", 1.0 / best, 1)]
    metrics = {
        "Top1": float(np.mean([int(min(de_metrics["best_rank"].get(qid, math.inf), hgb_metrics["best_rank"].get(qid, math.inf)) <= 1) for qid in positives_by_qid])),
        "Top5": float(np.mean([int(min(de_metrics["best_rank"].get(qid, math.inf), hgb_metrics["best_rank"].get(qid, math.inf)) <= 5) for qid in positives_by_qid])),
        "Top10": float(np.mean([int(min(de_metrics["best_rank"].get(qid, math.inf), hgb_metrics["best_rank"].get(qid, math.inf)) <= 10) for qid in positives_by_qid])),
        "Top20": float(np.mean([int(min(de_metrics["best_rank"].get(qid, math.inf), hgb_metrics["best_rank"].get(qid, math.inf)) <= 20) for qid in positives_by_qid])),
        "Top50": float(np.mean([int(min(de_metrics["best_rank"].get(qid, math.inf), hgb_metrics["best_rank"].get(qid, math.inf)) <= 50) for qid in positives_by_qid])),
        "MRR": float(np.mean([0.0 if not math.isfinite(min(de_metrics["best_rank"].get(qid, math.inf), hgb_metrics["best_rank"].get(qid, math.inf))) else 1.0 / min(de_metrics["best_rank"].get(qid, math.inf), hgb_metrics["best_rank"].get(qid, math.inf)) for qid in positives_by_qid])),
    }
    metrics["best_rank"] = {qid: min(de_metrics["best_rank"].get(qid, math.inf), hgb_metrics["best_rank"].get(qid, math.inf)) for qid in positives_by_qid}
    return ranked, metrics


def write_metrics_csv(path: Path, split_name: str, total_queries: int, eval_qids, metrics_map):
    rows = []
    for method_name, metrics in metrics_map.items():
        rows.append(
            {
                "split": split_name,
                "method": method_name,
                "n_queries_total_split": int(total_queries),
                "n_queries_eval_seen_vocab": int(len(eval_qids)),
                "coverage": float(len(eval_qids) / max(total_queries, 1)),
                "Top1": metrics["Top1"],
                "Top5": metrics["Top5"],
                "Top10": metrics["Top10"],
                "Top20": metrics["Top20"],
                "Top50": metrics["Top50"],
                "MRR": metrics["MRR"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_query_level(path: Path, split_name: str, eval_qids, metrics_map):
    rows = []
    for qid in eval_qids:
        row = {"query_id": qid, "split": split_name}
        for method_name, metrics in metrics_map.items():
            best = metrics["best_rank"].get(qid, math.inf)
            prefix = method_name.replace("(", "_").replace(")", "").replace(",", "_").replace(" ", "_")
            row[f"{prefix}_best_rank"] = best if math.isfinite(best) else ""
            row[f"{prefix}_hit10"] = int(best <= 10) if math.isfinite(best) else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_predictions(path: Path, ranked_map, method_name: str):
    rows = []
    for qid, ranked in ranked_map.items():
        for rank, (cand, score, label) in enumerate(ranked, start=1):
            rows.append({"query_id": qid, "method": method_name, "rank": rank, "candidate": cand, "score": score, "is_pos": label})
    write_jsonl(path, rows)


def main():
    queries = parse_manifest()
    vocab_df, vocab, freq_array = load_vocab()
    split_to_queries = defaultdict(dict)
    for qid, row in queries.items():
        split_to_queries[row["split_new"]][qid] = row
    eval_qids = {
        "val": [qid for qid, row in split_to_queries["val"].items() if row["target_any_seen_vocab"]],
        "blind_test": [qid for qid, row in split_to_queries["blind_test"].items() if row["target_any_seen_vocab"]],
    }
    train_paths = shard_paths("train")
    val_paths = shard_paths("val")
    blind_paths = shard_paths("blind_test")

    training_rows = [
        {"method": "Attachment_frequency", "phase": "train", "rows_seen_total": 0, "rows_used": 0, "positives_used": 0, "negatives_used": 0, "runtime_sec": 0.0, "config": "train-only attach_freq counts", "status": "PASS"},
        {"method": "Borda(DE,HGB)", "phase": "train", "rows_seen_total": 0, "rows_used": 0, "positives_used": 0, "negatives_used": 0, "runtime_sec": 0.0, "config": "rank fusion over DE/HGB top50", "status": "PASS"},
        {"method": "Oracle(DE,HGB)", "phase": "train", "rows_seen_total": 0, "rows_used": 0, "positives_used": 0, "negatives_used": 0, "runtime_sec": 0.0, "config": "query-level oracle diagnostic", "status": "PASS"},
    ]

    log("Attachment baseline scoring.")
    attach_val_ranked, attach_val_metrics = attach_rankings(val_paths, queries, set(eval_qids["val"]))
    attach_blind_ranked, attach_blind_metrics = attach_rankings(blind_paths, queries, set(eval_qids["blind_test"]))

    log("Training HGB.")
    hgb_model, hgb_scaler, hgb_train_row = train_hgb(train_paths, queries)
    training_rows.append(hgb_train_row)
    log("Scoring HGB on val/blind.")
    hgb_val_ranked, hgb_val_metrics = score_hgb(val_paths, queries, set(eval_qids["val"]), hgb_model, hgb_scaler)
    hgb_blind_ranked, hgb_blind_metrics = score_hgb(blind_paths, queries, set(eval_qids["blind_test"]), hgb_model, hgb_scaler)

    log("Training DE.")
    de_model, cand_t, log_af, de_train_row, best_alpha = train_de(split_to_queries["train"], split_to_queries["val"], vocab, freq_array.astype(np.float32))
    training_rows.append(de_train_row)
    log("Scoring DE on val/blind.")
    de_val_ranked, de_val_metrics = score_de({qid: split_to_queries["val"][qid] for qid in eval_qids["val"]}, vocab, de_model, cand_t, log_af, best_alpha)
    de_blind_ranked, de_blind_metrics = score_de({qid: split_to_queries["blind_test"][qid] for qid in eval_qids["blind_test"]}, vocab, de_model, cand_t, log_af, best_alpha)

    positives_val = {qid: split_to_queries["val"][qid]["positive_replacement_set"] for qid in eval_qids["val"]}
    positives_blind = {qid: split_to_queries["blind_test"][qid]["positive_replacement_set"] for qid in eval_qids["blind_test"]}
    borda_val_ranked, borda_val_metrics = borda_rankings(de_val_ranked, hgb_val_ranked, positives_val)
    borda_blind_ranked, borda_blind_metrics = borda_rankings(de_blind_ranked, hgb_blind_ranked, positives_blind)
    _oracle_val_ranked, oracle_val_metrics = oracle_metrics(de_val_metrics, hgb_val_metrics, positives_val)
    _oracle_blind_ranked, oracle_blind_metrics = oracle_metrics(de_blind_metrics, hgb_blind_metrics, positives_blind)

    pd.DataFrame(training_rows).to_csv(OUT / "d4s_b0_baseline_replay_training_log.csv", index=False)
    val_metrics_map = {
        "Attachment_frequency": attach_val_metrics,
        "DE": de_val_metrics,
        "HGB": hgb_val_metrics,
        "Borda(DE,HGB)": borda_val_metrics,
        "Oracle(DE,HGB)": oracle_val_metrics,
    }
    blind_metrics_map = {
        "Attachment_frequency": attach_blind_metrics,
        "DE": de_blind_metrics,
        "HGB": hgb_blind_metrics,
        "Borda(DE,HGB)": borda_blind_metrics,
        "Oracle(DE,HGB)": oracle_blind_metrics,
    }
    write_metrics_csv(OUT / "d4s_b0_baseline_replay_metrics_val.csv", "val", len(split_to_queries["val"]), eval_qids["val"], val_metrics_map)
    write_metrics_csv(OUT / "d4s_b0_baseline_replay_metrics_blind.csv", "blind_test", len(split_to_queries["blind_test"]), eval_qids["blind_test"], blind_metrics_map)
    write_query_level(OUT / "d4s_b0_query_level_metrics_val.csv", "val", eval_qids["val"], val_metrics_map)
    write_query_level(OUT / "d4s_b0_query_level_metrics_blind.csv", "blind_test", eval_qids["blind_test"], blind_metrics_map)

    with open(OUT / "d4s_b0_baseline_replay_predictions_val.jsonl", "w", encoding="utf-8") as handle:
        for ranked_map, method_name in [
            (attach_val_ranked, "Attachment_frequency"),
            (de_val_ranked, "DE"),
            (hgb_val_ranked, "HGB"),
            (borda_val_ranked, "Borda(DE,HGB)"),
        ]:
            for qid, ranked in ranked_map.items():
                for rank, (cand, score, label) in enumerate(ranked, start=1):
                    handle.write(json.dumps({"query_id": qid, "method": method_name, "rank": rank, "candidate": cand, "score": score, "is_pos": label}, ensure_ascii=False) + "\n")
    with open(OUT / "d4s_b0_baseline_replay_predictions_blind.jsonl", "w", encoding="utf-8") as handle:
        for ranked_map, method_name in [
            (attach_blind_ranked, "Attachment_frequency"),
            (de_blind_ranked, "DE"),
            (hgb_blind_ranked, "HGB"),
            (borda_blind_ranked, "Borda(DE,HGB)"),
        ]:
            for qid, ranked in ranked_map.items():
                for rank, (cand, score, label) in enumerate(ranked, start=1):
                    handle.write(json.dumps({"query_id": qid, "method": method_name, "rank": rank, "candidate": cand, "score": score, "is_pos": label}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
