#!/usr/bin/env python3
"""Run locked LBC-Ranker external validation on a standardized matrix.

This evaluator intentionally starts from a normalized candidate matrix rather than
raw BindingDB/ChEMBL/SwissBioisostere files. Dataset-specific builders should
write the matrix contract documented in protocol.md, then call this script.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = [
    "query_id",
    "old_fragment_smiles",
    "candidate_smiles",
    "label",
]

FEATURE_COLUMNS = [
    "morgan",
    "bit_corr",
    "dHeavy",
    "dRings",
    "dMW",
    "dLogP",
    "dTPSA",
]

FULLBLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def read_table(path: Path) -> pd.DataFrame:
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported matrix format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def validate_matrix(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")

    out = df.copy()
    out["query_id"] = out["query_id"].astype(str)
    out["old_fragment_smiles"] = out["old_fragment_smiles"].astype(str)
    out["candidate_smiles"] = out["candidate_smiles"].astype(str)
    out["label"] = out["label"].astype(int)

    bad_labels = sorted(set(out["label"]) - {0, 1})
    if bad_labels:
        raise ValueError(f"{name} has non-binary labels: {bad_labels[:10]}")

    if "dataset" not in out.columns:
        out["dataset"] = name

    dup = out.duplicated(["query_id", "candidate_smiles"]).sum()
    if dup:
        raise ValueError(f"{name} has duplicate query/candidate rows: {dup}")

    return out


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(present) == len(FEATURE_COLUMNS):
        out = df.copy()
        out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype(float).fillna(0.0)
        return out
    if present:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        raise ValueError(f"Partial feature matrix supplied. Missing: {missing}")

    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem, Descriptors
    except Exception as exc:  # pragma: no cover - depends on optional RDKit env
        raise ValueError(
            "Feature columns are absent and RDKit could not be imported. "
            "Precompute feature columns or install RDKit."
        ) from exc

    RDLogger.DisableLog("rdApp.*")
    fp_cache: dict[str, np.ndarray | None] = {}
    prop_cache: dict[str, np.ndarray | None] = {}

    def fp(smiles: str) -> np.ndarray | None:
        if smiles in fp_cache:
            return fp_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fp_cache[smiles] = None
            return None
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048), arr)
        fp_cache[smiles] = arr
        return arr

    def props(smiles: str) -> np.ndarray | None:
        if smiles in prop_cache:
            return prop_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            prop_cache[smiles] = None
            return None
        arr = np.array(
            [
                mol.GetNumHeavyAtoms(),
                Descriptors.RingCount(mol),
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
            ],
            dtype=np.float32,
        )
        prop_cache[smiles] = arr
        return arr

    rows = []
    for old_smi, cand_smi in zip(df["old_fragment_smiles"], df["candidate_smiles"]):
        ofp, cfp = fp(old_smi), fp(cand_smi)
        op, cp = props(old_smi), props(cand_smi)
        if ofp is None or cfp is None or op is None or cp is None:
            rows.append([0.0] * len(FEATURE_COLUMNS))
            continue
        inter = float(np.dot(cfp, ofp))
        denom = float(cfp.sum() + ofp.sum() - inter)
        morgan = inter / max(denom, 0.001)
        bit_corr = 0.0
        if cfp.sum() > 0 and ofp.sum() > 0:
            cfp_n = cfp / (cfp.sum() + 1e-10)
            ofp_n = ofp / (ofp.sum() + 1e-10)
            corr = float(np.corrcoef(cfp_n, ofp_n)[0, 1])
            bit_corr = corr if math.isfinite(corr) else 0.0
        rows.append(
            [
                morgan,
                bit_corr,
                abs(float(cp[0] - op[0])),
                abs(float(cp[1] - op[1])),
                abs(float(cp[2] - op[2])) / max(float(op[2]), 1.0),
                abs(float(cp[3] - op[3])) / max(abs(float(op[3])) + 1.0, 1.0),
                abs(float(cp[4] - op[4])) / max(float(op[4]) + 1.0, 1.0),
            ]
        )

    out = df.copy()
    out[FEATURE_COLUMNS] = pd.DataFrame(rows, columns=FEATURE_COLUMNS, index=out.index)
    return out


def hit_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> int:
    if labels.sum() == 0:
        return 0
    order = np.argsort(-scores)
    return int(labels[order[:k]].sum() > 0)


def z_apply(values: pd.Series, train_mean: float, train_std: float) -> pd.Series:
    std = train_std if train_std > 1e-12 else 1.0
    return (values - train_mean) / std


def z_by_query(values: pd.Series, query_ids: pd.Series) -> pd.Series:
    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std(ddof=0))
        if std <= 1e-12:
            return pd.Series(np.zeros(len(group), dtype=np.float32), index=group.index)
        return (group - float(group.mean())) / std

    return values.groupby(query_ids, sort=False).transform(_z)


def query_hit_values(frame: pd.DataFrame, scores: pd.Series, k: int) -> list[int]:
    hits = []
    for _qid, g in frame.groupby("query_id", sort=False):
        labels = np.asarray(g["label"], dtype=np.int8)
        if labels.sum() == 0:
            continue
        hits.append(hit_at_k(labels, np.asarray(scores.loc[g.index], dtype=float), k))
    return hits


def fit_ridge_content(frame: pd.DataFrame) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(frame[FEATURE_COLUMNS])
    model = Ridge(alpha=1.0, random_state=20260609)
    model.fit(x_scaled, np.asarray(frame["label"], dtype=float))
    return scaler, model


def predict_ridge_content(frame: pd.DataFrame, scaler: StandardScaler, model: Ridge) -> pd.Series:
    scores = model.predict(scaler.transform(frame[FEATURE_COLUMNS]))
    return pd.Series(scores, index=frame.index, dtype=float)


def frequency_scores_from_train(train_df: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    train_pos = train_df[train_df["label"] == 1]
    freq_counts = train_pos.groupby("candidate_smiles").size().to_dict()
    max_freq = max(freq_counts.values()) if freq_counts else 1
    return frame["candidate_smiles"].map(freq_counts).fillna(0.0).astype(float) / max_freq


def choose_fullblend_alpha(train_df: pd.DataFrame, k: int) -> float:
    ofs = sorted(train_df.loc[train_df["label"] == 1, "old_fragment_smiles"].unique())
    if len(ofs) < 3:
        scaler, model = fit_ridge_content(train_df)
        content = predict_ridge_content(train_df, scaler, model)
        freq = frequency_scores_from_train(train_df, train_df)
        scores_by_alpha = {}
        for alpha in FULLBLEND_ALPHAS:
            blended = alpha * z_by_query(freq, train_df["query_id"]) + (1.0 - alpha) * z_by_query(
                content, train_df["query_id"]
            )
            hits = query_hit_values(train_df, blended, k)
            scores_by_alpha[alpha] = float(np.mean(hits)) if hits else 0.0
        return max(FULLBLEND_ALPHAS, key=lambda alpha: (scores_by_alpha[alpha], -alpha))

    rng = np.random.RandomState(20260609)
    shuffled = ofs[:]
    rng.shuffle(shuffled)
    folds = np.array_split(np.asarray(shuffled, dtype=object), min(3, len(shuffled)))
    scores_by_alpha: dict[float, list[int]] = {alpha: [] for alpha in FULLBLEND_ALPHAS}

    for fold in folds:
        val_ofs = set(str(x) for x in fold)
        inner_train = train_df[~train_df["old_fragment_smiles"].isin(val_ofs)].copy()
        inner_val = train_df[train_df["old_fragment_smiles"].isin(val_ofs)].copy()
        if inner_train.empty or inner_val.empty:
            continue
        scaler, model = fit_ridge_content(inner_train)
        content = predict_ridge_content(inner_val, scaler, model)
        freq = frequency_scores_from_train(inner_train, inner_val)
        z_content = z_by_query(content, inner_val["query_id"])
        z_freq = z_by_query(freq, inner_val["query_id"])
        for alpha in FULLBLEND_ALPHAS:
            hits = query_hit_values(inner_val, alpha * z_freq + (1.0 - alpha) * z_content, k)
            scores_by_alpha[alpha].extend(hits)

    return max(
        FULLBLEND_ALPHAS,
        key=lambda alpha: (float(np.mean(scores_by_alpha[alpha])) if scores_by_alpha[alpha] else 0.0, -alpha),
    )


def add_scores(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    train_pos = train_df[train_df["label"] == 1]
    freq_counts = train_pos.groupby("candidate_smiles").size().to_dict()
    max_freq = max(freq_counts.values()) if freq_counts else 1

    def freq_score(frame: pd.DataFrame) -> pd.Series:
        return frame["candidate_smiles"].map(freq_counts).fillna(0.0) / max_freq

    train_freq = freq_score(train_df)
    test_freq = freq_score(test_df)

    train_ca = 0.75 * train_df["morgan"] + 0.25 * (
        1.0 / (1.0 + train_df["dHeavy"] + train_df["dRings"] + train_df["dMW"] + train_df["dLogP"])
    )
    test_ca = 0.75 * test_df["morgan"] + 0.25 * (
        1.0 / (1.0 + test_df["dHeavy"] + test_df["dRings"] + test_df["dMW"] + test_df["dLogP"])
    )

    train_x = train_df[FEATURE_COLUMNS].copy()
    train_x["freq"] = np.asarray(train_freq)
    test_x = test_df[FEATURE_COLUMNS].copy()
    test_x["freq"] = np.asarray(test_freq)

    if train_df["label"].nunique() < 2:
        raise ValueError("Training split has only one label class; cannot fit LBC.")

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(train_x)
    x_test_s = scaler.transform(test_x)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=20260609)
    clf.fit(x_train_s, np.asarray(train_df["label"]))

    scaler_nofreq = StandardScaler()
    x_train_nofreq_s = scaler_nofreq.fit_transform(train_df[FEATURE_COLUMNS])
    x_test_nofreq_s = scaler_nofreq.transform(test_df[FEATURE_COLUMNS])
    clf_nofreq = LogisticRegression(C=1.0, max_iter=2000, random_state=20260609)
    clf_nofreq.fit(x_train_nofreq_s, np.asarray(train_df["label"]))

    fullblend_alpha = choose_fullblend_alpha(train_df, k)
    content_scaler, content_model = fit_ridge_content(train_df)
    test_fullcontent = predict_ridge_content(test_df, content_scaler, content_model)
    test_fullblend = fullblend_alpha * z_by_query(test_freq, test_df["query_id"]) + (
        1.0 - fullblend_alpha
    ) * z_by_query(test_fullcontent, test_df["query_id"])

    out = test_df.copy()
    out["score_freq"] = np.asarray(test_freq)
    out["score_ca"] = np.asarray(test_ca)
    out["score_blend"] = (
        0.25 * z_apply(test_freq, float(train_freq.mean()), float(train_freq.std(ddof=0)))
        + 0.75 * z_apply(test_ca, float(train_ca.mean()), float(train_ca.std(ddof=0)))
    )
    out["score_fullcontent"] = np.asarray(test_fullcontent)
    out["score_fullblend"] = np.asarray(test_fullblend)
    out["score_lbc"] = clf.predict_proba(x_test_s)[:, 1]
    out["score_lbc_nofreq"] = clf_nofreq.predict_proba(x_test_nofreq_s)[:, 1]
    out["fullblend_alpha"] = float(fullblend_alpha)
    return out


def summarize_scored(scored: pd.DataFrame, k: int, seed: int | str) -> tuple[dict, pd.DataFrame]:
    detail_rows = []
    methods = {
        "freq": "score_freq",
        "ca": "score_ca",
        "blend": "score_blend",
        "fullblend": "score_fullblend",
        "lbc": "score_lbc",
        "lbc_nofreq": "score_lbc_nofreq",
    }
    for qid, g in scored.groupby("query_id", sort=False):
        labels = np.asarray(g["label"], dtype=np.int8)
        if labels.sum() == 0:
            continue
        row = {
            "seed": seed,
            "query_id": qid,
            "old_fragment_smiles": g["old_fragment_smiles"].iloc[0],
            "n_candidates": len(g),
            "n_positive": int(labels.sum()),
        }
        for method, col in methods.items():
            row[f"{method}_hit@{k}"] = hit_at_k(labels, np.asarray(g[col], dtype=float), k)
        detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise ValueError("No positive-label queries in evaluation split.")

    summary = {"seed": seed, "n_eval_queries": int(len(detail)), "n_eval_of": int(detail["old_fragment_smiles"].nunique())}
    if "fullblend_alpha" in scored.columns:
        summary["fullblend_alpha"] = float(scored["fullblend_alpha"].iloc[0])
    for method in methods:
        col = f"{method}_hit@{k}"
        query_score = float(detail[col].mean())
        of_score = float(detail.groupby("old_fragment_smiles")[col].mean().mean())
        summary[f"{method}_query_hit@{k}"] = query_score
        summary[f"{method}_of_macro_hit@{k}"] = of_score
    summary[f"delta_lbc_vs_blend_of_macro@{k}"] = summary[f"lbc_of_macro_hit@{k}"] - summary[f"blend_of_macro_hit@{k}"]
    summary[f"delta_lbc_vs_blend_query@{k}"] = summary[f"lbc_query_hit@{k}"] - summary[f"blend_query_hit@{k}"]
    summary[f"delta_lbc_vs_fullblend_of_macro@{k}"] = (
        summary[f"lbc_of_macro_hit@{k}"] - summary[f"fullblend_of_macro_hit@{k}"]
    )
    summary[f"delta_lbc_vs_fullblend_query@{k}"] = (
        summary[f"lbc_query_hit@{k}"] - summary[f"fullblend_query_hit@{k}"]
    )
    summary[f"delta_lbc_vs_lbc_nofreq_of_macro@{k}"] = (
        summary[f"lbc_of_macro_hit@{k}"] - summary[f"lbc_nofreq_of_macro_hit@{k}"]
    )
    summary[f"delta_lbc_vs_lbc_nofreq_query@{k}"] = (
        summary[f"lbc_query_hit@{k}"] - summary[f"lbc_nofreq_query_hit@{k}"]
    )
    return summary, detail


def split_ofs(df: pd.DataFrame, seed: int, train_fraction: float) -> tuple[list[str], list[str]]:
    positive_ofs = sorted(df.loc[df["label"] == 1, "old_fragment_smiles"].unique())
    if len(positive_ofs) < 3:
        raise ValueError("Need at least three positive-support OFs for repeated OF split.")
    rng = np.random.RandomState(seed)
    shuffled = positive_ofs[:]
    rng.shuffle(shuffled)
    n_train = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_fraction)))
    return shuffled[:n_train], shuffled[n_train:]


def run_repeated_of_split(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    details = []
    for i in range(args.n_seeds):
        split_seed = args.seed + i
        train_ofs, test_ofs = split_ofs(df, split_seed, args.train_fraction)
        train_df = df[df["old_fragment_smiles"].isin(train_ofs)].copy()
        test_df = df[df["old_fragment_smiles"].isin(test_ofs)].copy()
        scored = add_scores(train_df, test_df, args.k)
        summary, detail = summarize_scored(scored, args.k, i)
        summary.update({"split_seed": split_seed, "n_train_of": len(train_ofs), "n_test_of": len(test_ofs)})
        summaries.append(summary)
        details.append(detail)
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def run_external_holdout(train_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = add_scores(train_df, test_df, args.k)
    summary, detail = summarize_scored(scored, args.k, "external")
    return pd.DataFrame([summary]), detail


def make_selftest_matrix(path: Path) -> None:
    ofs = ["OF_A", "OF_B", "OF_C", "OF_D", "OF_E", "OF_F"]
    cands = ["C_GOOD", "C_NEAR", "C_BAD1", "C_BAD2", "C_BAD3"]
    rows = []
    for of_idx, of_s in enumerate(ofs):
        for q_idx in range(4):
            qid = f"{of_s}_q{q_idx}"
            good_idx = of_idx % 2
            for cand_idx, cand in enumerate(cands):
                label = int(cand_idx == good_idx)
                rows.append(
                    {
                        "dataset": "selftest",
                        "query_id": qid,
                        "old_fragment_smiles": of_s,
                        "candidate_smiles": cand,
                        "label": label,
                        "morgan": 0.90 if label else 0.20 + 0.02 * cand_idx,
                        "bit_corr": 0.85 if label else 0.05 * cand_idx,
                        "dHeavy": 0.0 if label else 3.0 + cand_idx,
                        "dRings": 0.0 if label else 1.0,
                        "dMW": 0.05 if label else 0.5,
                        "dLogP": 0.05 if label else 0.4,
                        "dTPSA": 0.05 if label else 0.4,
                    }
                )
    write_table(pd.DataFrame(rows), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, help="Evaluation matrix CSV/CSV.GZ/Parquet.")
    parser.add_argument("--train-matrix", type=Path, help="Training matrix for external_holdout protocol.")
    parser.add_argument("--dataset-name", default="external_matrix")
    parser.add_argument("--protocol", choices=["repeated_of_split", "external_holdout"], default="repeated_of_split")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        tmp_dir = Path(tempfile.mkdtemp(prefix="lbc_external_selftest_"))
        args.matrix = tmp_dir / "selftest_matrix.csv"
        args.protocol = "repeated_of_split"
        args.n_seeds = 3
        args.dataset_name = "selftest"
        make_selftest_matrix(args.matrix)

    if args.matrix is None:
        raise SystemExit("--matrix is required unless --self-test is set.")

    test_df = ensure_features(validate_matrix(read_table(args.matrix), args.dataset_name))
    if args.protocol == "external_holdout":
        if args.train_matrix is None:
            raise SystemExit("--train-matrix is required for external_holdout.")
        train_df = ensure_features(validate_matrix(read_table(args.train_matrix), f"{args.dataset_name}_train"))
        summary, detail = run_external_holdout(train_df, test_df, args)
    else:
        summary, detail = run_repeated_of_split(test_df, args)

    summary_path = args.out_dir / f"{args.dataset_name}_summary.csv"
    detail_path = args.out_dir / f"{args.dataset_name}_detail.csv"
    meta_path = args.out_dir / f"{args.dataset_name}_run_metadata.json"
    write_table(summary, summary_path)
    write_table(detail, detail_path)
    meta_path.write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "protocol": args.protocol,
                "matrix": str(args.matrix),
                "train_matrix": str(args.train_matrix) if args.train_matrix else None,
                "k": args.k,
                "n_rows": int(len(test_df)),
                "n_queries": int(test_df["query_id"].nunique()),
                "n_of": int(test_df["old_fragment_smiles"].nunique()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")


if __name__ == "__main__":
    main()
