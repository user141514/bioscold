#!/usr/bin/env python3
"""Counterfactual content calibration scorers for candidate-heldout diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONTENT_FEATURE_COLUMNS = [
    "morgan",
    "bit_corr",
    "dHeavy",
    "dRings",
    "dMW",
    "dLogP",
    "dTPSA",
]

CONTEXT_COLUMNS = ["ca_bucket", "attachment_signature", "endpoint", "target_context"]
CFC3_C_VALUES = [0.03, 0.1, 0.3, 1.0]


@dataclass(frozen=True)
class CFC3Config:
    name: str = "cf_c3"
    use_ca_bucket: bool = True
    use_context: bool = True
    use_ca_matched_selection: bool = True
    use_target_context: bool = True


CFC3_VARIANT_CONFIGS = {
    "cf_c3": CFC3Config(name="cf_c3"),
    "cf_c3_no_ca_bucket": CFC3Config(name="cf_c3_no_ca_bucket", use_ca_bucket=False),
    "cf_c3_no_context": CFC3Config(
        name="cf_c3_no_context",
        use_ca_bucket=False,
        use_context=False,
        use_target_context=False,
    ),
    "cf_c3_no_ca_match": CFC3Config(name="cf_c3_no_ca_match", use_ca_matched_selection=False),
    "cf_c3_no_target": CFC3Config(name="cf_c3_no_target", use_target_context=False),
}


@dataclass
class CFC3Model:
    scaler: StandardScaler
    encoder: OneHotEncoder | None
    classifier: LogisticRegression
    ca_edges: np.ndarray
    kept_targets: set[str]
    config: CFC3Config
    context_columns: list[str]
    c_value: float
    training_rows: int
    context_policy: str
    feature_names: list[str]


def ca_content_score(frame: pd.DataFrame) -> pd.Series:
    score = 0.75 * frame["morgan"] + 0.25 * (
        1.0 / (1.0 + frame["dHeavy"] + frame["dRings"] + frame["dMW"] + frame["dLogP"])
    )
    return pd.Series(score, index=frame.index, dtype=float)


def ca_bucket_edges(scores: pd.Series, n_buckets: int = 10) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return np.asarray([0.0, 1.0], dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_buckets + 1)))
    if len(edges) < 2:
        value = float(edges[0])
        return np.asarray([value - 0.5, value + 0.5], dtype=float)
    return edges.astype(float)


def assign_ca_buckets(scores: pd.Series, edges: np.ndarray) -> pd.Series:
    if len(edges) <= 2:
        buckets = np.zeros(len(scores), dtype=np.int16)
    else:
        buckets = np.searchsorted(edges[1:-1], np.asarray(scores, dtype=float), side="right").astype(np.int16)
    return pd.Series([f"ca_bucket_{int(x)}" for x in buckets], index=scores.index, dtype=object)


def resolve_cf_c3_config(config: CFC3Config | None = None) -> CFC3Config:
    return config if config is not None else CFC3_VARIANT_CONFIGS["cf_c3"]


def context_columns_for_config(config: CFC3Config | None = None) -> list[str]:
    config = resolve_cf_c3_config(config)
    columns: list[str] = []
    if config.use_ca_bucket:
        columns.append("ca_bucket")
    if config.use_context:
        columns.extend(["attachment_signature", "endpoint"])
        if config.use_target_context:
            columns.append("target_context")
    return columns


def collapse_target_context(
    frame: pd.DataFrame,
    min_rows: int = 500,
    min_positive: int = 20,
    kept_targets: set[str] | None = None,
) -> tuple[pd.Series, set[str]]:
    if "target_key" not in frame.columns:
        return pd.Series(["__OTHER__"] * len(frame), index=frame.index, dtype=object), set()

    targets = frame["target_key"].fillna("__MISSING__").astype(str)
    if kept_targets is None:
        stats = frame.assign(_target_key=targets).groupby("_target_key")["label"].agg(["size", "sum"])
        kept_targets = set(stats[(stats["size"] >= min_rows) & (stats["sum"] >= min_positive)].index.astype(str))
    collapsed = targets.where(targets.isin(kept_targets), "__OTHER__")
    return pd.Series(collapsed, index=frame.index, dtype=object), set(kept_targets)


def add_cf_c3_context_columns(
    frame: pd.DataFrame,
    ca_edges: np.ndarray,
    kept_targets: set[str] | None = None,
    target_min_rows: int = 500,
    target_min_positive: int = 20,
    config: CFC3Config | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    config = resolve_cf_c3_config(config)
    out = frame.copy()
    kept = set(kept_targets or set())
    if config.use_ca_bucket:
        out["ca_bucket"] = assign_ca_buckets(ca_content_score(out), ca_edges)
    if config.use_context:
        for column in ["attachment_signature", "endpoint"]:
            if column not in out.columns:
                out[column] = "__MISSING__"
            out[column] = out[column].fillna("__MISSING__").astype(str)
        if config.use_target_context:
            out["target_context"], kept = collapse_target_context(
                out,
                min_rows=target_min_rows,
                min_positive=target_min_positive,
                kept_targets=kept_targets,
            )
    return out, kept


def select_ca_matched_rows(frame: pd.DataFrame, negatives_per_positive: int = 5) -> pd.DataFrame:
    ca_scores = ca_content_score(frame)
    selected: set[int] = set()
    for _qid, group in frame.groupby("query_id", sort=False):
        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0]
        selected.update(int(idx) for idx in positives.index)
        if positives.empty or negatives.empty:
            continue
        negative_scores = ca_scores.loc[negatives.index]
        for pos_idx in positives.index:
            distances = (negative_scores - float(ca_scores.loc[pos_idx])).abs().sort_values(kind="mergesort")
            selected.update(int(idx) for idx in distances.head(negatives_per_positive).index)
    return frame.loc[sorted(selected)].copy()


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["label"].astype(int)
    n = float(len(frame))
    class_counts = labels.value_counts().to_dict()
    class_weight = labels.map(lambda value: n / (2.0 * class_counts[int(value)])).astype(float)

    query_counts = frame.groupby("query_id")["query_id"].transform("size").astype(float)
    query_weight = 1.0 / query_counts

    candidate_counts = frame.groupby("candidate_smiles")["candidate_smiles"].transform("size").astype(float)
    candidate_weight = 1.0 / np.sqrt(candidate_counts)

    weights = np.asarray(class_weight * query_weight * candidate_weight, dtype=float)
    mean = float(weights.mean()) if len(weights) else 1.0
    return weights / mean if mean > 1e-12 else weights


def _make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover - sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _interaction_matrix(x_scaled: np.ndarray, context_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    blocks = [sparse.csr_matrix(x_scaled)]
    for feature_idx in range(x_scaled.shape[1]):
        blocks.append(context_matrix.multiply(x_scaled[:, feature_idx][:, None]))
    return sparse.hstack(blocks, format="csr")


def _feature_names(encoder: OneHotEncoder | None, context_columns: list[str]) -> list[str]:
    names = list(CONTENT_FEATURE_COLUMNS)
    if not context_columns or encoder is None:
        return names
    if hasattr(encoder, "get_feature_names_out"):
        context_names = list(encoder.get_feature_names_out(context_columns))
    else:  # pragma: no cover - sklearn < 1.0
        context_names = list(encoder.get_feature_names(context_columns))
    for feature in CONTENT_FEATURE_COLUMNS:
        names.extend([f"{feature}*{context}" for context in context_names])
    return names


def build_cf_c3_matrix(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    encoder: OneHotEncoder | None,
    fit: bool,
    context_columns: list[str] | None = None,
) -> sparse.csr_matrix:
    if context_columns is None:
        context_columns = CONTEXT_COLUMNS
    x_values = frame[CONTENT_FEATURE_COLUMNS].astype(float)
    x_scaled = scaler.fit_transform(x_values) if fit else scaler.transform(x_values)
    if not context_columns:
        return sparse.csr_matrix(np.asarray(x_scaled, dtype=float))
    if encoder is None:
        raise ValueError("A OneHotEncoder is required when CF-C3 context columns are enabled.")
    contexts = frame[context_columns].astype(str)
    context_matrix = encoder.fit_transform(contexts) if fit else encoder.transform(contexts)
    return _interaction_matrix(np.asarray(x_scaled, dtype=float), context_matrix.tocsr())


def describe_context_policy(
    config: CFC3Config,
    target_min_rows: int,
    target_min_positive: int,
) -> str:
    parts = []
    parts.append("ca10" if config.use_ca_bucket else "no_ca_bucket")
    if config.use_context:
        parts.extend(["attachment", "endpoint"])
        if config.use_target_context:
            parts.append(f"target_ge{target_min_rows}_pos{target_min_positive}")
        else:
            parts.append("no_target")
    else:
        parts.append("global_content_only")
    parts.append("ca_matched" if config.use_ca_matched_selection else "all_train_rows")
    return f"{config.name}:{'+'.join(parts)}"


def fit_cf_c3(
    train_df: pd.DataFrame,
    c_value: float = 0.1,
    negatives_per_positive: int = 5,
    target_min_rows: int = 500,
    target_min_positive: int = 20,
    config: CFC3Config | None = None,
) -> CFC3Model:
    config = resolve_cf_c3_config(config)
    selected = (
        select_ca_matched_rows(train_df, negatives_per_positive=negatives_per_positive)
        if config.use_ca_matched_selection
        else train_df.copy()
    )
    if selected.empty or selected["label"].nunique() < 2:
        raise ValueError("CF-C3 selected training rows must contain both classes.")

    ca_edges = ca_bucket_edges(ca_content_score(selected))
    selected_context, kept_targets = add_cf_c3_context_columns(
        selected,
        ca_edges,
        target_min_rows=target_min_rows,
        target_min_positive=target_min_positive,
        config=config,
    )
    context_columns = context_columns_for_config(config)
    scaler = StandardScaler()
    encoder = _make_encoder() if context_columns else None
    x_train = build_cf_c3_matrix(selected_context, scaler, encoder, fit=True, context_columns=context_columns)
    clf = LogisticRegression(C=c_value, max_iter=1000, solver="liblinear", random_state=20260609)
    clf.fit(x_train, np.asarray(selected_context["label"], dtype=int), sample_weight=sample_weights(selected_context))
    return CFC3Model(
        scaler=scaler,
        encoder=encoder,
        classifier=clf,
        ca_edges=ca_edges,
        kept_targets=kept_targets,
        config=config,
        context_columns=context_columns,
        c_value=float(c_value),
        training_rows=int(len(selected_context)),
        context_policy=describe_context_policy(config, target_min_rows, target_min_positive),
        feature_names=_feature_names(encoder, context_columns),
    )


def predict_cf_c3(model: CFC3Model, frame: pd.DataFrame) -> pd.Series:
    context_frame, _kept = add_cf_c3_context_columns(
        frame,
        model.ca_edges,
        kept_targets=model.kept_targets,
        config=model.config,
    )
    x_test = build_cf_c3_matrix(
        context_frame,
        model.scaler,
        model.encoder,
        fit=False,
        context_columns=model.context_columns,
    )
    scores = model.classifier.decision_function(x_test)
    return pd.Series(scores, index=frame.index, dtype=float)
