#!/usr/bin/env python3
"""Route-A D4S2 preflight and train/val reranker matrix build."""

from __future__ import annotations

import pandas as pd

from routeA_d4s2_common import (
    ART,
    B0_VAL_QUERY_PATH,
    MATRIX_DIR,
    OUT,
    build_feature_tensor_rows,
    compute_query_meta,
    fit_baselines,
    fit_old_fragment_clusterer,
    full_attach_arrays,
    load_manifest,
    load_vocab_df,
    preflight_checks,
    score_de_full,
    score_hgb_full,
    write_feature_schema,
    dump_json,
    log,
)


def main():
    report, summary = preflight_checks()
    report.to_csv(OUT / "d4s2_preflight_report.csv", index=False)
    dump_json(OUT / "d4s2_preflight_summary.json", summary)
    if (report["status"] == "FAIL").any():
        raise SystemExit("PREFLIGHT_FAIL")

    manifest = load_manifest()
    vocab_df = load_vocab_df()
    artifacts = fit_baselines(manifest, vocab_df)

    clusterer = fit_old_fragment_clusterer(
        manifest.loc[manifest["split_new"] == "train", "old_fragment_smiles"].astype(str).tolist()
    )
    query_meta_df, freq_thresholds = compute_query_meta(
        manifest,
        artifacts.global_counts,
        clusterer,
        B0_VAL_QUERY_PATH,
    )
    schema = write_feature_schema(freq_thresholds)

    build_rows = []
    for split_name in ["train", "val"]:
        log(f"Building D4S2 {split_name} matrix.")
        split_df = manifest.loc[manifest["split_new"] == split_name].copy()
        de_scores, de_ranks = score_de_full(split_df, artifacts)
        hgb_scores, hgb_ranks = score_hgb_full(split_df, artifacts)
        attach_scores, attach_ranks = full_attach_arrays(split_df, artifacts)
        meta = query_meta_df.loc[query_meta_df["split"] == split_name].copy()
        out = build_feature_tensor_rows(
            query_df=split_df,
            query_meta_df=meta,
            artifacts=artifacts,
            de_scores=de_scores,
            de_ranks=de_ranks,
            hgb_scores=hgb_scores,
            hgb_ranks=hgb_ranks,
            attach_scores=attach_scores,
            attach_ranks=attach_ranks,
            split_name=split_name,
            include_labels=True,
            csv_prefix=f"d4s2_reranker_matrix_{split_name}",
        )
        build_rows.append(
            {
                "split": split_name,
                "n_queries": out["n_queries"],
                "n_candidates_per_query": out["n_candidates"],
                "feature_dim": out["feature_dim"],
                "matrix_paths": "|".join(out["csv_paths"]),
                "tensor_feature_path": out["feature_path"],
                "tensor_label_path": out["label_path"],
                "query_meta_path": out["query_meta_path"],
                "blind_labels_used": 0,
                "notes": "train/val only; blind matrix deferred until post-selection final evaluation",
            }
        )
    build_rows.append(
        {
            "split": "blind_test",
            "n_queries": "",
            "n_candidates_per_query": len(vocab_df),
            "feature_dim": len(schema["feature_sets"]["F3_rank_score_frequency_chemistry"]),
            "matrix_paths": "",
            "tensor_feature_path": "",
            "tensor_label_path": "",
            "query_meta_path": "",
            "blind_labels_used": 0,
            "notes": "blind matrix will be built after validation model selection",
        }
    )
    pd.DataFrame(build_rows).to_csv(OUT / "d4s2_matrix_build_report.csv", index=False)

    # Convenience copies for the expected top-level names.
    val_dir = MATRIX_DIR / "val"
    blind_dir = MATRIX_DIR / "blind_test"
    blind_dir.mkdir(parents=True, exist_ok=True)
    for src in val_dir.glob("d4s2_reranker_matrix_val*.csv.gz"):
        target = OUT / src.name
        target.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
