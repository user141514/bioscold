from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_external_validation import add_scores, summarize_scored


def _row(qid, old_fragment, candidate, label, freq_like, content_like):
    return {
        "dataset": "diagnostic_v2_test",
        "query_id": qid,
        "old_fragment_smiles": old_fragment,
        "candidate_smiles": candidate,
        "label": label,
        "morgan": content_like,
        "bit_corr": content_like * 0.8,
        "dHeavy": 1.0 - content_like,
        "dRings": 0.0,
        "dMW": 1.0 - content_like,
        "dLogP": 1.0 - content_like,
        "dTPSA": 1.0 - content_like,
        "freq_like": freq_like,
    }


def test_diagnostic_v2_scores_include_fullblend_and_lbc_nofreq():
    train = pd.DataFrame(
        [
            _row("tr_a", "OF_A", "C_GLOBAL", 1, 1.0, 0.20),
            _row("tr_a", "OF_A", "C_NEAR_A", 0, 0.0, 0.95),
            _row("tr_b", "OF_B", "C_GLOBAL", 1, 1.0, 0.25),
            _row("tr_b", "OF_B", "C_NEAR_B", 0, 0.0, 0.90),
            _row("tr_c", "OF_C", "C_LOCAL", 1, 0.2, 0.90),
            _row("tr_c", "OF_C", "C_FAR", 0, 0.0, 0.10),
        ]
    )
    test = pd.DataFrame(
        [
            _row("te_a", "OF_D", "C_GLOBAL", 1, 1.0, 0.25),
            _row("te_a", "OF_D", "C_NEAR_D", 0, 0.0, 0.95),
            _row("te_b", "OF_E", "C_LOCAL", 1, 0.2, 0.90),
            _row("te_b", "OF_E", "C_FAR", 0, 0.0, 0.10),
        ]
    )

    scored = add_scores(train, test, k=1)
    summary, detail = summarize_scored(scored, k=1, seed=0)

    assert "score_fullblend" in scored.columns
    assert "score_lbc_nofreq" in scored.columns
    assert "fullblend_hit@1" in detail.columns
    assert "lbc_nofreq_hit@1" in detail.columns
    assert "fullblend_query_hit@1" in summary
    assert "lbc_nofreq_query_hit@1" in summary
    assert summary["fullblend_alpha"] in {0.0, 0.25, 0.5, 0.75, 1.0}
