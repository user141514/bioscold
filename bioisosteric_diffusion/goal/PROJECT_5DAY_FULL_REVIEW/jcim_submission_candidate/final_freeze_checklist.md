# Final Conservative Freeze Checklist

Audit date: 2026-06-03.

| # | Criterion | Yes/No | Evidence |
|---:|---|:---:|---|
| 1 | Fresh Blind2 is the only primary prospective evaluation. | Yes | Abstract, Introduction, Results 4.1-4.2 |
| 2 | HistGB82 is the primary Top-10 HistGB configuration. | Yes | Abstract, Results Table 6, Discussion 5.1 |
| 3 | HistGB77 is diagnostic comparator only. | Yes | Abstract, Table 6 evidence role, Results 4.4 |
| 4 | Old secondary-blind 0.9243 is absent from main text. | Yes | Banned phrase audit; retained only in SI diagnostic-history tables |
| 5 | Conclusion says query-level Top-10 and grouped uncertainty caveat. | Yes | Conclusion paragraph 1 |
| 6 | Candidate universe is full training-derived vocabulary, not hard attachment-compatible primary policy. | Yes | Methods 3.1 and 3.2 |
| 7 | Table 3 overlap terminology is mathematically consistent. | Yes | Table 3 and SM5a use old-to-replacement pair overlap `(f_old, c)` excluding attachment signature |
| 8 | Provenance table is numbered or moved to supplement. | Yes | Supplementary Table S13 |
| 9 | Supplementary citations all resolve. | Yes | `supplement_cross_reference_audit.md` |
| 10 | Data archive tag resolves publicly. | No | Branch resolves, but exact tag raw manifest returned 404 |
| 11 | References are metadata-verified. | Yes | `reference_metadata_audit.md` |
| 12 | No banned phrases remain. | Yes | `banned_phrase_audit.md` |
| 13 | No new experiments or new claims were added during freeze. | Yes | Only reframing, SI relocation, reference correction, and audit files |
| 14 | Manuscript ends on benchmark-audit framing. | Yes | Final sentence of Conclusion |

Verdict: DO NOT FREEZE AS SUBMISSION-READY until the data archive tag/manifest blocker is resolved.
