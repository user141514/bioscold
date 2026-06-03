# V5 Surgical Revision Log

## Scope

Revised `main_manuscript.md` from the old secondary-blind / 77-feature headline structure into a Fresh Blind2 benchmark-audit structure.

## Main hierarchy changes

- Promoted Fresh Blind2 to the primary prospective evaluation.
- Reframed HistGB82 as the primary prospective Top-10 HistGB configuration.
- Reframed HistGB77 as a no-prior-rank diagnostic that remains baseline-beating but does not reproduce a Top-10 advantage over HistGB82.
- Demoted the original secondary-blind no-prior-rank gain to Supplementary Table S1 / diagnostic history.
- Reframed categorical-aware LambdaRank as a future architecture direction, not the current main method.
- Removed main-text headline use of the old 77-feature secondary-blind Top-10 result.

## Major section edits

- Abstract rewritten around Fresh Blind2 and the HistGB82 prospective result.
- Introduction rewritten around benchmark contribution, Fresh Blind2 prospective scoring, prior-rank instability, and LTR future direction.
- Methods updated to define the query-side transform key, Fresh Blind2 full-161 policy, 82/77 training protocol, and ranking-score language.
- Results section replaced with:
  - 4.1 Benchmark and candidate matrix construction.
  - 4.2 Fresh Blind2 primary prospective performance.
  - 4.3 Query-level and grouped uncertainty.
  - 4.4 Prior-Rank Deletion Does Not Prospectively Replicate as a Top-10 Improvement.
  - 4.5 DE/HGB complementarity.
  - 4.6 Candidate-matrix sensitivity.
  - 4.7 Categorical-aware LambdaRank diagnostic.
  - 4.8 Activity-comparable reranking boundary.
  - 4.9 Calibration sanity.
- Discussion and Conclusion rewritten to close on benchmark-audit framing rather than 77-feature improvement.

## Locked Fresh Blind2 values inserted

- Score Blend Top-10 = 0.8077.
- Borda(DE,HGB) Top-10 = 0.8176.
- HistGB82 Top-10 = 0.8480.
- HistGB77 Top-10 = 0.8411.
- HistGB82 - Score Blend Top-10 = +0.0403, 95% CI [0.0355, 0.0446].
- HistGB77 - HistGB82 Top-10 = -0.0070, 95% CI [-0.0094, -0.0044].

## Boundary cleanup

Final forbidden-term scan of `main_manuscript.md` returned no hits for:

- `0.9243`
- `probability` / `probabilities`
- `prior_ranks improves`
- `improves transfer`
- `valid replacement`
- `activity-preserving`
- `activity preservation`
- `final 77`
- `final method`
- `locked`
- `establishes`
- `calibrated`
- `external validation`
- `temporal validation`
- `wet-lab`
- `expert review`
- `uniform improvement`
- `highest novelty`

## Ruflo status

Ruflo swarm use was attempted for this revision stream, but the available Ruflo tools were not usable in this session:

- Earlier agent execution failed because no LLM provider was configured.
- A follow-up `agent_list` call failed with `Transport closed`.

The revision was therefore completed locally.

## Remaining reviewer-facing risks

- Supplementary Table S1 should contain the original secondary-blind diagnostic history so the old no-prior-rank result remains auditable without becoming the headline.
- Supplementary grouped-uncertainty tables should remain synchronized with the main-text statements that grouped Top-10 intervals are wider and can cross zero.
- Candidate-matrix construction details should be kept aligned with the Fresh Blind2 full-161 primary policy and top-150 comparability sensitivity.
- Any later figure/table caption pass must preserve the same evidence hierarchy: Fresh Blind2 / HistGB82 first, HistGB77 diagnostic, LambdaRank future direction.
