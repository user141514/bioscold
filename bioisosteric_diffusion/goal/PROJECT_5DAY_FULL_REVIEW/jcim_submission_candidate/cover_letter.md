# Cover Letter

Dear Editor,

We submit the manuscript entitled "Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring" for consideration in the *Journal of Chemical Information and Modeling*.

This work addresses a methodological problem in structure-derived fragment replacement ranking: random matched molecular pair splits can leak fragment-attachment transforms, and empirical prior features can become non-transferable shortcuts. We introduce a transform-heldout benchmark for closed-vocabulary scaffold-conditioned replacement ranking, evaluate base-ranker fusion under a secondary blind protocol, and present a candidate-level histogram gradient-boosted scorer with an explicit audit trail for feature leakage and post-audit feature pruning.

The main empirical result is a candidate-level scoring framework evaluated on 13,347 secondary blind queries. The initial 82-feature scorer is the prospective candidate-level blind result. A post-audit 77-feature scorer, obtained after removing five sparse prior-rank features identified through blind diagnostics, is reported as a locked post-selection finding that motivates prospective replication rather than as a fully prospective feature-selection result. We have written the manuscript to keep this evidence hierarchy explicit.

The study uses structure-derived labels from ChEMBL matched molecular pairs. It does not claim activity-preserving prediction, wet-lab validation, expert validation, or production-ready medicinal chemistry decision-making. The A4C provenance strata are reported only as a computational triage proxy.

A public source-code and evidence archive is available at https://github.com/user141514/paper1/tree/codex/jcim-algorithm-archive. The curated code archive is recorded at commit `b750de266d6d47e63de0eeaa945cc999e6f3e08c`, the manuscript/supplement synchronization at commit `72b7807948c9a1432b99bf1529d98b3d994a3d1f`, and this submission candidate under tag `jcim-submission-candidate-20260601`.

Sincerely,

Corresponding author
