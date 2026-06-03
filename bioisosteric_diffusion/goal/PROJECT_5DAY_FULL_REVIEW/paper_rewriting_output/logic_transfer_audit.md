# Logic Transfer Audit

Compares original manuscript against revised manuscript. Every change verified against writing rationale matrix.

---

## Structural Changes

| Change | Original | Revised | Verdict |
|---|---|---|---|
| Title | 16 words: "Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring" | 10 words: "Leakage-Controlled Fragment Replacement Ranking with Audited Candidate-Level Scoring" | PASS — "Audited" signals prior_ranks finding |
| Abstract structure | Single dense paragraph | 2 paragraphs: ¶1=benchmark, ¶2=shortcut audit + A4C | PASS — matches 2-contribution motivation |
| Introduction hook | "Replacing a selected substituent...remains difficult to benchmark rigorously" | "Chemical databases contain millions of structure-derived substitution patterns...yet computational methods for ranking fragment replacements lack evaluation protocols that prevent models from exploiting dataset-specific regularities" | PASS — computational framing per JCIM norm |
| Introduction difficulties | 3 difficulties (leakage, strong baselines, signal integration) | 2 difficulties (leakage with literature anchors, shortcut features under distribution shift) | PASS — sharper, each tied to a contribution |
| Introduction contributions | 4 contributions with protocol separation and A4C as separate items | 2 contributions + "methodological safeguards" language | PASS — matches confirmed motivation |
| Introduction quantitative preview | 4-sentence data dump | 3 sentences, tightened | PASS — cleaner |
| Related Work | NeBULA: "311,543 rules, 5 categories" | NeBULA: "over 700 references, multiple categories" | PASS — trimmed detail |
| Methods preamble | Mixed roadmap + timeline disclosure ("The base ranker architecture...was configured...") | Clean 4-sentence roadmap. Timeline moved to §3.5. | PASS — logical placement |
| §3.3 Base rankers | No routing mention | "+1 sentence: query-level routing failed during development (see §5.2)" | PASS — foreshadows routing insight |
| §3.5 Prior_ranks | Timeline not in this section | "The base ranker architectures...were configured on the development set. The prior_ranks removal was identified during post-audit analysis..." | PASS — timeline where it belongs |
| §4.1 Primary result | No routing mention | "+1 sentence: candidate-level design partially motivated by routing failure (see §5.2)" | PASS — foreshadows routing insight |
| §6 Conclusion ¶1 | 4-contribution echo, routing prominent | 2-contribution structure, routing brief, caveats present | PASS — aligned with introduction |
| References | 18 references | 20 references (+2 leakage literature) | PASS — C44 and C46 support motivation |

## Quantitative Claims — Preservation Audit

| Evidence ID | Original Location | Revised Location | Preserved? | Notes |
|---|---|---|---|---|
| E01-E25 | §4.1-4.5 | §4.1-4.5 | YES | All locked numbers unchanged |
| Table 1-4 | §4 | §4 | YES | All tables identical |
| Table M1-M5 | §3 | §3 | YES | All tables identical |
| All CIs | Throughout | Throughout | YES | All bootstrap CIs preserved |
| Post-audit caveat | Throughout | Throughout | YES | Caveat present in Abstract, Introduction, §3.5, §4.1-4.2, §5.1, §6 |

## Sections NOT Changed — Audit

| Section | Status | Notes |
|---|---|---|
| §3.1 Task Formulation | UNCHANGED | Notation preserved |
| §3.2 Benchmark Construction | UNCHANGED | Tables M1-M2 preserved |
| §3.3.1-3.3.4 (5 rankers) | UNCHANGED | Math preserved |
| §3.4 Candidate-Level Scorer | UNCHANGED | Table M3 preserved |
| §3.6 Evaluation Protocol | UNCHANGED | Table M4 preserved |
| §3.7 A4C | UNCHANGED | Table M5 preserved |
| §4.2 Prior_ranks ablation | UNCHANGED | Table 2 preserved |
| §4.3 Rescue/lost | UNCHANGED | Table 3 preserved |
| §4.4 Schema alignment | UNCHANGED | — |
| §4.5 A4C alert rates | UNCHANGED | Table 4 preserved |
| §5.1 Principal Findings | UNCHANGED | — |
| §5.2 Routing insight | UNCHANGED | Now foreshadowed in §3.3, §4.1 |
| §5.3 Prior_ranks shortcuts | UNCHANGED | — |
| §5.4 Dual-mode workflow | UNCHANGED | — |
| §5.5 Relationship to other approaches | UNCHANGED | — |
| §5.6 Limitations | UNCHANGED | — |
| Data & Software | UNCHANGED | — |

## Citation Changes

| Ref | Status | Notes |
|---|---|---|
| [1]-[18] | PRESERVED | All existing references kept |
| [44] Kapoor & Narayanan 2023 | ADDED | Supports leakage motivation in Introduction ¶2 |
| [46] Wallach et al. 2018 | ADDED | JCIM paper directly supporting memorization vs. generalization |

## Overall Verdict

**PASS** — All quantitative claims preserved. All structural changes are rhetorical/structural only. No evidence changed. No numbers modified. Post-audit caveat maintained throughout. Contribution hierarchy clarified from 4→2. Prior_ranks finding structurally elevated. Routing insight foreshadowed. A4C scoped as supporting. Two new references added for motivation anchoring.
