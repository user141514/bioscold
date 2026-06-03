# Writing Rationale Matrix

Execution plan for the rewrite. Follows confirmed motivation: Option A — benchmark arc with prior_ranks shortcut audit as secondary thread.

---

## Whole-Work Framework

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| W0 | Whole-paper structure | 4-contribution framing dilutes primary/secondary hierarchy. Prior_ranks finding is structurally submerged. | Confirmed motivation: benchmark arc (primary) + shortcut audit (secondary). A4C is supporting workflow. | JCIM benchmark papers use 1-2 contribution framing with supporting analyses as methodological safeguards, not separate contributions. Ertl (2020): one core method + ablation. | JCIM expects clear contribution statement in Introduction. Reviewers penalize overclaiming. | All 25 quantitative claims (E01-E25) remain locked. | Restructure to 2-contribution framing: (1) benchmark + scorer, (2) prior_ranks shortcut audit. A4C and protocol separation are methodological safeguards. Elevate prior_ranks from Methods §3.5 footnote to Results §4.2 headline. Foreshadow routing insight. | Verify: Introduction lists exactly 2 contributions. Prior_ranks finding is in Abstract and Introduction, not buried in Methods. A4C is described as "supporting workflow feature." |

---

## Title

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| T1 | Title | 16 words, misses prior_ranks narrative. "Closed-Vocabulary" is accurate but adds length. | Shortened title signals sharper positioning. Including "audited" or "shortcut" hints at the paper's most citeable finding. | JCIM titles average 8-14 words. Ertl (2020): "Identification of Bioisosteric Substituents by a Deep Neural Network" (10 words). Hussain & Rea (2010): more descriptive (12 words). | JCIM: descriptive titles, no colons unless necessary, method-focused. | Evidence E02 (Top-10 0.9243), E19 (5 prior_ranks features). | Shorten to 10-14 words. Proposed: "Leakage-Controlled Fragment Replacement Ranking with Audited Candidate-Level Scoring." | Title ≤14 words. Contains "leakage-controlled" and "candidate-level scoring." Prior_ranks audit signaled. |

## Abstract

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| A1 | Abstract ¶1 — benchmark | Single dense paragraph mixes all findings. Prior_ranks narrative submerged. | Restructure to 2-part: benchmark first (primary), shortcut audit second (secondary), A4C as brief supporting note. | JCIM abstracts follow: problem → method → key result with numbers → significance. Ertl (2020) abstract: 4 sentences, one quantitative result. | JCIM: unstructured, 150-250 words, must be self-contained, quantitative. | E01, E02, E03, E04 | Split into 2 paragraphs: ¶1 = benchmark + scorer (Top-10 0.9243, Δ+0.0686), ¶2 = prior_ranks deletion (+0.0393, 5.9× reduction) + A4C scope caveats. ~170 words total. | Abstract has exactly 2 paragraphs. ¶1 covers benchmark contribution. ¶2 covers prior_ranks and A4C scope. All numbers have CIs. Post-audit caveat present. |

## Introduction

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| I1 | Hook (¶1) | Vague: "remains difficult to benchmark rigorously." No computational scale hook. | JCIM norm: open with computational framing (chemical space scale, algorithmic challenge). | Ertl (2020): opens with "bioisosteric replacement is a key concept in drug design." Hussain & Rea (2010): opens with computational scale ("millions of compounds"). | JCIM computational papers open with informatics challenge, not biological motivation. | — | Rewrite: "Chemical databases contain millions of structure-derived substitution patterns, but computational tools for ranking fragment replacements lack rigorous evaluation protocols." → narrow to leakage. | Hook opens with computational scale/scope. No biological motivation in first 2 sentences. |
| I2 | Three difficulties (¶2) | Listed as three separate items. "Non-trivial" is weak. | Sharpen to two core difficulties that directly motivate the two contributions. | Wallach et al. (2018, JCIM): random splits reward memorization → directly supports leakage motivation. Geirhos et al. (2020): shortcut learning framework → supports prior_ranks motivation. | Gap stated as technical limitation, not generic need. | C44 (Kapoor 2023), C46 (Wallach 2018) for leakage; C16 (Geirhos 2020) for shortcuts. | Restructure to 2 difficulties: (1) Random splits leak transform identities, inflating scores. (2) Models exploit distribution-specific shortcuts (e.g., rank features) that fail under fragment identity shift. | Exactly 2 difficulties stated. Each tied to a specific technical mechanism. Literature anchors cited. |
| I3 | Contributions (¶3-4) | 4 contributions: benchmark, scorer, protocol separation, A4C workflow. Dilutes hierarchy. | 2 contributions: (1) benchmark + scorer, (2) prior_ranks shortcut audit. | JCIM papers typically claim 1-2 contributions. Four is uncommon and dilutes. | Clear contribution statement in Introduction is reviewer expectation. | All evidence in E01-E25. | Restructure: Contribution 1 = transform-heldout benchmark + candidate-level scorer (primary). Contribution 2 = prior_ranks shortcut audit demonstrating that feature deletion improves generalization (secondary). A4C and protocol separation are methodological safeguards. | Introduction lists exactly 2 contributions. "Methodological safeguards" language used for A4C and protocol separation. |
| I4 | Quantitative preview (¶5) | Functional but reads as data dump. | Preview key numbers to anchor expectations. | JCIM norm: state quantitative finding in Introduction, not just Abstract. | — | E02, E03, E04 | Tighten to: Top-10 = 0.9243, Δ+0.0686 over Score Blend, prior_ranks removal Δ+0.0393, post-audit caveat. 3 sentences. | 3-sentence quantitative preview. All CIs present or caveat stated. |

## Related Work (minor polish only — structure preserved)

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| R1 | MMP paragraph | Establishes data foundation. Clear scope caveat. | Background. | — | — | C01-C04 | Keep. Minor polish: trim any excess. | Under 80 words. Scope caveat present. |
| R2 | Database tools | SwissBioisostere, CReM, NeBULA. NeBULA paragraph is dense. | Positions work relative to database-driven tools. | — | Related Work should cover breadth without excessive detail on any single tool. | C05-C07 | Trim NeBULA detail: drop "311,543 rules" and "5 categories" — keep "700+ references" and "web-based platform." | NeBULA ≤2 sentences. |
| R3 | Deep learning | Ertl, DeepBioisostere, GraphBioisostere. | Positions work relative to learning-based methods. | — | — | C08-C10 | Keep. Minor polish. | Transition sentence to ranking perspective present. |
| R4 | Rank fusion | Borda, RRF. | Context for fusion choice. | — | — | C11-C12 | Keep. | — |
| R5 | Alerts | PAINS, Brenk, Helmke. | A4C background. | — | — | C13-C15 | Keep. | — |

## Methods (structure preserved — targeted changes only)

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| M0 | Methods preamble | Long paragraph mixes roadmap + timeline disclosure. | Timeline disclosure belongs in §3.5, not preamble. | — | Methods should be declarative, not apologetic. | — | Move timeline disclosure to §3.5. Preamble: clean roadmap (3.1-3.7). 3 sentences max. | Preamble ≤3 sentences. No caveats. |
| M1 | §3.3: Base rankers overview | Lists 5 base rankers. No mention of query-level routing. | Foreshadows routing insight (Arc D). | — | — | E22 (DE gap), E23 (Borda gain) | Add 1 sentence after base ranker overview: "Query-level routing strategies that select among base rankers on a per-query basis were explored during development but did not improve over per-candidate scoring (Section 5.2)." | 1 sentence foreshadowing present. |
| M2 | §3.5: Prior_ranks | Strong section. Timeline disclosure should be here, not preamble. | Central to secondary contribution. | Geirhos et al. (2020) shortcut learning framework. | Methods should clearly separate development timeline from final configuration. | E19-E20, Table 2 | Move timeline disclosure here. Add: "The prior_ranks removal was identified during post-audit analysis after the initial 82-feature blind evaluation revealed fragment-specific degradation. We therefore report the 77-feature configuration as a post-audit locked model." | Timeline disclosure in §3.5. Mechanism (8→19 fragment shift) clearly stated. |

## Results (structure preserved — targeted additions)

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| RS1 | §4.1: Primary blind result | Table 1 with all methods. Narrative separates prospective from post-audit. | Primary evidence. | JCIM results pattern: benchmark → comparison table → interpretation. | Benchmark-first results. | E01-E11 | Keep. Add 1 sentence: "The consistent failure of query-level routing to improve over per-candidate scoring (Section 5.2, Table SX) further motivates the candidate-level design." | 1 sentence routing foreshadowing added. Table 1 unchanged. |
| RS2 | §4.2: Prior_ranks ablation | Strong section. Table 2. | Secondary evidence — strongest finding. | Ablation as mechanism, not just comparison. | JCIM values mechanistic understanding over pure ranking tables. | E04, E19, E24-E25 | Keep as-is. This is the paper's best section. | Table 2 unchanged. Shortcut interpretation present. |

## Discussion (minor elevation)

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| D1 | §5.2: Candidate-level > query-level | Strong insight but only in Discussion. Needs Results foreshadowing. | Elevated secondary thread. | — | Discussion should return to motivation and synthesize, not introduce new Results. | E22, E23, routing experiments | Keep analysis. Verify that §3.3 and §4.1 now foreshadow this. | Foreshadowing present in §3.3 and §4.1. §5.2 synthesizes rather than introduces. |
| D2 | §5.3: Prior_ranks as shortcuts | Strongest paragraph. Borda structural role well-integrated. | Central mechanism finding. | — | Discussion should explain WHY, not just report WHAT. | E04, E19, E20 | Keep as-is. This is citeable. | Alternative explanations acknowledged. Design rule stated. |

## Conclusion

| Row ID | Manuscript Unit | Original Problem or Planned Function | Motivation Link | Reference/SOTA Pattern Learned | Target Scene or Venue Norm | User Evidence or Citation Anchor | Planned Change | Final Text Check |
|---|---|---|---|---|---|---|---|
| CN1 | ¶1 | Restates contributions but routing insight (Arc D) appears for first time prominently. | Better alignment with 2-contribution structure. | JCIM conclusions: focused assessment of scope + concrete future work. | Limitation-aware close. | E02, E03, E04 | Restructure: sentence 1 = benchmark contribution, sentence 2 = prior_ranks deletion, sentence 3 = routing insight (brief), sentence 4 = post-audit + A4C caveats. | 2-contribution structure reflected. Routing insight ≤1 sentence. Caveats present. |
| CN2 | ¶2-3 | Dual-mode + limitations → future work. | Supporting. | — | — | — | Keep as-is. | — |
