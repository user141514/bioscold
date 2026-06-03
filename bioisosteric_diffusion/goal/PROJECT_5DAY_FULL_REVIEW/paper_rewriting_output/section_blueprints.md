# Section Blueprints

Redesigned structure per confirmed motivation (Option A: benchmark arc with prior_ranks shortcut audit as secondary thread).

---

## Title (rewrite)
**Current:** "Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring" (16 words)

**Proposed:** "Leakage-Controlled Fragment Replacement Ranking with Audited Candidate-Level Scoring" (10 words)

Or with subtitle: "Leakage-Controlled Fragment Replacement Ranking: A Transform-Heldout Benchmark and Shortcut-Audited Scorer"

---

## Abstract (rewrite — 2-part structure)

**¶1 Benchmark contribution:** Fragment replacement ranking needs leakage-controlled evaluation. We introduce a transform-heldout benchmark (13,347 blind queries) and a candidate-level HistGB scorer achieving Top-10 = 0.9243 (post-audit locked), Δ+0.0686 over Score Blend baseline.

**¶2 Shortcut audit + A4C:** The most impactful feature decision is a deletion — removing five per-query prior-rank features eliminates a non-transferable shortcut (+0.0393, 5.9× hit-loss reduction). A dual-mode A4C computational review proxy provides provenance-stratified alert reporting as a supporting workflow feature. Post-audit finding: replication needed. A4C: unvalidated triage only.

**Style:** Nominal constructions, quantitative throughout, caveats embedded. ~150 words.

---

## 1. Introduction (rewrite — 2-contribution structure)

**¶1 Hook (computational framing per JCIM norm):** Chemical space scale → fragment replacement as algorithmic challenge → existing benchmarks fail under random splits because transform identities leak across sets. (3-4 sentences)

**¶2 Two core difficulties (sharpened from current three):**
1. Leakage: random splits inflate performance through transform memorization
2. Shortcut features: model exploits dataset-specific regularities that fail under distribution shift. The 8-to-19 fragment identity gap.

**¶3 Two contributions (down from current four):**
1. Transform-heldout benchmark + candidate-level scorer (primary)
2. Prior_ranks shortcut audit as mechanism finding (secondary)
A4C workflow and protocol separation are methodological safeguards, not separate contributions.

**¶4 Quantitative preview:** Top-10 = 0.9243 (77-feat), Δ+0.0686, prior_ranks removal Δ+0.0393. Post-audit caveat. A4C scope caveat.

**Style:** Gap-narrowing pattern from exemplars. Technical limitation framing. No "first"/"novel" without justification. ~250 words.

---

## 2. Related Work (minor polish — keep structure)

Current 5-paragraph structure is solid. Changes:
- Trim NeBULA detail (reduce 311,543 rules / 5 categories to essential)
- Add 1 sentence on benchmark methodology literature (data leakage in ML)
- Keep all 5 paragraphs: MMP, database tools, deep learning, rank fusion, alerts

---

## 3. Methods (keep structure, elevate prior_ranks)

Current structure is logical. Key change: move the timeline disclosure from preamble to §3.5.

**Sections (unchanged order):**
- 3.1 Task Formulation (keep)
- 3.2 Benchmark Construction (keep)
- 3.3 Base Ranking Methods (keep — add 1 sentence: why no query-level router)
- 3.4 Candidate-Level Scorer (keep)
- 3.5 Feature Ablation: The Prior_Ranks Removal (keep — move timeline disclosure here)
- 3.6 Evaluation Protocol (keep)
- 3.7 Computational Review Proxy (A4C) (keep)

---

## 4. Results (keep structure, minor reordering)

**Sections (unchanged order):**
- 4.1 Candidate-Level Scoring Improves Blind Ranking (keep — add 1 sentence foreshadowing routing insight)
- 4.2 Removing Non-Transferable Prior-Rank Shortcuts Improves Blind Transfer (keep — this is the paper's strongest section)
- 4.3 Feature Pruning Recovers Baseline Misses (keep)
- 4.4 Categorical Schema Alignment Audit (keep — consider moving to SI if space needed)
- 4.5 Dual-Mode Provenance Strata (keep)

---

## 5. Discussion (keep, minor elevation)

- 5.1 Principal Findings (keep)
- 5.2 Why Candidate-Level Scoring Succeeds (keep — add brief foreshadowing reference in §3.3 and §4.1)
- 5.3 Prior_Ranks as Non-Transferable Shortcuts (keep — strongest paragraph)
- 5.4 Dual-Mode Workflow (keep)
- 5.5 Relationship to Other Approaches (keep)
- 5.6 Limitations and Future Work (keep)

---

## 6. Conclusion (minor rewrite for alignment)

- ¶1 (restate benchmark + prior_ranks finding — 2-part contribution)
- ¶2 (dual-mode workflow — supporting)
- ¶3 (limitations → future work — keep current structure)

---

## Data & Software Availability (keep)

---

## References (keep — add 1-2 from citation bank for Discussion breadth if needed)

Two candidates for addition:
- C44 (Kapoor & Narayanan 2023, leakage in ML-based science) — supports Introduction motivation
- C46 (Wallach et al. 2018, memorization vs generalization in ligand-based classification, JCIM) — directly supports leakage motivation

---

## Structural Changes Summary

| Change | Type | Impact |
|---|---|---|
| Title shortened | Rewrite | High — sharper positioning |
| Abstract: 2-part structure | Rewrite | High — clarifies contribution hierarchy |
| Introduction: 4→2 contributions | Structural | High — aligns with confirmed motivation |
| Introduction: 3→2 difficulties | Structural | Medium — sharper gap framing |
| Introduction hook: computational framing | Rhetorical | Medium — JCIM norm |
| §3.3: add routing foreshadowing | Addition | Medium — supports §5.2 elevation |
| §4.1: add routing insight foreshadowing | Addition | Medium — supports §5.2 elevation |
| §3.5: move timeline disclosure from preamble | Move | Low — logical placement |
| §4.4: consider move to SI | Optional | Low — space management |
| §6: align conclusion with 2-part contribution | Rewrite | Medium — narrative coherence |
| References: +2 citations | Addition | Low — strengthens motivation |
