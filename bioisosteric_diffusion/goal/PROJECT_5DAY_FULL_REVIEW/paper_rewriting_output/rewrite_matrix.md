# Rewrite Matrix

Maps original manuscript units to final units. Each change classified: structural, rhetorical, evidence-related, or language-only.

---

| Original Unit | Final Unit | Change Type | Description |
|---|---|---|---|
| Title (16 words) | Title (10-14 words) | Rhetorical + Structural | Shortened; "audited" signals prior_ranks finding. Removes "Closed-Vocabulary" (accurate but adds length). |
| Abstract — single dense ¶ | Abstract ¶1 (benchmark) + ¶2 (shortcut audit + A4C) | Structural | Split into 2 paragraphs. ¶1 covers benchmark + scorer primary result. ¶2 covers prior_ranks deletion + A4C scope caveats. |
| Introduction ¶1 (vague hook) | Introduction ¶1 (computational framing) | Rhetorical | Sharper computational hook. Opens with chemical space scale, narrows to leakage problem. JCIM computational norm. |
| Introduction ¶2 (3 difficulties) | Introduction ¶2 (2 difficulties) | Structural | Merged difficulties 2+3 into one. Two difficulties: (1) leakage inflates scores, (2) shortcut features fail under distribution shift. |
| Introduction ¶3-4 (4 contributions) | Introduction ¶3-4 (2 contributions + safeguards) | Structural | Two contributions: (1) benchmark + scorer, (2) prior_ranks shortcut audit. A4C and protocol separation reframed as methodological safeguards. |
| Introduction ¶5 (results preview) | Introduction ¶5 (tightened preview) | Language | 3 sentences. Removes data-dump feel. |
| §2 Related Work | §2 Related Work | Language | Minor polish only. NeBULA detail trimmed. Structure preserved. |
| §3 preamble (mixed roadmap + caveat) | §3 preamble (clean roadmap) | Structural | Timeline disclosure moved to §3.5. Preamble: 3-sentence roadmap. |
| §3.3 Base rankers (no routing mention) | §3.3 Base rankers (+ routing foreshadowing) | Structural + Evidence | Added 1 sentence: query-level routing failed → motivates candidate-level design. |
| §3.5 Prior_ranks (timeline in preamble) | §3.5 Prior_ranks (timeline moved here) | Structural | Timeline disclosure ("removal was identified during post-audit analysis...") moved from preamble to §3.5. |
| §4.1 Primary result (no routing mention) | §4.1 Primary result (+ routing foreshadowing) | Structural | Added 1 sentence referencing routing failure as further motivation for candidate-level design. |
| §5.2 Routing insight | §5.2 Routing insight | None | Kept as-is. Now foreshadowed in §3.3 and §4.1. |
| §6 ¶1 (4-contribution echo, routing first) | §6 ¶1 (2-contribution, routing brief) | Structural | Restructured: benchmark first, prior_ranks second, routing brief, caveats last. |
| All other sections | All other sections | None | Kept as-is. Manuscript is already strong in these areas. |

## Sections NOT Changed
- §3.1 Task Formulation — precise notation, no changes needed
- §3.2 Benchmark Construction — tables, leakage verification, all solid
- §3.4 Candidate-Level Scorer — feature families, training protocol, all complete
- §3.6 Evaluation Protocol — bootstrap, rescue/lost, protocol separation, all solid
- §3.7 A4C — well-scoped, no overclaiming
- §4.2-4.5 Results — all tables and narratives strong
- §5.1, 5.3-5.6 Discussion — strong analysis
- Data & Software Availability — complete
- References — kept, may add 2 from citation bank

## Change Count
- **Structural:** 7 changes (Abstract, Introduction, preamble, routing foreshadowing ×2, timeline move, Conclusion)
- **Rhetorical:** 2 changes (Title, Introduction hook)
- **Language:** 2 changes (Introduction preview, Related Work polish)
- **Evidence:** 1 change (routing foreshadowing)
- **None:** 15+ units preserved as-is
