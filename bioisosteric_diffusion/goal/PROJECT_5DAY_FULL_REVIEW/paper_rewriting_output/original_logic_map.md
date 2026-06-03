# Original Logic Map

Maps the existing manuscript in reading order. Each unit diagnosed against confirmed motivation (Option A: benchmark arc with prior_ranks shortcut audit as secondary thread).

---

## Title
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| Title | "Leakage-Controlled Closed-Vocabulary Fragment Replacement Ranking with Candidate-Level Scoring" | — | States three key terms: leakage-controlled, closed-vocabulary, candidate-level scoring | Long (16 words). Misses prior_ranks shortcut finding in title. Could shorten or add subtitle. | Rewrite — consider shortening to ~10 words or adding "and Feature Audit" subtitle |

## Abstract
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| Abstract ¶1 | Full paper summary: benchmark, 82-feat scorer, 77-feat scorer, prior_ranks removal, A4C caveat | Top-10 numbers, CIs, Δ values | Strong — all key numbers present with CIs | Dense single paragraph. Prior_ranks removal narrative is submerged within flow of numbers. Could sharpen: state benchmark contribution, then the deletion finding, then the locked-result caveat, then A4C scope. | Rewrite — restructure to highlight two-part contribution (benchmark + shortcut audit), move A4C to supporting role in abstract |

## 1. Introduction
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| ¶1 (lines 11-12) | Hook: fragment replacement in lead optimization is important but hard to benchmark | — | Opens with practical need → computational gap | Generic. "Remains difficult to benchmark rigorously" is vague. No specific scale/impact hook. | Rewrite — computational framing per JCIM norm: open with chemical space scale or algorithmic challenge |
| ¶2 (lines 13-15) | Three difficulties: data leakage, strong baselines, signal integration | ChEMBL, MMP literature | Good — frames motivation as three technical difficulties | Three difficulties enumerated but not prioritized. "Non-trivial" is weak. | Rewrite — sharpen to two core difficulties: (1) leakage makes benchmarks deceptive, (2) shortcut features degrade under distribution shift. Drop third difficulty or merge. |
| ¶3 (lines 15-16) | Two contributions + two safeguards | — | Direct motivation link | "Supporting safeguards" is defensive framing. Contributions listed as items, not integrated narrative. | Rewrite — restate as: (1) benchmark contribution (primary), (2) shortcut audit finding (secondary). Frame safeguards as methodological rigor, not separate contributions. |
| ¶4 (lines 17-19) | Quantitative results summary | Top-10 numbers, CIs, rescue/lost | Strong evidence | Results paragraph in Introduction is effective but reads as data dump. Post-audit caveat correctly stated. | Keep with minor polish — tighten numbers, sharpen the "deletion not addition" framing |
| ¶5 (lines 19-20) | Four enumerated contributions | — | Lists contributions but dilutes the primary/secondary hierarchy | Items 3 (protocol separation) and 4 (A4C workflow) are methodological safeguards, not contributions. Item 1 (benchmark) and Item 2 (scorer + prior_ranks) are the actual contributions. | Rewrite — two contributions not four |

## 2. Related Work
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| ¶1 MMP section (lines 23-25) | Establishes MMP data and its scope limitation | ChEMBL, MMP refs | Background — sets up why supervised ranking is possible from structure data | Clear and concise. Good scope caveat ("do not encode activity continuity"). | Keep with minor polish |
| ¶2 Database tools (lines 26-28) | SwissBioisostere, CReM, NeBULA | Wirth 2013, Polishchuk 2020, Huang 2025 | Context — situates work among database-driven tools | NeBULA paragraph is dense. 311,543 rules and 5 categories are specific but may distract. | Keep — trim NeBULA detail slightly |
| ¶3 Deep learning (lines 29-31) | Ertl DNN, DeepBioisostere, GraphBioisostere | Ertl 2020, Kim 2025, Masunaga 2026 | Context — situates work among learning-based methods | Good coverage. Transition to "our work complements" is effective. | Keep with minor polish |
| ¶4 Rank fusion (lines 31-32) | Borda, RRF basics | Dwork 2001, Cormack 2009 | Methods context — explains fusion choice | Short and functional. | Keep |
| ¶5 Alerts (lines 33-34) | PAINS, Brenk, Helmke off-target | Baell 2010, Brenk 2008, Helmke 2025 | A4C context | Good integration of recent off-target reference. | Keep |

## 3. Methods (overall)
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| Methods preamble (lines 37-44) | Roadmap of Methods sections + timeline disclosure | — | Clear upfront about post-audit nature of prior_ranks removal | Paragraph is long and mixes roadmap with caveat. The caveat belongs in 3.5, not preamble. | Rewrite — move timeline disclosure to 3.5, keep preamble as clean roadmap |

## 3.1 Task Formulation
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| Formal definitions (lines 45-79) | Mathematical formalization of ranking task | — | Establishes rigorous notation for reproducibility | Notation is complete and self-contained. LaTeX is clean. | Keep — minor polish only |
| Metrics (lines 97-122) | Top-K, MRR, CI definitions | — | Defines evaluation framework | Good. Bootstrap CI definition is precise. | Keep |

## 3.2 Benchmark Construction
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| Data source + MMP (lines 126-129) | ChEMBL37K, MMP extraction, quality filters | ChEMBL, MMP algorithm | Establishes data provenance | Clear. PAINS/Brenk filtering caveat is properly stated. | Keep |
| Decoy construction (lines 130-131) | Negative example construction | — | Methodological detail | Functional. | Keep |
| Split design (lines 132-141) | Transform-heldout split, three evaluation tiers | — | Core methodological contribution | Good. Transform-heldout concept clearly explained. Three-tier structure is logical. | Keep — consider moving canonical analysis details (specific query counts) to a table footnote |
| Tables M1, M2 (lines 144-171) | Candidate matrix stats, leakage verification | Count data | Evidence for leakage control claim | Tables are clean and informative. | Keep |

## 3.3 Base Ranking Methods
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| Overview (lines 175-178) | Five baseline rankers intro | — | Establishes comparative baseline | Brief and informative. | Keep |
| Best-of-DE+HGB (lines 181-182) | Diagnostic bound definition | — | Important methodological guardrail | Clear. "Not a ceiling" repeated — good. | Keep |
| 3.3.1-3.3.4 (lines 183-260) | Five ranker descriptions | DE, HGB, Borda math | Methods completeness | All mathematical definitions are complete and precise. Score Blend description is detailed. | Keep — LaTeX is clean, no changes needed |

## 3.4 Candidate-Level Scorer
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| 3.4.1 Architecture (lines 263-283) | HistGB architecture + loss function + three design rationales | — | Core method contribution | Strong. Three reasons for HistGB are well-articulated. No-architectural-novelty caveat is honest. | Keep |
| 3.4.2 Feature families (lines 286-316) | Table M3 + feature descriptions | — | Methodological completeness | Table M3 is excellent — audit trail per family. Feature descriptions are thorough. Prior_ranks distinction from model_ranks is clear. | Keep — excellent as-is |
| 3.4.3 Training (lines 318-331) | CV protocol, hyperparameters, reproducibility | — | Reproducibility | Complete. GroupKFold, neg subsampling, frozen schema all documented. | Keep |
| 3.4.4 Categorical schema (lines 327-330) | Frozen schema alignment | — | Bug prevention | Important practical detail. Good to document. | Keep |

## 3.5 Feature Ablation: Prior_Ranks
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| 3.5.1-3.5.3 (lines 333-374) | Ablation design, mechanism, final set | Shortcut learning theory, fragment identity shift data | **Secondary contribution — central to Option A narrative** | This is the paper's strongest single finding. Currently positioned as methods detail (3.5) but deserves structural elevation. The shortcut learning interpretation is compelling and rigorous. 8-to-19 fragment shift is concrete mechanism. | **Elevate.** Move key mechanism insight to Results (Section 4.2 already does this). The Methods section should keep the technical ablation definition. Consider: stronger causal language in Section 3.5.2 — the mechanism is well-supported. |

## 3.6 Evaluation Protocol
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| 3.6.1-3.6.5 (lines 377-455) | Metrics, bootstrap, rescue/lost, protocol separation, leakage control | Bootstrap theory | Methodological rigor | Comprehensive. Protocol separation table (M4) is important guardrail. Five-part leakage control checklist is thorough. | Keep — minor polish on protocol separation table |

## 3.7 A4C Computational Review Proxy
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| 3.7.1-3.7.3 (lines 457-490) | Dual-mode workflow, provenance labels, scope | PAINS, Brenk, Helmke | Supporting workflow contribution | Well-scoped. G4 coverage limitation honestly stated. | Keep as supporting section — do not expand, do not overclaim |

## 4. Results
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| 4.0 preamble (lines 491-494) | Results roadmap | — | Navigation | Functional. | Keep |
| 4.1 (lines 497-527) | Primary blind result: Table 1 | Top-10, CIs, Δ values | **Primary evidence** | Strong. Table 1 is comprehensive. Narrative separates 82-feat (prospective) from 77-feat (post-audit) correctly. | Keep — minor polish on prose flow |
| 4.2 (lines 527-548) | Prior_ranks ablation: Table 2 | Δ+0.0393, CI, mechanism | **Secondary contribution evidence** | Strong. This is the paper's most novel finding. Well-argued. | Keep — possibly add a figure showing per-fragment improvement |
| 4.3 (lines 550-571) | Rescue/lost analysis: Table 3 | Counts, ratios, arithmetic verification | Reliability evidence | Good. 5.9× reduction in lost queries is compelling. | Keep |
| 4.4 (lines 574-578) | Schema alignment audit | Bug→fix narrative | Implementation integrity | Useful but brief. | Keep — consider moving to SI if space-constrained |
| 4.5 (lines 580-597) | A4C alert rates: Table 4 | Coverage, alert rates | Supporting evidence | Appropriately caveated. G4 coverage limitation restated. | Keep as supporting section |

## 5. Discussion
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| 5.1 Principal Findings (lines 600-605) | Three-point summary | Top-10, Δ, CIs | Recapitulation of results | Good synthesis. Post-audit caveat repeated appropriately. | Keep |
| 5.2 Candidate-level > query-level (lines 607-616) | Why routing fails, scoring succeeds | AUC 0.61 router, architecture insight | **Elevated secondary thread (Arc D)** | Most interesting discussion paragraph. Design principle is testable. Currently in Discussion only — should be foreshadowed in Methods/Results. | **Elevate** — add brief mention in Section 3.3 (why not query-level routing) and Section 4.1 (add routing failure as context). Keep the Discussion analysis. |
| 5.3 Prior_ranks as shortcuts (lines 613-620) | Mechanism analysis | Fragment shift data, Geirhos framework | **Core mechanism finding** | Strong. Alternative explanations acknowledged. Borda structural role well-integrated. Design rule is testable. | Keep — this is the paper's most citeable single paragraph |
| 5.4 Dual-mode workflow (lines 621-627) | A4C interpretation, scope | — | Supporting | Appropriate scope. Three limitations clearly stated. | Keep |
| 5.5 Relationship to other approaches (lines 628-632) | Positioning vs. generation/database tools | — | Context | Good positioning. Open-vocabulary future work is natural extension. | Keep |
| 5.6 Limitations (lines 633-638) | Six specific limitations | — | Honesty / rigor | Excellent. Each limitation is concrete and scoped. | Keep — this is a model limitations section |

## 6. Conclusion
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| ¶1 (lines 640-643) | Summary: benchmark, scorer, prior_ranks deletion, routing insight | Top-10, Δ values | Restates contribution hierarchy | Good synthesis. However, the routing insight (candidate-level > query-level) appears in Conclusion but was only briefly foreshadowed in Results. | Rewrite — better alignment with Results/Discussion flow |
| ¶2 (lines 643-644) | Dual-mode workflow summary | — | Supporting | Brief. Appropriate scope. | Keep |
| ¶3 (lines 645-647) | Limitations summary → future work | — | Honesty | Good. Four limitations map to four future directions. | Keep |

## Data & Software Availability
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| (lines 650-653) | Repo, commit, tag, archival DOI plan | GitHub URLs, commit hashes | Reproducibility | Complete. | Keep |

## References
| Original Unit | Current Text Role | Evidence Used | Motivation Link | Problem | Keep / Move / Rewrite / Delete |
|---|---|---|---|---|---|
| (lines 656-676) | 18 references in ACS style | All citations | Complete | Good coverage. | Keep — may add 1-2 from citation bank for Discussion/Introduction breadth |

---

## Overall Assessment

**Strengths of current draft:**
1. Mathematical notation is precise and self-contained
2. Post-audit vs. prospective distinction is consistently maintained
3. Limitations section is excellent — 6 specific, concrete items
4. Statistical rigor (bootstrap CIs, paired differences, B=5000, arithmetic verification)
5. Protocol separation (Table M4) is clear guardrail
6. A4C scoping is honest and appropriate

**Weaknesses per confirmed motivation:**
1. Title too long, misses prior_ranks narrative
2. Introduction enumerates 4 contributions — should be 2 (benchmark, shortcut audit)
3. Prior_ranks finding (strongest result) is structurally submerged in Methods §3.5 rather than elevated
4. Candidate-level > query-level insight (Section 5.2) lacks Results foreshadowing
5. Abstract is dense — restructure to two-part contribution
6. Introduction hook could be stronger (JCIM computational framing, not medicinal chemistry framing)

**Rewrite priority (by impact):**
1. High: Title, Abstract, Introduction contributions reframing
2. High: Elevate prior_ranks narrative (structural, not content change)
3. Medium: Foreshadow routing insight in Methods/Results
4. Low: Prose polish across all sections per style_profile.md
5. Low: Add 1-2 selective citations for Discussion breadth
