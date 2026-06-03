# Exemplar Learning Dossier

**Target journal:** JCIM (Journal of Chemical Information and Modeling)
**Tier:** flash (3 papers)
**Role:** Exemplar Learner for PaperSpine

---

## 1. Exemplar Inventory

| # | Title (short) | Venue | Year | Why Selected |
|---|---|---|---|---|
| E1 | Bioisosteric replacement with deep neural networks (Ertl) | JCIM | 2020 | Direct subfield match: computational bioisostere prediction with ML, same journal, same validation paradigm |
| E2 | Computationally efficient algorithm to identify matched molecular pairs for large data sets (Hussain & Rea) | JCIM | 2010 | Foundational MMP algorithm paper referenced by the manuscript; prototypical JCIM methods paper with complexity analysis and large-scale validation |
| E3 | CReM: chemically reasonable mutations framework for structure generation (Polishchuk) | J. Cheminform. | 2020 | Field exemplar: context-aware fragment replacement generation, adjacent journal, complementary structural logic (generation vs. ranking) |

---

## 2. Structural Patterns

**Pattern 1 -- The gap-narrowing introduction.** All three exemplars open with a broad statement of importance (bioisosterism in drug discovery, MMP utility, molecular generation), then rapidly narrow to a specific technical gap. E1 and E3 accomplish this in 2-3 paragraphs: first paragraph establishes the problem domain, second paragraph surveys existing solutions and their limitations, third paragraph states the specific gap and the paper's contribution. The key move is that the gap is stated as a *technical limitation* (e.g., "existing approaches require explicit fragment definitions" or "current methods do not consider chemical context"), not as a generic need for better methods. This framing allows the Methods section to directly address the stated limitation.

**Pattern 2 -- Methods as direct response to the gap.** E2 is especially instructive: the title announces an algorithmic contribution, and the Methods section delivers a precise algorithmic description with pseudocode, complexity analysis (O(n log n) scaling), and explicit handling of edge cases. E1 similarly structures Methods around the stated limitation -- because the gap was "needs expert-crafted fragment definitions," the method description emphasizes automated feature learning. E1 and E3 both use a two-part Methods structure: (i) representation/descriptor generation, then (ii) model/algorithm design. This creates a logical flow where the reader understands what goes in before how it is processed.

**Pattern 3 -- Benchmark-first results.** All three follow a consistent results architecture: establish the benchmark protocol first (data source, split strategy, evaluation metrics), then present primary results on the benchmark, then ablation/variant analyses, and finally qualitative examples or case studies. E1 and E3 both include a table comparing performance against existing methods on matched tasks. E2 validates on multiple large databases (ChEMBL, PubChem) to demonstrate generalizability. Importantly, all three present negative results or limitations openly -- E2 discusses match rate limits, E3 discusses the tradeoff between chemical validity and novelty -- which builds credibility rather than diminishing it.

---

## 3. Rhetorical Patterns

**Opening technique -- the computational framing.** JCIM computational papers consistently avoid opening with biological motivation (e.g., "drug discovery is expensive"). Instead, they open with a *computational* framing: the size and complexity of chemical space, the need for automated analysis methods, or the scalability limitations of existing computational approaches. This signals to the JCIM readership that the contribution is algorithmic rather than domain-specific. E2's opening sentence immediately establishes the computational scale problem ("millions of compounds"), and E1's opening frames bioisosterism as an informatics challenge rather than a medicinal chemistry concept.

**Closing technique -- limitation-aware forward look.** Rather than claiming broad applicability, JCIM exemplars close with a focused assessment of what the method can and cannot do, followed by concrete future work that directly addresses the identified limitations. E3 is representative: the conclusion states the method's scope explicitly ("designed for R-group replacement, not scaffold hopping") and proposes three specific extensions. This creates a clean narrative arc from the introduction's gap statement through the method's resolution to the limitations that remain.

---

## 4. Language Patterns

The JCIM register is consistently precise and quantitative, with a strong preference for:
- **Nominal constructions** over active verbs for describing method behavior ("the algorithm achieves" rather than "we made the algorithm do").
- **Explicit operational definitions** of evaluation metrics (always define exactly how a metric is computed, even for standard measures).
- **Conservative comparative language** -- "competitive with" rather than "superior to," "comparable to" rather than "outperforms," unless backed by a formal statistical test.
- **Structured presentation** of benchmark results: tables with clear column headers and footnotes explaining deviations, figures that show distributions rather than single-point comparisons.
- **Avoidance of promotional language** -- no "novel," "first," "state-of-the-art" without justification; claims are supported by the reported numbers, not by adjectives.
- **Abbreviation discipline** -- every abbreviation is defined at first use and used consistently thereafter; no field-specific shorthand is assumed.
