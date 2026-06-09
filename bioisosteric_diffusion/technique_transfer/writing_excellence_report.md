# Writing Excellence Report: AI × Biology × Computational Chemistry

Date: 2026-06-09

## Part 1: Three-Field Writing Audit

### Field A: AI/ML Conference Writing (NeurIPS/ICML/ICLR norms)

**Core norm**: Contribution-first. Abstract must quantify. Introduction takes a stand.

What works:
- Numbered contribution bullets in Introduction (reviewers form judgment here)
- Abstract ends with quantified result, not vague promise
- "Claim → Evidence → Limitation" triad in every section
- Active voice mandatory: "We propose" not "It is proposed"
- Honest limitations: now explicitly non-penalized by reviewer guidelines

What fails:
- Hedge language ("seems to", "may improve") without genuine uncertainty
- Missing ablations — the single most common rejection reason
- No error bars or significance tests — now a baseline expectation
- Acronym clutter — 2024 study found ~89% of NeurIPS acronyms used <10 times

**2024-2025 trend**: Readability crisis. Sensational language +50% since 2015. Explicit calls for readability standards.

**Key citation**: Meng et al. (2025), *J. Informetrics* — analysis of 10M+ articles. Ethos + pathos → citation boost. Logos-only → negative effect. Evidence alone does not persuade.

### Field B: Computational Chemistry / Cheminformatics (JCIM/JMC norms)

**Core norm**: Rigor in details. Reproducibility first. Fragment-level interpretability valued.

What works:
- Complete software versions, random seeds, hyperparameter grids
- Multiple validation layers (internal CV + external test + Y-randomization)
- Fragment-level SAR interpretation — identifying which substructures drive activity
- Clear separation of Results from Discussion (JCIM is strict on this)

What fails:
- Vague activity thresholding — must state exact criteria (pIC50 ≥ 6 or equivalent)
- Missing applicability domain analysis for QSAR models
- "Under standard conditions" / "using routine methods" — automatic rejection triggers
- Claiming superiority without paired statistical tests

**2024-2025 trend**: Interpretable ML gaining ground. ACES-GNN integrates explanation supervision into training. Lamens & Bajorath (Chem. Sci. 2026) formalize explainability vs. interpretability distinction.

### Field C: Drug Discovery / Medicinal Chemistry (JMC / Eur. J. Med. Chem. norms)

**Core norm**: SAR tables. Multi-property optimization. Structure guides argument.

What works:
- SAR tables: R-group → biological activity → LE → LLE → commentary
- Multi-property optimization: potency + selectivity + ADME + toxicity simultaneously
- Fragment-based rationale: which fragment contributes what, quantified
- Integration of computational prediction with experimental validation

What fails:
- Reporting only potency without selectivity/ADMET context
- "Visual inspection" without quantitative confirmation
- Ignoring context-dependence: bioisosteric success is target-specific
- Overgeneralizing from single-target findings

### The Three-Field Tension

| Dimension | AI/ML | Comp Chem | Med Chem |
|-----------|-------|-----------|----------|
| Contribution claim | Explicit, numbered | Implicit, woven into narrative | SAR-driven |
| Evaluation | Benchmark tables with Δ | CV + external test + Y-randomization | Multi-property SAR |
| Interpretation | Ablation studies | Fragment importance | Structure → activity reasoning |
| Honesty | Limitations section mandatory | Applicability domain | Context-dependent caveats |
| Audience | Other ML researchers | Comp chem + some med chem | Medicinal chemists |

## Part 2: Interdisciplinary Bridge Strategies

### The "Three-Audience Problem"

LBC-Ranker must speak to:
1. Computational chemists (methods, evaluation, reproducibility)
2. Medicinal chemists (fragment replacement intuition, SAR logic)
3. ML reviewers (ablation, baselines, significance, honest limitations)

### Bridge Strategy 1: The "Component Ladder" as Universal Rhetoric

The component ladder (Freq → CA → Blend → LBC) speaks to all three audiences simultaneously:
- To ML people: ablation architecture, stepwise evidence
- To comp chem people: quantitative comparison, CV-tuned baselines
- To med chem people: "track record vs. structural match" is intuitively grasped

### Bridge Strategy 2: "One Number, Three Interpretations"

Every key result should be stated once quantitatively, then interpreted in three languages:
> LBC improves over Blend by +0.055 OF-macro Hit@10 (p = 0.002).
> → ML interpretation: This incremental but consistent gain suggests feature-resolved representation provides a reliable edge.
> → Comp chem interpretation: Retaining individual structural dimensions (bit_corr, five physchem deltas) captures information lost in scalar blending.
> → Med chem interpretation: Learning when a historically reliable replacement fits the specific query fragment context improves ranking beyond simple popularity + similarity.

### Bridge Strategy 3: "Negation + Pivot" for Memorability

Nature Computational Science (2025) recommends:
> "Our data do not support the prevailing view that…; instead, they reveal a previously unrecognized…"

LBC-Ranker application:
> "A tuned frequency–content blend does not merely approximate LBC-Ranker; the residual +0.055 gap indicates that coarse blending discards information that feature-resolved compatibility preserves."

### Bridge Strategy 4: The Reviewer Fatigue Curve

Top reviewers make provisional decisions at three reading points:
1. Abstract last sentence → must give quantified answer
2. First paragraph of Results → must establish credibility without Methods
3. First paragraph of Discussion → self-disclose one limitation with path forward

LBC-Ranker check:
- Abstract: ✓ ends with "0.852 ± 0.032, +0.055 over Blend, p = 0.002"
- Results opening: ⚠️ Currently opens with Table 2 data — could be sharper
- Discussion opening: ✓ "modest but consistent" self-discloses

## Part 3: Storytelling Architecture for LBC-Ranker

### The Core Story (Three-Act Structure)

**Act 1 (Setup)**: Fragment replacement ranking matters. Existing approaches each address only half the problem. A natural question: "Can't I just blend frequency and similarity?"
**Act 2 (Obstacle)**: A tuned blend is strong (0.797) but still leaves a gap (+0.055). Why? Because blending compresses structural information into a single scalar.
**Act 3 (Resolution)**: LBC-Ranker retains individual structural dimensions. The +0.055 gain, though modest, is consistent (10/10 seeds). LR matches HGB — representation, not capacity, is the bottleneck.

### SUCCES Check (Made to Stick framework)

| Element | LBC-Ranker |
|---------|-----------|
| Simple | "Learn which candidates are both popular AND structurally right" ✓ |
| Unexpected | "A simple 9-parameter model matches a tree ensemble" ✓ |
| Concrete | "0.852 vs. 0.797 — feature resolution beats coarse blending" ✓ |
| Credible | 10-seed, inner CV, exact sign-flip test, tuning ledger ✓ |
| Emotional | Weak — could add: "For the 14 zero-label OFs, LBC still provides chemically sensible rankings" |
| Stories | The component ladder IS the story: "Why blending fails" |

## Part 4: Rhetorical Moves from Top Papers

### Move 1: "The Honest Baseline" (from MOLTOP, ACNet)
> "We do not claim that ML is unnecessary. We show that, under fair comparison, a carefully constructed simple baseline matches complex alternatives. This constrains where future complexity investments should be directed."

### Move 2: "The Diagnostic Experiment" (from Nature Methods papers)
> "To test whether X alone explains our results, we constructed a Y baseline that isolates the contribution of X..."

LBC: The Freq+CA blend IS this diagnostic experiment.

### Move 3: "The Modest Claim" (from high-impact computational papers)
> "The improvement is modest (+0.055) but consistent (10/10 seeds). In pragmatic terms, this means..."

This move is counterintuitively powerful — modest claims with strong evidence are believed; strong claims with weak evidence are rejected.

### Move 4: "The Open Question" (from Nature papers)
> "Whether additional features (3D shape, electrostatics) would amplify this gap remains an open question."

This frames the work as a foundation, not a finale.

## Part 5: Final Manuscript Prescription

### Title (current → recommended)
Current: "Support-Conditioned Substructure Matching for Top10 Fragment Replacement Ranking"
→ The word "Support-Conditioned" is AI jargon that med chem readers won't parse.
→ Consider: "Learning When Replacement Track Records Transfer Across Fragment Contexts" (more narrative, less jargon)

### Abstract Structure (current → recommended pattern)
Current: Problem → Existing gap → Blend finding → Method → Results → Ablation → Practical note
→ Good, but the "Blend finding" could be sharper as a rhetorical question-answer pair.

### Introduction Structure
Current: Task → Challenge → Prior work → Gap → Method → Results preview
→ Add a rhetorical question: "Can't we just blend frequency and similarity?" at the gap point. This engages the skeptical reader.

### Results Structure
Current: Primary → Component Ladder → Model Capacity → Ablation → Tuning → Query-weighted
→ Good flow. Component Ladder answers the rhetorical question from Introduction.

### Discussion Structure
Current: Why it works → Retrieval vs. compatibility → Representation vs. capacity → Practical → Limitations → Future
→ Add a "What LBC-Ranker does NOT claim" subsection to preempt reviewer over-interpretation.

### Key Quotable Sentences to Add

1. "A tuned frequency–content blend is already a strong baseline (0.797). LBC-Ranker's additional +0.055 comes from learning which structural dimensions matter, not from the mere inclusion of frequency as a feature."

2. "The improvement is modest in magnitude but consistent in direction: LBC-Ranker outperforms the blend in all 10 seeds, indicating that feature-resolved compatibility provides a reliable, if incremental, gain."

3. "LR and HGB achieve parity (0.852 vs. 0.851), consistent with the growing body of evidence that carefully constructed simple models match complex ones for fingerprint-based molecular tasks."

4. "To our knowledge, this is the first evaluation of supervised candidate ranking under an OF-level holdout protocol with cross-validated baseline tuning for fragment replacement."
