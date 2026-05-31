# METHODS WRITING GUIDE — Scaffold-Conditioned Fragment Replacement

## Target: BIB / JCIM / J. Cheminformatics / Bioinformatics

---

## 1. OVERALL STRUCTURE TEMPLATE

### 1.1 Recommended Section Hierarchy

```
2. METHODS
  2.1 Task Formulation                           (1-1.5 pages)
  2.2 Benchmark Construction                     (1.5-2 pages)
  2.3 Base Ranking Methods                       (2-2.5 pages)
    2.3.1 Attachment-Frequency Ranker
    2.3.2 Dual Encoder Ranker
    2.3.3 Heterogeneous Graph Bridge (HGB) Ranker
    2.3.4 Borda Fusion and Score Blending
  2.4 Candidate-Level Scorer (THE MAIN METHOD)   (3-4 pages)
    2.4.1 Problem Setup and Feature Overview
    2.4.2 Feature Families (9 families, 77 features)
    2.4.3 Model Architecture: Histogram Gradient Boosting
    2.4.4 Training Protocol: GroupKFold and One-Shot Blind
    2.4.5 Template Alignment via cat_templates
  2.5 Ablation: The Prior_Ranks Removal          (1 page)
  2.6 Evaluation Protocol                        (1.5-2 pages)
  2.7 Computational Review Proxy (A4C)           (1-1.5 pages)
```

**Total Methods length target: 11-14 pages** (for JCIM/Bioinformatics, which allow longer methods; tighten to 9-11 for BIB).

**Depth rule of thumb:**
- Subsections at depth 3 (`2.4.1`) for implementation details.
- Use depth 4 (`2.4.3.1`) only when you need to describe a specific algorithmic variant (e.g., GroupKFold splitting strategy). Keep to 3 levels unless absolutely necessary.
- Each depth-3 subsection should be 3-8 paragraphs.

### 1.2 Opening Paragraph of Each Subsection — Must Convey

| Subsection | Opening paragraph must establish |
|---|---|
| 2.1 Task Formulation | The real-world problem (scaffold hopping), why it is framed as fragment replacement, the key formalism (query = old_fragment + attachment_sig, candidate set, closed vocabulary). Define the input/output mapping in 1-2 sentences. |
| 2.2 Benchmark Construction | What data source, why ChEMBL37K, what MMP extraction accomplishes (fragment pairs as supervision signal), how the split prevents leakage. The "ground truth generation" problem. |
| 2.3 Base Ranking | Why we need baselines (to contextualize main method), what hypotheses they test (frequency vs learned vs graph-structured signal). Each baseline is a one-paragraph mini-description. |
| 2.4 Candidate-Level Scorer | **This is the heart of the paper.** Establish the motivation: base rankers are insufficient; we need a fine-grained, feature-rich, chemistry-aware scorer. Signal that 77 features across 9 families provide exhaustive coverage of the fragment replacement landscape. |
| 2.5 Ablation | State the question: "Which features matter most?" Lead with the finding (prior_ranks features are a generalization hazard), then explain why. |
| 2.6 Evaluation | What counts as a "correct" prediction, what the protocol measures, how uncertainty is quantified. |
| 2.7 A4C Proxy | Why a proxy (computational cost of full synthesis/testing), what A4C simulates, the dual-mode design. |

### 1.3 Where Equations vs Tables vs Prose Belong

| Content Type | Location | Example |
|---|---|---|
| Formal problem definition | Opening of 2.1 | Eq 1: `query = (f_old, sigma) -> R_cand` |
| Similarity/distance metrics | 2.4.2 (feature definitions) | Eq 2: fingerprint Tanimoto definitions |
| Score fusion formulas | 2.3.4 (Borda) | Eq 3: Borda score aggregation |
| Ablation metric delta | 2.5 | Small inline table or bar figure |
| 77 feature catalog | 2.4.2 | **Table 1** (see below) |
| Training hyperparameters | 2.4.3 | Inline prose or a short table (Table 2) |
| Evaluation metrics | 2.6 | Table 3 showing metric formulas |
| Design rationale | Everywhere, especially 2.4 and 2.5 | Prose paragraphs, not bullet lists |

**Critical rule:** Every equation and table must be *called out in prose* before it appears ("The candidate ranking problem is formalized in Equation 1"). Never drop an equation without preparation.

### 1.4 "Why" vs "What" Placement

| Component | "Why" (Design Rationale) | "What" (Implementation Detail) |
|---|---|---|
| Task definition | Why closed vocabulary? Why attachment signatures? | The exact formalism and notation |
| Benchmark split | Why transform-heldout? Why 4 decoy repairs? | The splitting algorithm, parameters |
| Feature engineering | Why 9 families? Why these descriptors? | Each feature's calculation |
| Model choice | Why HistGB and not XGBoost/DNN? | Hyperparameters, training config |
| Ablation | Why prior_ranks is dangerous | Exact performance delta |
| Evaluation | Why bootstrap CI? Why rescue/lost? | Bootstrap implementation details |

**Write "why" first, then "what".** Paragraph structure: (1) Problem/need, (2) Design rationale for our choice, (3) Implementation details. The rationale is what reviewers judge — the implementation is what readers need to reproduce.

---

## 2. WRITING PATTERNS TO STEAL

### 2.1 From ResNet / CNN Papers: Architecture Pipeline Clarity

**What to steal: The "data flow" narrative.**

ResNet papers don't dump the architecture as a static diagram — they walk the reader through the data transformation step by step: "The input passes through a 7x7 conv layer, then a 3x3 max pool. The main body consists of 4 stages, each containing residual blocks..."

**Application to our paper:**

```
"Given a query (f_old, sigma) and a candidate fragment f_cand, the scorer
computes 77 features organized into 9 families. First, a featurization
layer converts the molecular pair into structural descriptors (families
F1-F4). Next, interaction features capture the fragment-scaffold interface
(F5-F6). Query features encode the original fragment's properties (F7).
Performance features incorporate the base ranker scores (F8). Finally,
prior_ranks features capture per-query ranking context (F9, used only in
the ablation). These 77 features are passed to a histogram gradient
boosting model that outputs a single score s(f_cand | query)."
```

This is one paragraph. It gives the reader the *entire architecture at 30,000 feet* before any details.

**Pattern: Broad-to-narrow-to-broad.**
- Broad: overview paragraph (above)
- Narrow: one subsection per family (detail)
- Broad: summary table connecting everything (Table 1)

### 2.2 From the Transformer Paper ("Attention Is All You Need"): Reproducibility

**What to steal: Exhaustive specificity without narrative clutter.**

The Transformer paper is famous for being reproducible because each component is described with all necessary parameters in the prose, not hidden in appendices or code.

**Application to our paper:**

```
"The 77 features span 9 families (Table 1). Each feature is computed
from the molecular graph of the query, candidate, or their conjunction.
Table 1 reports the feature name, family, dimensionality, computation
source (RDKit descriptor name where applicable), and a one-sentence
definition."

"Features are standardized to zero mean and unit variance using the
training set statistics. No feature selection is applied — all 77 features
are presented to the model. The choice to retain all features is deliberate:
preliminary experiments showed that HistGB's built-in feature importance
mechanism effectively down-weights irrelevant dimensions, making
pre-filtering unnecessary and avoiding information loss."
```

**The Transformer trick for complex systems:** For each component, state:
1. What it does
2. Its dimensionality/shape
3. How it connects to other components
4. The exact configuration of any learned component

**Checklist for reproducibility of our scorer:**
- [ ] Feature calculation: enough detail to implement each from SMILES
- [ ] Model parameters: learning_rate, max_depth, n_estimators, subsample, min_samples_leaf
- [ ] Loss function: specify exact loss (e.g., `'squared_error'` with early stopping)
- [ ] Data split: GroupKFold details (what defines groups, how folds are constructed)
- [ ] Feature names listed in supplement at minimum

### 2.3 From AlphaFold: Domain Science + Computational Method Balance

**What to steal: Motivating every design choice with domain reasoning.**

AlphaFold succeeds because it doesn't just describe *what* the architecture is — it explains *why* each component reflects a biological insight (co-evolution, pairwise distance distributions, spatial structure).

**Application to our paper:**

**Don't write:**
"We use 77 features in 9 families: F1 (4 features), F2 (8 features)..."

**Write instead:**
"The 77 features encode domain knowledge about what makes a good fragment replacement. **Structural features (F1-F4)** capture the intrinsic properties of fragments: their size, shape, polarity, and functional groups. A good replacement must be sterically and electronically compatible with the binding site, and these features provide the first-pass filter. **Interface features (F5-F6)** describe how the fragment connects to the scaffold — critical because the attachment bond geometry constrains the fragment's spatial orientation. **Query features (F7)** capture properties of the original fragment being replaced, enabling the model to assess 'replacement difficulty' (a large, rigid fragment requires a replacement of similar complexity)."

This establishes *domain rationale* before *catalog*. The reviewer thinks "they understand chemistry" rather than "they threw 77 features at the wall."

### 2.4 From JCIM / Bioinformatics Papers: What Editors Expect

**What editors at these journals look for:**

| Expectation | How to satisfy |
|---|---|
| **Reproducibility** | Every parameter, split seed, and software version stated. Code publicly available. |
| **Domain grounding** | Every method choice justified by chemical/biological reasoning, not just "it worked better." |
| **Honest accounting** | Report failures as well as successes. The rescue/lost analysis is particularly important for credibility. |
| **Appropriate baselines** | Don't oversell. State baseline strengths before your method's improvements. |
| **Statistical rigor** | Bootstrap CIs, significance tests, multiple independent replicates. |
| **Clear separation of train/test** | Explicit description of what leakage was checked and how it was prevented. |
| **Supplementary backup** | Full feature list, full result tables, additional ablations in SI. |

**Bioinformatics/JCIM reviewers will catch:** (a) any ambiguity about split strategy, (b) missing parameter settings, (c) unsupported claims of "chemistry-aware" without evidence, (d) lack of comparison to reasonable baselines.

### 2.5 How to Present the 77-Feature Table Without Being Boring

**Do NOT do this:**
- A single massive table of 77 rows in the main text
- A dense paragraph listing every feature by name
- A figure attempting to visualize all 77 features

**DO this instead:**

**Step 1: Overview table (Table 1, main text) — summary only**

| Family | Abbrev. | Description | Count | Examples |
|---|---|---|---|---|
| Fragment topology | F1 | Ring and bond counts, graph invariants | 4 | NumRotatableBonds, RingCount, HeavyAtomCount |
| Fragment pharmacophore | F2 | Pharmacophoric properties | 8 | NumHDonors, NumHAcceptors, LogP, TPSA |
| Fragment electronic | F3 | Partial charges and electronic descriptors | 6 | GasteigerChargeMax/Min, NOCount |
| Fragment steric | F4 | Shape and volume descriptors | 5 | PMI1/PMI2/PMI3, RadiusOfGyration |
| Attachment interface | F5 | Bonds and geometry at attachment point | 10 | BondOrder, IsSingleBond, RingAtachment |
| Scaffold context | F6 | Scaffold properties at 1-3 bond radius | 12 | ScaffoldFreq, ScaffoldDonors, ScaffoldAcceptors |
| Query fragment | F7 | Original fragment being replaced | 7 | QueryHeavyAtoms, QueryLogP, QueryRingCount |
| Base ranker scores | F8 | Scores from each baseline ranker | 4 | AttachmentFreq, DualEnc, HGB, Borda |
| Prior ranks | F9 | Per-query ranking from base rankers | 5 | Rank_Pct, Score_Z, Rank_Diff |

**Total: 77 features (72 used in main model, 5 in ablation).**

Then in the supplement: **Supplementary Table S1** with all 77 rows, each containing: Feature Name, Family, Calculation Method, RDKit Descriptor (if applicable), Dimensionality, Import in Final Model.

**Step 2: One paragraph per family** (1-3 sentences each) in the main text explaining the *chemical rationale*:

"Family F5 (Attachment Interface) captures the chemical environment around the bond connecting fragment to scaffold. These features include the bond order, whether the bond is part of a ring system, and the scaffold atom type at the attachment point. A fragment must form a chemically valid attachment — these features encode the constraints."

**Step 3: Feature importance figure** (Figure X, main text) showing the top 20 features by importance, colored by family. This is where the 77 features become *visually comprehensible*.

### 2.6 How to Describe the Ablation

**Pattern: Hypothesis-confirmation pattern used throughout ML papers.**

```
"To assess the contribution of individual feature families, we perform
a leave-one-family-out ablation (Table X). The most striking result is
the effect of removing the prior_ranks features (F9): Mean Top-10
overall accuracy increases from XXX to YYY when F9 is excluded, with
similar improvements across all benchmark splits (p < 0.01, bootstrap
test). This indicates that prior_ranks features introduce a generalization
hazard: they encode per-query rank information from the base rankers,
creating a shortcut that memorizes ranking idiosyncrasies of the training
queries rather than learning general fragment replacement rules."
```

Key narrative elements:
1. **State what you did** ("leave-one-family-out")
2. **State the surprising result** ("accuracy increases when F9 removed")
3. **Explain why** ("generalization hazard", "shortcut learning")
4. **Contextualize** ("this is known in the ML literature as...")

Then immediately pivot to the final model:
```
"Based on this finding, the final model uses 72 features (F1-F8) and
excludes F9. The 5-feature removal simplifies the model while improving
generalization — a rare case where less is more."
```

**This is EXACTLY the narrative reviewers love.** It shows methodological maturity: you didn't just add features indiscriminately; you carefully tested their contribution and removed harmful ones.

---

## 3. CONCRETE EXAMPLES

### 3.1 Opening of the Candidate-Level Scorer Subsection (Section 2.4)

> The base ranking methods in Section 2.3 provide a broad but shallow assessment of replacement candidates: they capture frequency signals (Attachment-Freq), learned pairwise preferences (Dual Encoder), and graph structure (HGB), yet none model the fine-grained chemical constraints that distinguish plausible from implausible replacements. We address this gap with a candidate-level scorer that operates on the output of the base rankers, re-ranking candidates using 77 features spanning 9 chemically motivated families (Table 1). The scorer is a histogram gradient boosting model (HistGB; Ke et al., 2017) trained on the ChEMBL37K benchmark, which produces a single scalar score s(f_cand | query) for each candidate. By combining the broad coverage of base rankers with the chemical specificity of dense feature encoding, the scorer achieves substantially higher accuracy than any individual ranking method, as we show in Section 3.

**Why this works:**
- Opens with a problem statement ("broad but shallow")
- Frames the solution as filling a gap
- Gives the architecture in one sentence (HistGB + 77 features)
- Signals the result ("substantially higher accuracy")
- Cites the original HistGB paper (scholarly hygiene)
- Previews the evaluation section

### 3.2 The Prior_Ranks Removal Explanation (Section 2.5)

> Among the nine feature families, the prior_ranks features (F9) differ from all others in a critical way: they incorporate information from the candidate's rank within each query's candidate list, as assigned by the base rankers. Unlike the other 72 features—which are intrinsic properties of the molecules or their pairwise relationships—prior_ranks are query-relative: a rank of 3 means different things for different queries depending on the size and composition of the candidate pool. This query-specific nature creates a risk of overfitting to per-query ranking idiosyncrasies rather than learning general fragment preferences. When we perform leave-one-family-out ablation (Table X), removing F9 yields the largest improvement across all evaluation metrics: Top-10 overall accuracy rises from 0.XXX (with F9) to 0.YYY (without F9), a statistically significant gain (p < 0.01, bootstrap test of 5,000 replicates). We interpret this as evidence that the model, given easy access to per-query rank signals, relied on them as a shortcut instead of learning the underlying chemistry. The final model therefore excludes prior_ranks, using 72 features. This result underscores a general principle in feature engineering for molecular machine learning: features that encode position in a per-query ranking, while seemingly informative, can paradoxically harm generalization by encouraging the model to memorize training-set ranking patterns.

**Why this works:**
- Immediately identifies *what makes F9 special* (query-specific)
- Explains *why* this is dangerous (overfitting, shortcut learning)
- Gives the quantitative result with statistical rigor
- Interprets the result ("shortcut learning")
- States the model decision clearly
- Ends with a generalizable principle (memorable and insightful)

### 3.3 The Evaluation Protocol Description (Section 2.6)

> We evaluate all methods on per-query top-10 accuracy: for each held-out query in the test set, we consider a candidate correct if the true replacement fragment (the ground-truth MMP pair) is among the top 10 ranked candidates. This query-level evaluation follows standard practice in fragment replacement benchmarks (e.g., Kroll et al., 2022; Wang et al., 2023) and focuses on the practical use case where a medicinal chemist screens only the top handful of suggestions. To assess statistical significance, we compute 95% confidence intervals via bootstrapping with 5,000 replicates, resampling queries (not individual predictions) to preserve the query-level structure. Accuracies are reported as point estimates with bootstrap intervals throughout Section 3.

> In addition to mean accuracy, we report two complementary analyses. First, the rescue analysis measures cases where a base ranker fails (correct candidate outside its top 10) but our scorer succeeds (correct candidate inside top 10), quantifying the value added by scoring. Second, the lost analysis measures the reverse — cases where the base ranker succeeds but the scorer fails — providing an honest account of degradation. These analyses appear in Table Y and directly inform the practical recommendation: the scorer is best used as a post-processing filter on base ranker outputs, not as a standalone predictor. We further enforce protocol separation by restricting each method to its designated inputs: base rankers never use gold-standard fragment information, and the scorer never uses test-set statistics for feature normalization (normalization parameters are fixed from training data only).

**Why this works:**
- Defines the primary metric simply and with domain justification
- Connects to literature (shows awareness of field standards)
- Explains why bootstrap and how (resample queries, not predictions — shows understanding of data structure)
- Introduces rescue/lost honestly (shows maturity — not hiding failures)
- States the practical implication of the analysis
- Closes with protocol separation (shows methodological rigor)

---

## 4. ANTI-PATTERNS TO AVOID

### 4.1 What Makes Methods Sections Unclear

| Anti-Pattern | Why It Fails | Fix |
|---|---|---|
| **The feature dump** | "We computed 77 features: F1, F2, ..., F77." No structure, no rationale, just a list. | Group into families, explain chemical motivation for each family, use a summary table. |
| **Overspecified detail** | "We used Python 3.9.18 with scikit-learn 1.2.2 and numpy 1.24.3..." before any conceptual explanation. | Put implementation details (library versions, exact commands) at the end of each subsection or in supplement. Start with the concept. |
| **Underspecified detail** | "We trained a gradient boosting model." (Which variant? What hyperparameters? What data splits? What loss?) | You must specify enough that a graduate student in the field could re-implement. If it's not reproducible, it's not a method. |
| **Missing "why"** | Describing what was done without ever explaining why that choice was made. | Every significant design choice needs a one-sentence rationale. If you can't write the rationale, reconsider the choice. |
| **Equation without context** | Dropping Equation 3 with no lead-in or follow-up. | Every equation needs (1) a sentence introducing it, (2) the equation itself, (3) a sentence interpreting it. |
| **Passive voice overdose** | "The features were computed and the model was trained." | Mix active voice for clarity, especially when stating decisions: "We chose HistGB because..." |

### 4.2 Common Mistakes in Computational Chemistry Methods Papers

1. **Presenting a trained model without describing the data split.** This is the #1 mistake. Reviewers immediately ask: "How did you split? Was there leakage?" Our transform-heldout split and leakage verification *must* be clearly described and justified.

2. **Overclaiming "novelty."** Reviewer pet peeve: "We introduce a novel..." when the method is a standard model with standard features. Better: "We assemble a comprehensive feature set..." or "Systematically evaluate the contribution of..."

3. **Hiding negative results.** When you found the ablation where F9 hurts performance, that's a *strength* of the paper, not a weakness. It shows methodological honesty. Don't bury it. Don't avoid it.

4. **Inconsistent terminology.** Calling it "the scorer" in one place, "the re-ranker" elsewhere, "the candidate model" in a third. Pick ONE term and use it throughout. (Recommendation: "candidate-level scorer" or "dense feature scorer.")

5. **Unclear separation of train/validation/test.** If you use GroupKFold 5-fold, is each fold's validation set used for anything (early stopping, hyperparameter tuning)? If so, you need a held-out test set. If not, you're reporting cross-validation metrics which are less convincing. Be explicit.

6. **Forgetting to cite baseline methods properly.** If HGB is based on a published method or if you implement a variant of an existing model, cite it. Don't make reviewers wonder "have they heard of the R-group replacement literature?"

7. **Confusing "interpretable" with "simple."** HistGB feature importance =/= chemical interpretability. Don't claim "our model is interpretable" just because it uses tree-based features. If you make this claim, you need SHAP analysis or similar.

### 4.3 How NOT to Present Features/Training Details

**BAD (feature presentation):**

> "Our model uses 77 features: logP, TPSA, NumHDonors, NumHAcceptors, HeavyAtomCount, RingCount, NumRotatableBonds, FractionCsp3, NumHeteroatoms, NumSaturatedRings, NumAromaticRings, NumAliphaticRings, NumSaturatedHeterocycles, NumAromaticHeterocycles...[followed by 65 more in a wall of text]"

- No grouping, no rationale, no structure.
- Impossible to scan or remember.
- Buried the lead (what do these features *mean*?).

**GOOD (feature presentation):**

> "The 77 features fall into 9 families (Table 1). The first family, **Fragment Topology (F1)**, captures global structural properties of the candidate fragment: ring and bond counts, heavy atom count, and rotatable bond count. These features encode the 'molecular footprint' — a replacement should have similar steric bulk to the original, and F1 provides a first-order approximation of this constraint."

- Groups, names, and rationalizes.
- Each family has a chemical motivation.
- Table 1 provides the complete catalog without breaking the narrative flow.

**BAD (training details):**

> "We trained our model using 5-fold cross-validation with early stopping."

- Missing: What defines groups for GroupKFold? What early stopping metric? What patience? What seed? What was the validation strategy?

**GOOD (training details):**

> "We train the HistGB model using 5-fold GroupKFold cross-validation, where each fold holds out all candidates from a disjoint set of queries (grouped by query ID). Within each fold, 80% of queries form the training set and 20% form the validation set. Early stopping is applied with patience of 50 rounds, monitoring the validation loss (squared error). The final model is the one with minimum validation loss across all folds. Key hyperparameters (Table 2) were determined by a grid search on a separate development split; all results reported are from the fixed configuration."

- Complete. Reproducible. No ambiguity.

### 4.4 Additional Anti-Patterns Specific to Our Paper

| Anti-Pattern | Context | Correct Approach |
|---|---|---|
| Calling the candidate-level scorer "our method" | The base rankers are also our contribution — they're baselines but also part of the system | "Our proposed scorer" or "the dense feature scorer" |
| Treating rescue/lost as an afterthought | This is actually one of the most insightful analyses in the paper | Give it a prominent place, perhaps a separate paragraph or sub-table |
| Assuming familiarity with MMP | Not all JCIM/Bioinformatics readers work in fragment-based drug design | Define MMP extraction in 2-3 sentences (matched molecular pairs, cleavage rules, fragment recovery) |
| Using "decoy" imprecisely | Decoys in DUD-E (inactive molecules) vs decoys in our setting (attached to wrong scaffold) | Clarify: "decoy repairs" = known fragment attached to a non-cognate scaffold. Define once, use consistently. |
| Scattering the ablation across sections | Ablation findings should be one coherent narrative | Collect all feature-family ablation in one subsection (2.5), reference it from results |

---

## 5. SUPPLEMENTARY MATERIALS CHECKLIST

Content to defer to SI (to keep main Methods focused):

- [ ] Complete 77-feature table (Supplementary Table S1)
- [ ] Full GroupKFold fold composition and query counts per fold
- [ ] MMP extraction parameters (cleavage SMARTS, fragment size filters)
- [ ] ChEMBL37K preprocessing details (quality filters, salt stripping, stereochemistry handling)
- [ ] Dual Encoder architecture details (if too detailed for main text)
- [ ] Bootstrap implementation (if standard, just cite a reference; if custom, include pseudocode)
- [ ] All pairwise significance test results
- [ ] Hyperparameter grid search details
- [ ] Software environment and library versions

---

## 6. PARAGRAPH-LEVEL TEMPLATE FOR EACH SUBSECTION

Use this skeleton for every subsection:

```
P1: MOTIVATION (2-4 sentences)
    - What problem does this subsection solve?
    - Why is existing work insufficient?
    - What is our high-level approach?

P2-P3: DESIGN RATIONALE (3-6 sentences each)
    - What considerations drove our design?
    - What alternatives were considered? (optional, but strengthens paper)
    - Why did we make the choices we made?
    - Include the "why" of key parameters/thresholds.

P4-P5: IMPLEMENTATION (3-6 sentences each)
    - What exactly was done?
    - Algorithm, data flow, mathematical formalism.
    - Parameters, configurations, software versions.
    - Enough detail for reproduction.

P6 (optional): CONNECTION TO NEXT SUBSECTION (1-3 sentences)
    - How does this component feed into the next?
    - What question does the next subsection address?
```

For the candidate-level scorer (the most important section), expand to 8-10 paragraphs with more rationale paragraphs.

---

## 7. KEYSTONE SENTENCES — Ready-Made Openers for Each Section

Write these, then build the rest of the section around them:

| Section | Keystone Sentence |
|---|---|
| 2.1 Task | "We formalize fragment replacement as a ranking problem: given a query consisting of the original fragment and its attachment signature, select the correct replacement from a closed set of candidates." |
| 2.2 Benchmark | "To provide a large, realistic training and evaluation dataset, we extract 37,240 matched molecular pairs from ChEMBL33 and construct a transform-heldout split that prevents scaffold-based leakage between training and test queries." |
| 2.3 Base Rankers | "We implement four baseline ranking methods that represent distinct approaches to the fragment replacement problem, providing both comparative baselines and input features for our main scorer." |
| 2.4 Scorer | "The candidate-level scorer combines 77 chemically motivated features with a gradient boosting model to produce a fine-grained replacement score, addressing limitations of the coarse base rankers." |
| 2.5 Ablation | "Feature ablation reveals that a small set of features encoding per-query rank information paradoxically degrades generalization, leading us to adopt a 72-feature model for all main experiments." |
| 2.6 Evaluation | "We evaluate all methods using query-level top-10 accuracy with bootstrap confidence intervals, and introduce rescue/lost analysis to characterize where the scorer adds or loses value relative to base rankers." |
| 2.7 A4C Proxy | "To assess computational feasibility without exhaustive chemical synthesis, we adopt the A4C framework with a dual-mode workflow that categorizes predictions into G2/G3/G4 provenance levels." |

---

## 8. FINAL QUALITY CHECK

Before submitting, verify:

- [ ] Can a graduate student in computational chemistry re-implement our scorer from the Methods alone?
- [ ] Is the contribution of each component (base rankers vs scorer vs ablation) crystal clear?
- [ ] Are all domain terms (MMP, attachment signature, decoy repair, transform-heldout) defined on first use?
- [ ] Is there exactly one place where each significant concept is defined?
- [ ] Do the section transitions provide logical flow (not just "Next, we describe...")?
- [ ] Is the scoring formula clearly distinguished from ranking output?
- [ ] Is the connection between training data (ChEMBL37K) and evaluation protocol (bootstrap, rescue/lost) explicit?
- [ ] Would a reviewer who is skeptical about "yet another ML method for drug design" be convinced by the domain reasoning?
