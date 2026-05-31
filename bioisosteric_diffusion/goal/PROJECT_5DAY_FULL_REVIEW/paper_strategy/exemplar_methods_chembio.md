# Exemplar Methods Sections in Computational Chemistry & Biology
## Comparative Analysis and Template Recommendation

**Date:** 2026-05-29
**Context:** Methods-section analysis for bioisosteric diffusion paper targeting BIB / JCIM / Bioinformatics

---

## 1. Paper-by-Paper Methods Analysis

### 1.1 AlphaFold (Jumper et al., 2021, Nature)

**Reference:** Jumper, J. et al. "Highly accurate protein structure prediction with AlphaFold." *Nature* 596, 583--589 (2021). DOI: 10.1038/s41586-021-03819-2

#### Section Structure and Flow

The main text Methods is a concise summary (~3 pages). The complete methodological exposition is in a **~60-page Supplementary Information (SI)** document organized as:

```
SI Section 1:  Problem overview and notation
  - SS1.1: Notation and conventions (tensor shapes, mathematical symbols)
  - SS1.2: Data pipeline (9 sub-subsections: parsing, genetic search, template 
           search, training data mix, filtering, MSA block deletion, MSA clustering,
           residue cropping, feature extraction)
  - SS1.3: Self-distillation dataset
  - SS1.4: Inference algorithm (formal pseudocode, Algorithm 2)
  - SS1.5: Input embeddings
  - SS1.6: Evoformer (6 sub-modules)
  - SS1.7: Extra MSA and template embedding
  - SS1.8: Structure module (Invariant Point Attention, iterative refinement)
  - SS1.9: Auxiliary heads
  - SS1.10: Recycling
  - SS1.11: Training details
  - SS1.12: Ablation studies
```

The structure flows from **inputs (data pipeline) -> representation learning (Evoformer) -> 3D structure generation (Structure Module) -> outputs (auxiliary heads, confidence).** Each step is causally motivated: "to enable X, we design Y."

#### Problem Formulation / Task Definition

The problem is defined in domain terms first, then given formal mathematical structure:

> "Protein structure prediction -- predicting the 3D coordinates of all heavy atoms in a protein given its amino acid sequence -- is one of the most important open problems in computational biology."

The formal framing is done through a **pairwise distance prediction lens**: the model predicts a distribution over pairwise distances (distogram), and for each residue, the position in the local frame defined by the backbone geometry. This is not a pure regression; it is structured output prediction with explicit geometric constraints.

#### Data / Benchmark Construction

Described in exquisite detail in SI SS1.2 (9 subsections covering the full pipeline):

- **Source databases:** PDB (training), Uniclust30/BFD (self-distillation), PDB70 (templates)
- **Filtering criteria:** Resolution < 9 Angstrom, length rebalancing, inverse cluster-size weighting
- **Self-distillation strategy:** 75% self-distillation predictions + 25% PDB known structures
- **MSA processing:** Block deletion, clustering, feature extraction with specific tensor shapes
- **Data splits:** CASP14 as blind test set, PDB structures clustered at 30% sequence identity

**Key practice:** Every tensor dimension is specified. Every filtering threshold is quantified. Every search tool version and database release is cited.

#### Model Architecture

The architecture is described through **a combination of pseudocode (Algorithm 2), formal equations, and architectural diagrams in the SI**. Sub-modules are described in dedicated subsections with explicit tensor shapes:

- **Evoformer:** MSA representation [N_seq, N_res, c_m=256], Pair representation [N_res, N_res, c_z=128]
- **Triangle multiplicative updates:** Enforce triangle inequality on pair representations
- **Invariant Point Attention (IPA):** Augments attention with 3D point distances in local frames
- **Recycling:** Whole-network output fed back recursively for N_cycle=4 passes

Each sub-module gets: (1) what it does, (2) why it is needed (geometric/biological motivation), (3) formal definition with equations, (4) tensor shapes for reproducibility.

#### Evaluation Protocol and Metrics

Metrics defined in both **domain terms** (RMSD, LDDT, TM-score, GDT) and **the model's internal metrics** (pLDDT, PAE):

> "We measure accuracy using the Local Distance Difference Test (LDDT)... A backbone accuracy of 0.96 Angstrom RMSD95 for CASP14 targets."

Evaluation includes: CASP14 blind-test results, ablation studies, per-residue confidence calibration, and proteome-wide application.

#### Prose-to-Equation-to-Table Ratio

**Heavy on equations** for the architecture (IPA, triangle attention, FAPE loss) but **heavy on prose for the data pipeline** (tool descriptions, curation rules). Approximately 40% equations, 40% prose, 20% algorithms/pseudocode in the SI. The main text is reverse: ~60% prose, ~15% equations, ~25% figures.

#### Clarity and Reproducibility Techniques

- Explicit tensor shapes in every description
- Versioned tool references (JackHMMER, HHBlits version numbers)
- Pseudocode for the inference algorithm (Algorithm 2)
- Hyperparameter tables with learning rates, warmup schedules, dropout rates
- Ablation experiments isolating each design choice
- Self-distillation protocol spelled out end-to-end

---

### 1.2 ChemBERTa / ChemBERTa-2 (Chithrananda et al., 2020; Ahmad et al., 2022)

**Reference:** Chithrananda, S., Grand, G., Ramsundar, B. "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." arXiv:2010.09885 (2020). / Ahmad, W. et al. "ChemBERTa-2: Towards Chemical Foundation Models." arXiv:2209.01712 (2022).

#### Section Structure and Flow

Standard ML-paper structure with domain adaptation:

```
1. Introduction (motivation from drug discovery)
2. Related Work
3. Methods / Approach
   3.1: Model Architecture (RoBERTa variant, tokenization)
   3.2: Pretraining Strategy (MLM, MTR)
   3.3: Finetuning Protocol
4. Experiments (MoleculeNet benchmarks)
5. Discussion
```

The Methods section follows the **"what is the model, how is it trained, how is it evaluated"** sequence typical of ML papers, but each subsection includes domain-specific justification.

#### Problem Formulation

Framed as a **NLP-for-chemistry** problem -- treating SMILES strings as a language:

> "We propose to treat molecular structures as a language, leveraging BERT's bidirectional encodings to learn rich molecular representations through self-supervised pretraining on large unlabeled datasets."

**Notably:** The problem is formulated at the representation level (learn embeddings), not as direct property prediction. The downstream tasks (property prediction via MoleculeNet) are the evaluation, not the method itself.

#### Data / Benchmark Construction

Handled in an Experimental Setup section rather than a dedicated Methods subsection:

- **Pretraining data:** 77M unique canonical SMILES from PubChem (with subsets of 100K, 250K, 1M, 5M, 10M, 77M)
- **Tokenization:** Byte-Pair Encoding (52K vocab) vs. custom SmilesTokenizer vs. SELFIES -- compared empirically
- **Evaluation data:** MoleculeNet benchmarks (BBBP, ClinTox, HIV, Tox21, BACE, Clearance, Delaney, Lipophilicity)
- **Data splitting:** Scaffold split (80/10/10) via DeepChem
- **Canonicalization:** All SMILES canonicalized and globally shuffled

**Contrast with AlphaFold:** Much less detail on data curation. No explicit "Data Curation" section -- this would be flagged as a weakness by JCIM standards.

#### Model Architecture

Standard RoBERTa architecture with two domain-specific adaptations:

- **Base:** HuggingFace RoBERTa (6 layers, 12 attention heads, ~40-50M parameters for ChemBERTa-2)
- **Tokenization:** Character-level SMILES dictionary (591 tokens for ChemBERTa-2, vs. 52K BPE for original)
- **Key difference from standard RoBERTa:** Vocabulary redesign for chemical tokens (atoms, bonds, brackets, digits)
- **Architecture search:** 5 best configurations from hyperparameter sweep

> "We use a RoBERTa architecture... implemented using the HuggingFace transformers library. For tokenization, we employ a character-level SMILES dictionary consisting of 591 tokens..."

#### Evaluation Protocol and Metrics

- **Metrics:** ROC-AUC for classification tasks
- **Protocol:** Fine-tune linear classifier on frozen pretrained representations
- **Early stopping:** 100 epochs max, patience = 1 full pass through data
- **Baselines:** Graph-based models (GCN, Weave, MPNN), fingerprint-based methods

#### Prose-to-Equation-to-Table Ratio

**Prose-dominant** with minimal equations. The MLM objective is described in words rather than mathematical expectation notation. The main technical depth is in the data scaling analysis and hyperparameter tables.

#### Contrast with Conference ML Papers

Unlike a NeurIPS paper, ChemBERTa does not formally define the MLM loss with equations, does not prove any theorems, and spends more space on practical design choices (tokenization, dataset construction, training efficiency). This is characteristic of domain applications in journal venues.

---

### 1.3 Neural Message Passing for Quantum Chemistry (Gilmer et al., 2017, ICML)

**Reference:** Gilmer, J. et al. "Neural Message Passing for Quantum Chemistry." *ICML 2017* (PMLR 70:1263-1272).

#### Section Structure and Flow

This is a **pure ML conference paper** with a domain application. Structure:

```
1. Introduction (motivation + MPNN overview)
2. MPNN Framework (formal definition)
3. Existing Models as MPNNs (unification)
4. New MPNN Variants (contributions)
5. Experiments (QM9)
6. Related Work
7. Conclusion
```

The Methods is woven into Sections 2-4. Section 2 is the framework definition, Section 4 has the novel variants.

#### Problem Formulation

Very compact, formal notation:

> "We consider supervised learning on graphs. Given a dataset of graphs G and labels y, we consider learning a function f that maps a graph G to a label y_G..."

The task is **regression of quantum mechanical properties** from molecular graphs. The molecule is represented as an undirected graph G with node features x_v (atom properties) and edge features e_{vw} (bond properties).

This is the **standard ML framing**: learn a function f: G -> y.

#### Data / Benchmark Construction

Handled briefly:

> "Our experiments are on the QM9 dataset... which consists of 134k molecules with 13 quantum mechanical properties computed using DFT."

The paper does **not** have a data curation section. Data is treated as a fixed benchmark. This would be unacceptable for JCIM.

#### Model Architecture

The MPNN framework is defined through three core equations (the "trinity" of MPNN):

**Message function:**
> m_v^{t+1} = sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw})

**Update function:**
> h_v^{t+1} = U_t(h_v^t, m_v^{t+1})

**Readout:**
> \hat{y} = R({h_v^T | v \in G})

Then a table maps existing models (Duvenaud, Li, Battaglia, Kearnes, Kipf) onto this framework. Novel variants (Edge Network, Master Node, Set2Set, Multiple Towers) are described in prose with minimal additional equations.

#### Evaluation Protocol and Metrics

- **Metric:** Mean Absolute Error (MAE)
- **Threshold:** Chemical accuracy (1 kcal/mol for energy properties)
- **Protocol:** Train/val/test split on QM9
- **Result reporting:** All 13 targets, with best results in bold, method comparison table

#### Prose-to-Equation-to-Table Ratio

**Equation-heavy in the MPNN definition**, then **table-heavy for model unification**. Approximately 40% equations, 30% prose, 30% tables. Very compact (10 pages total).

#### Key Differences from ChemBio Journals

- **No data section** -- assumes the benchmark is known
- **No domain explanation** -- the meaning of QM9 properties (HOMO, LUMO, gap, etc.) is not explained
- **No interpretability analysis** -- no analysis of what the model learns
- **Theorem-free** despite being a conference paper -- the contribution is a framework, not a proof
- **Code not required** at submission (though it was open-sourced)

---

### 1.4 Kipf & Welling GCN (2017, ICLR)

**Reference:** Kipf, T.N. & Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR 2017*.

#### Section Structure and Flow

Exceptionally compact:

```
1. Introduction
2. Fast Approximate Graph Convolutions
   2.1 Spectral Graph Convolutions (equations 3-5)
   2.2 Layer-Wise Linear Model (equations 6-8)
3. Semi-Supervised Node Classification
4. Experiments
5. Discussion
```

The Methods is Section 2 only (~2 pages in an 8-page paper).

#### Problem Formulation

> "We consider the problem of classifying nodes in a graph, where labels are only available for a small subset of nodes."

This is pure ML framing with no domain context. The graph is abstract (not specifically molecular).

#### Data

> "We evaluate our model on three standard citation network datasets: Cora, Citeseer, and Pubmed... as well as a knowledge graph dataset (NELL)."

Brief. No curation. Standard benchmarks.

#### Model Architecture

Derived step by step from first principles:

| Step | Equation | Rationale |
|------|----------|-----------|
| Spectral convolution | g_theta * x = U g_theta(Lambda) U^T x | Intractable O(N^2) |
| Chebyshev truncation | g_theta * x ~ sum theta_k T_k(~L) x | K-localized, O(E) |
| K=1 linearization | g_theta * x ~ theta_0 x - theta_1 D^{-1/2} A D^{-1/2} x | Reduces parameters |
| Parameter tying | g_theta * x = theta (I + D^{-1/2} A D^{-1/2}) x | Prevents overfitting |
| Renormalization | I + D^{-1/2} A D^{-1/2} -> ~D^{-1/2} ~A ~D^{-1/2} | Numerical stability |
| Final form | H^{(l+1)} = sigma(~D^{-1/2} ~A ~D^{-1/2} H^{(l)} W^{(l)}) | Scalable, effective |

This is the **paradigm of ML methods writing**: start from the general theory, apply successive simplifications justified by computational or statistical arguments, arrive at a simple and clean final form.

#### Prose-to-Equation-to-Table Ratio

**Extremely equation-dense.** Section 2 is primarily equations with linking prose. The paper defines 6 numbered equations in ~2 pages. Minimal tables (results table only).

#### What Makes This Effective

The method is presented as an **inevitable derivation**: each step follows from the limitations of the previous one. The reader arrives at the GCN layer as the natural conclusion, not as an ad-hoc design. This is the gold standard for method motivation in ML papers.

---

### 1.5 JCIM Exemplar: "Combining Group-Contribution Concept and Graph Neural Networks" (Aouichaoui et al., 2023)

**Reference:** Aouichaoui, A.R.N. et al. *J. Chem. Inf. Model.* 2023, 63(3), 725-744. DOI: 10.1021/acs.jcim.2c01091

#### Section Structure and Flow

```
1. Introduction
2. Methodology
   2.1 Data Curation
   2.2 Molecular Representation (graph construction, node/edge features)
   2.3 Model Architecture (AGC and GroupGAT)
       - Base GNN framework
       - Attention-based group contribution
       - Readout
   2.4 Training Procedure (hyperparameters, optimizer, etc.)
   2.5 Evaluation Protocol (metrics, cross-validation, baselines)
3. Results and Discussion
4. Conclusion
```

This is the **canonical JCIM structure**: separate data curation subsection, detailed molecular representation, explicit training and evaluation subsections. Note how the domain concept ("group contribution") is elevated to a central design principle, bridging classical chemistry with deep learning.

#### Data Curation

The paper includes a dedicated data curation subsection (demanded by JCIM):

> "The data used in this study were compiled from multiple sources... All compounds were standardized using a consistent protocol: (i) removal of salts and solvents, (ii) neutralization of charges, (iii) kekulization, (iv) canonicalization of tautomers. Compounds with ambiguous stereochemistry were flagged and reviewed manually."

Every filter step is enumerated with a rationale. The final dataset sizes after each filtering step are reported. This level of transparency is the JCIM standard.

#### Model Architecture

Described with both **domain concepts** (functional groups, contribution additivity) and **deep learning terminology** (attention, message passing, readout). The paper bridges these worlds:

> "The GroupGAT model extends the classical group-contribution method by replacing fixed additive contributions with context-dependent attention weights... Each molecule's total property prediction is computed as a weighted sum of group contributions, where the weights are learned through graph attention."

This is the **ideal chemical ML methods writing**: the model is explained both in terms of what it does computationally AND what it means chemically.

#### Prose-to-Equation-to-Table Ratio

Moderate equations (4-5 numbered equations), heavy on tables (dataset statistics, hyperparameters, baseline results, ablation studies). Approximately 30% equations, 40% prose, 30% tables.

---

### 1.6 JCIM Exemplar: "Enhancing Molecular Representations Via Graph Transformation Layers" (Ren et al., 2023)

**Reference:** Ren, G.-P., Wu, Y.-J., He, T.-T. *J. Chem. Inf. Model.* 2023, 63(9), 2679-2688. DOI: 10.1021/acs.jcim.3c00059

#### Section Structure and Flow

```
2. Methods (Note: paper starts with Introduction as Section 1)
   2.1 Line Graph (mathematical definition)
   2.2 Feature Evolution (attention mechanism for atom features)
   2.3 Position Evolution (3D coordinate generation for line graph nodes)
3. Experiments
   3.1 Datasets (QM9, MD17, etc.)
   3.2 Implementation Details
   3.3 Baseline Methods
   3.4 Results
```

The Methods section is **compact** (3 subsections) but **conceptually dense**: each subsection introduces a graph-theoretic concept, defines it formally, then connects it to molecular representation.

#### Handling the Domain Balance

The paper maintains a careful balance:

> "Given a molecular graph G = (V, E), we define its line graph L(G) where each edge in E becomes a node in L(G)... This transformation allows the GNN to capture higher-order interactions -- specifically triple-atom relationships -- that are chemically meaningful as bond angles and torsion angles."

The mathematical definition (line graph from graph theory) is immediately followed by the chemical interpretation (triple-atom relationships = bond angles). This pattern repeated throughout the methods.

---

## 2. Comparative Analysis

### 2.1 How Bio/Chem Papers Differ from Pure CS Papers in Methods Style

| Dimension | Pure CS / ML Conference | Chem/Bio Journal |
|-----------|------------------------|-------------------|
| **Problem framing** | Abstract, mathematical | Domain-first, then formal |
| **Data section** | Minimal (cite benchmark) | Mandatory data curation subsection |
| **Model description** | Equations first, brief prose | Prose->equations->prose pattern |
| **Reproducibility** | Implicit (same math implies same result) | Explicit (tool versions, thresholds, architectures) |
| **Domain concepts** | Not explained | Explained and motivated |
| **Evaluation** | % improvement over SOTA | Domain metrics + statistical rigor |
| **Interpretability** | Not required | Increasingly required |
| **Applicability domain** | Not discussed | Required for models |
| **Theorems/proofs** | Common (theoretical contribution) | Rare (empirical contribution) |
| **Length** | Compact (8-10 pages) | Comprehensive (12-20+ pages) |
| **Code/data availability** | Encouraged, often optional | Required at submission (for most venues) |

### 2.2 What JCIM/Bioinformatics Expects Differently from NeurIPS/ICML

**JCIM-specific requirements (from author guidelines and 2015 editorial):**

1. **MANDATORY: Data Curation Section.** "Any paper without a section on data curation" is flagged as less likely to succeed in review.

2. **MANDATORY for models: OECD Compliance.**
   - A defined endpoint (what are you predicting?)
   - An unambiguous algorithm (enough detail to reimplement)
   - A defined domain of applicability (when does your model work?)
   - Appropriate measures of goodness-of-fit, robustness, and predictivity
   - Mechanistic interpretation where possible

3. **REQUIRED: Data Availability Statement** (ACS Research Data Policy Level 2).

4. **REQUIRED: Software Availability** -- open-source code at submission for methodological papers.

**Bioinformatics journal (Oxford) specific requirements:**

1. **Section structure:** Introduction -> System and Methods -> Algorithm -> Implementation -> Discussion
2. **Algorithm subsection:** Must include complexity analysis and parameter sensitivity testing
3. **Reproducibility:** Source code freely available at stable URLs at submission
4. **ML papers require:** Training dataset composition, cross-validation sets, independent test set NOT used during training (reporting only CV error is insufficient)
5. **Homology handling:** For biological sequence data, must address homology between training/test sequences
6. **Length limit:** Original papers ~5000 words

### 2.3 Key Differences Summarized

| Aspect | NeurIPS/ICML | JCIM | Bioinformatics |
|--------|-------------|------|----------------|
| Methods section required | Yes (compact) | Yes (comprehensive) | Yes (split into System&Methods, Algorithm, Implementation) |
| Data curation subsection | No | Mandatory | Implicit |
| Applicability domain | No | Mandatory (models) | Expected |
| Reproducibility standard | "should release code" | "must release code" | "must release code at submission" |
| Evaluation rigor | Benchmark ranking | Domain metrics + statistics | Independent test set required |
| Interpretability | Optional | Expected | Expected |
| Length | 8-10 pages | ~12 pages typical | ~5000 words |

---

## 3. Recommended Patterns for Bioisosteric Diffusion Paper

### 3.1 Structural Template

Based on the analysis, the recommended Methods section structure for our target venue is:

```
2. Methods
   2.1 Problem Formulation
       - Formal statement of the bioisosteric replacement task
       - Notation: molecular graphs, fragment definitions, replacement objective
       - Connection to drug discovery / medicinal chemistry context

   2.2 Data Curation [MANDATORY for JCIM]
       - Source databases (ChEMBL, ZINC, etc.)
       - Fragment extraction protocol
       - Quality filters (validity, diversity, synthesisability)
       - Train/val/test split strategy (scaffold split, temporal split, or similar)
       - Dataset statistics and coverage
       - Applicability domain definition

   2.3 Molecular Representation
       - Graph construction (atoms as nodes, bonds as edges)
       - Node features (atom type, hybridization, formal charge, etc.)
       - Edge features (bond type, ring membership, etc.)
       - Fragment representation (subgraph extraction, anchor identification)

   2.4 Model Architecture
       - 2.4.1 Overall Framework (diagram recommended, high-level flow)
       - 2.4.2 Fragment Encoder (GNN backbone: GIN, MPNN, or equivariant variant)
       - 2.4.3 Diffusion Model (forward/noising process, reverse/denoising process)
       - 2.4.4 Condition Integration (how the scaffold/context conditions generation)
       - 2.4.5 Output Heads and Decoding (atom/type prediction, bond prediction, validity enforcement)

   2.5 Training Procedure
       - Loss functions (with equations)
       - Optimization (optimizer, learning rate schedule, batch size)
       - Training hyperparameters
       - Computational resources
       - Validation during training

   2.6 Evaluation Protocol
       - Metrics: validity, uniqueness, novelty, synthesisability, docking score, etc.
       - Baseline comparisons (list of competing methods)
       - Statistical significance (multiple runs, confidence intervals)
       - Ablation studies
       - Case studies (qualitative examples)
```

### 3.2 Prose-to-Equation-to-Table Ratio Recommendation

| Section | Prose % | Equations % | Tables/Figures % | Notes |
|---------|---------|-------------|------------------|-------|
| Problem Formulation | 80 | 15 | 5 | Domain context first, then formalize |
| Data Curation | 90 | 0 | 10 | Narrative + tables |
| Molecular Representation | 60 | 20 | 20 | Feature definitions in tables |
| Model Architecture | 40 | 40 | 20 | Equations + diagrams |
| Training | 70 | 20 | 10 | Hyperparameter tables |
| Evaluation | 60 | 10 | 30 | Results tables, comparison plots |

**Overall target:** ~55-60% prose, ~20-25% equations/algorithms, ~15-20% tables/figures. This matches the JCIM exemplars.

### 3.3 Specific Writing Techniques to Adopt

**From AlphaFold:**
1. **Explicit tensor shapes** for neural network layers -- enhances reproducibility enormously
2. **Rationale-first presentation** -- "To enable X, we design Y" rather than "We use Y"
3. **Auxiliary supervision** as a training signal (multiple losses for different aspects)
4. **Recycling / iterative refinement** described as a natural way to improve outputs

**From Gilmer et al. (MPNN):**
1. **Unifying framework presentation** -- show how existing methods are instances of your framework
2. **Three-function decomposition** (message, update, readout) as a clean architectural template
3. **Table of model variants** -- maps choices to performance
4. **Virtual node / global features** as a design pattern

**From Aouichaoui et al. (JCIM):**
1. **Bridging domain and computational language** -- explain every method in both chemistry and ML terms
2. **Dedicated data curation section** with rationale for each filter
3. **OECD-aligned evaluation** with applicability domain

**From Ren et al. (JCIM):**
1. **Immediate domain interpretation** after every mathematical definition
2. **Compact but complete** subsections -- each one is self-contained

### 3.4 Patterns to AVOID

1. **No data curation section** -- this is the single biggest reason for desk rejection at JCIM
2. **Equations without chemical interpretation** -- every equation should be followed by what it means chemically
3. **Theorems without empirical connection** -- not expected at JCIM/Bioinformatics
4. **Ignoring applicability domain** -- must state where the method is valid and where it fails
5. **Insufficient baseline comparisons** -- need multiple methods covering different paradigms
6. **No code release** -- increasingly mandatory
7. **Reporting only CV error** -- JCIM and Bioinformatics require independent test set evaluation

### 3.5 Additional Recommendations

1. **Lead Method with Motivation:** Every method subsection should start with the chemical/biological question it answers, then the computational solution.

2. **Reinforce Terminology:** Define a term, use it consistently. E.g., "bioisosteric fragment replacement" should be used throughout.

3. **Connect to Domain Impact:** Frame the evaluation in terms meaningful to drug discovery: "Our method identifies bioisosteres that maintain target affinity while improving ADMET properties."

4. **Visual Architecture First:** A clear diagram of the full architecture at the start of the Methods section, with numbered components that correspond to subsection numbers.

5. **Algorithmic Pseudocode:** For the diffusion process (training and sampling), use pseudocode boxes (like AlphaFold's Algorithm 2). This is rare in JCIM but highly valued by reviewers.

6. **Explicit Validity Constraints:** In molecular generation, explicitly describe how you enforce chemical validity (valence constraints, ring aromaticity, etc.) -- this distinguishes serious methods from "generate-and-filter."

---

## 4. Quoted Passages to Guide Method Writing

From **AlphaFold** (the "rationale-first" approach):
> "The Evoformer block is designed to enable reasoning about the protein structure from both the sequence level (MSA representation) and the pairwise interaction level (pair representation), with bidirectional information flow between them."

From **Gilmer et al. (MPNN)** (the "three-function" decomposition):
> "The first is a message function M_t, the second is a vertex update function U_t, and the third is a readout function R. The message and update functions are applied during the message passing phase, while the readout function is applied during the readout phase."

From **Ren et al. (JCIM)** (the "domain-interpretation" pattern):
> "Given a molecular graph G = (V, E), we define its line graph L(G)... This transformation allows the GNN to capture higher-order interactions -- specifically triple-atom relationships -- that are chemically meaningful as bond angles and torsion angles."

From **Aouichaoui et al. (JCIM)** (the "data curation" pattern):
> "All compounds were standardized using a consistent protocol: (i) removal of salts and solvents, (ii) neutralization of charges, (iii) kekulization, (iv) canonicalization of tautomers."

From **ChemBERTa** (the "domain-as-language" framing):
> "We propose to treat molecular structures as a language, leveraging BERT's bidirectional encodings to learn rich molecular representations through self-supervised pretraining on large unlabeled datasets."

---

## 5. Summary: One-Page Cheatsheet

```
For BIB / JCIM / Bioinformatics Methods Section:

DO:
+ Dedicated Data Curation subsection (JCIM: mandatory)
+ Every equation followed by chemical interpretation
+ Explicit tensor shapes and feature dimensions
+ OECD-aligned evaluation (fit, robustness, predictivity)
+ Applicability domain statement
+ Open-source code at submission
+ Architecture diagram as section roadmap
+ Domain motivation before computational solution
+ Ablation studies isolating key design choices
+ Multiple baselines spanning different paradigms
+ Independent test set (not just CV)

DON'T:
- Omit data curation (JCIM: desk-reject risk)
- Present equations without chemical meaning
- Skip applicability domain
- Use non-standard metrics without domain justification
- Forget interpretability analysis
- Report CV error as sole evaluation
- Leave out computational resource details

Prose:Equation:Table ratio target = 55:25:20
```
