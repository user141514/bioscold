# Integration Notes: Four New References for "A Dual-Mode Workflow for Scaffold-Conditioned Fragment Replacement Proposal via Neural-Empirical Rank Fusion"

---

## 1. NeBULA 2025 (Huang et al., Medicine in Drug Discovery)

### One-Paragraph Description for Related Work

NeBULA (Huang et al., 2025) is a web-based platform for novel drug design that provides the largest publicly available bioisosteric replacement database to date. Mining over 700 medicinal chemistry references, the platform catalogues 311,543 qualitative bioisosteric replacement reaction rules encoded as SMARTS/SMIRKS patterns, derived from a library of 20.8 million drug-like molecules yielding 2.86 million fragment replacement entries. The rules are organized into five categories -- Scaffold Hopping, Ring Equivalence, Linker-Isostere, R-Isostere, and General Bioisosteric Replacement -- and the system implements a novel heavy-atom quantity difference filter algorithm to improve replacement quality. NeBULA represents the most comprehensive rule-based bioisosteric resource currently available, substantially exceeding the coverage of earlier databases such as SwissBioisostere.

### Specific Claim/Findings to Cite

- 311,543 qualitative bioisosteric replacement rules extracted from >700 medicinal chemistry references
- 20.8 million drug-like molecules and 2.86 million fragment replacement library entries
- Five-category taxonomy of bioisosteric replacements (Scaffold Hopping, Ring Equivalence, Linker-Isostere, R-Isostere, General)
- Novel heavy-atom quantity difference filter algorithm

### Relationship to Our Work

| Dimension | NeBULA | Our Work |
|-----------|--------|----------|
| Paradigm | Database-driven, rule-based lookup | Closed-vocabulary ranking via feature-based scorer |
| Coverage | 311K rules, very broad | 150-152 fragment vocabulary, focused ranking |
| Task formulation | Web-based lookup and rule retrieval | Query-level candidate ranking with explicit evaluation |
| Derivation | Literature-mined SMARTS/SMIRKS rules | MMP-derived structure pairs from ChEMBL |
| Ranking | Implicit via quality filters | Explicit candidate-level scoring (77 features, HistGB) |

**Differences:** NeBULA is a comprehensive resource platform focused on *coverage and retrieval* of known bioisosteric relationships from the literature. It does not define a ranking task, does not perform learned scoring, and does not evaluate replacement proposal quality under controlled splits. Our work is narrowly focused on a specific ranking task with rigorous leakage-controlled evaluation, trading broad coverage for deep methodological validation.

**Similarities:** Both systems exploit empirical bioisosteric knowledge (NeBULA from literature mining, our work from MMP analysis). Both organize replacements into categories (NeBULA's five types, our provenance-based G2/G3/G4 strata).

### Citation Placement

| Where | Section | Context | Citation Text |
|-------|---------|---------|--------------|
| **Primary** | Section 2, Related Work, paragraph 2 (rule-based/database tools) | Alongside SwissBioisostere and CReM as a major database-driven resource | "More recently, NeBULA [Huang2025] has extended this paradigm to over 311,000 bioisosteric rules mined from the medicinal chemistry literature, organized into five replacement categories with a dedicated heavy-atom filter." |
| Secondary | Section 5.6, Limitations (closed vocabulary) | Acknowledging the broader coverage of database-driven alternatives | "Database-driven platforms such as NeBULA [Huang2025] achieve far broader coverage (311K rules) by mining the literature, but their task formulation -- retrieval from a rule database -- differs fundamentally from our ranking benchmark." |
| Secondary | Section 5.7, Future Work (vocabulary enrichment) | Multi-database expansion and coverage | "Multi-database expansion could enrich the fragment vocabulary substantially; NeBULA's 20.8-million-molecule library [Huang2025] suggests the scale of coverage that is achievable." |

---

## 2. GraphBioisostere 2026 (Masunaga et al., J Supercomputing)

### One-Paragraph Description for Related Work

GraphBioisostere (Masunaga et al., 2026) applies a Siamese graph neural network architecture with an MMGX attention-based global pooling encoder to predict bioisosteric replacements from whole-molecule inputs. Trained on 779,081 compound pairs derived from ChEMBL35 (108,607 compounds across 1,897 targets), the model processes the complete fragment-environment context rather than isolated substituents. A fusion module combines the twin graph embeddings through difference, product, or concatenation operations followed by an MLP classifier. The method demonstrates transfer learning capability by predicting target-specific potency changes for BACE, JNK1, P38, Thrombin, PTP1B, and CDK2, showing that graph-based representations capture activity-relevant features beyond structural similarity.

### Specific Claim/Findings to Cite

- Siamese GNN with MMGX encoder processing whole-molecule inputs (not isolated fragments)
- 779,081 compound pairs from ChEMBL35, 108,607 compounds, 1,897 targets
- Fusion module with three combination strategies (difference, product, concatenation) + MLP
- Transfer learning to target-specific potency change prediction across 6 therapeutically relevant targets
- Demonstrates that GNN representations capture activity-relevant information, not just structural similarity

### Relationship to Our Work

| Dimension | GraphBioisostere | Our Work |
|-----------|------------------|----------|
| Architecture | Siamese GNN (whole-molecule) | HistGB on 77 engineered features + Dual Encoder |
| Task | Binary classification (is this pair bioisosteric?) + potency change prediction | Closed-vocabulary ranking (which fragment is the correct replacement?) |
| Input | Whole molecule (fragment + scaffold context) | Isolated fragment + attachment signature |
| Training data | 779K compound pairs from ChEMBL35 | MMP pairs from ChEMBL37K |
| Activity linkage | Yes -- transfer learning to potency prediction | No -- structure-derived labels only |
| Evaluation | Random splits (likely) | Transform-heldout + secondary blind (leakage-controlled) |

**Differences:** GraphBioisostere operates on whole molecules and can predict activity changes, making it a more ambitious modeling framework. Our work deliberately constrains the task to closed-vocabulary ranking with controlled leakage evaluation, which GraphBioisostere does not address. GraphBioisostere's evaluation likely uses random splits, which our analysis shows can inflate generalization estimates.

**Similarities:** Both use ChEMBL-derived supervision. Both demonstrate that learned representations capture information beyond frequency statistics. Both find that combining multiple signals (their fusion module, our Borda fusion and candidate-level scorer) improves performance.

### Citation Placement

| Where | Section | Context | Citation Text |
|-------|---------|---------|--------------|
| **Primary** | Section 2, Related Work, paragraph 3 (learning-based methods) | Already cited as Masunaga2026; update with more specific detail | "GraphBioisostere [Masunaga2026] extends this paradigm with a Siamese GNN (MMGX encoder) that processes whole-molecule inputs and demonstrates transfer learning to target-specific potency prediction across six therapeutically relevant targets, suggesting that graph representations capture activity-relevant features beyond structural similarity." |
| Secondary | Section 5.6, Limitations (no direct comparison) | Alongside CReM and DeepBioisostere as methods with different task formulations | "GraphBioisostere [Masunaga2026] operates in a different task formulation -- open-vocabulary classification with whole-molecule inputs -- making direct comparison to our closed-vocabulary ranking benchmark non-trivial." |
| Secondary | Section 5.7, Future Work (richer representations) | GNN embeddings as a feature augmentation direction | "Incorporating GNN-based representations, such as the MMGX encoder used in GraphBioisostere [Masunaga2026], could capture whole-molecule context effects that our feature set does not currently model." |

**Note:** The paper currently cites this as "Masunaga2026: Masunaga Y, et al. 'GraphBioisostere: graph neural network-based bioisostere prediction.' Journal of Chemical Information and Modeling, 2026." The actual venue is the Journal of Supercomputing, Vol. 82, Article 132, 2026. The DOI is 10.1007/s11227-026-08232-y. The first author's given name may also differ (README says Sho Masunaga, current reference uses Y. Masunaga). These bibliographic details should be verified and corrected as needed.

---

## 3. Helmke et al. 2025 -- Data-Driven Off-Target Assessment (RSC Med Chem)

### One-Paragraph Description for Related Work

Helmke et al. (2025) present a KNIME-based workflow for the data-driven assessment of bioisosteric replacements and their influence on off-target activity profiles. The workflow integrates ChEMBL bioactivity data with matched molecular pair analysis and bioisosteric SMIRKS patterns to evaluate replacements across 88 safety-relevant off-targets, including GPCRs, ion channels, kinases, and transporters. The study introduces a suite of decision quality metrics -- accuracy (ACCR), specificity (STCR), sensitivity (SFCR), and discriminatory power (DCR) -- to quantify how well bioisosteric recommendations preserve or alter off-target engagement. A key case finding demonstrates that ester-to-secondary-amide replacement increases CHRM2 muscarinic receptor binding approximately tenfold, illustrating that structurally conservative replacements can produce substantial off-target liability changes.

### Specific Claim/Findings to Cite

- KNIME workflow integrating ChEMBL bioactivity data, MMP analysis, and bioisosteric SMIRKS
- 88 safety-relevant off-targets evaluated (GPCRs, ion channels, kinases, transporters)
- Decision quality metrics: ACCR, STCR, SFCR, DCR
- Key case finding: ester-to-secondary-amide replacement increases CHRM2 binding ~10x
- Demonstrates that structurally conservative replacements can produce large off-target effects

### Relationship to Our Work

| Dimension | Helmke 2025 | Our Work |
|-----------|-------------|----------|
| Primary focus | Off-target activity consequences of bioisosteric replacements | Ranking accuracy of replacement proposals |
| Data integration | ChEMBL bioactivity + MMP + SMIRKS | ChEMBL MMP pairs only (structure-derived) |
| Safety scope | 88 specific off-target proteins | A4C rule-based alerts (PAINS, Brenk) |
| Output | Per-replacement off-target risk profiles | Provenance-stratified alert rates (G2/G3/G4) |
| Validation | Bioactivity data (ChEMBL assay endpoints) | Computational alerts only (no activity data) |
| Workflow | KNIME pipeline | Dual-mode (Conservative + Exploration) with provenance labels |

**Differences:** Helmke et al. ground their assessment in actual bioactivity data from ChEMBL assays across 88 specific off-targets, giving their risk assessments an evidential basis in measured activity. Our A4C proxy uses only rule-based structural alerts (PAINS, Brenk), which are computationally cheap but have high false-positive rates and do not correspond to specific off-target proteins.

**Similarities:** Both systems address the same fundamental question -- how to evaluate the risk of bioisosteric replacements beyond simple structural similarity. Both use MMP-derived relationships as a foundation. Both organize their outputs into categorical risk tiers that inform decision-making.

### Citation Placement

| Where | Section | Context | Citation Text |
|-------|---------|---------|--------------|
| **Primary** | Section 2, Related Work, new paragraph 5 (computational alert/risk assessment) | Introduce as the most relevant prior work on data-driven bioisosteric risk assessment | "Beyond structural alert filters, Helmke et al. [Helmke2025] recently introduced a data-driven KNIME workflow that integrates ChEMBL bioactivity data with MMP analysis to evaluate bioisosteric replacements across 88 safety-relevant off-targets, demonstrating that even conservative replacements (e.g., ester-to-secondary-amide) can induce substantial off-target liability changes (~10x CHRM2 binding increase). Their decision quality metrics (ACCR, STCR, SFCR, DCR) provide a template for quantitative risk assessment beyond binary alert flags." |
| **Critical** | Section 3.7, Computational Review Proxy (A4C) | Direct support for the motivation behind A4C; strengthens the rationale for computational risk stratification | "The need for such stratification is underscored by recent data-driven analyses demonstrating that structurally conservative replacements can induce substantial off-target effects [Helmke2025] -- a finding that motivates explicit risk annotation even in the absence of full bioactivity data." |
| Secondary | Section 5.6, Limitations (A4C limitations) | Acknowledging that our A4C proxy lacks the bioactivity grounding of Helmke's approach | "Unlike the data-driven off-target assessment of Helmke et al. [Helmke2025], which grounds risk estimates in ChEMBL bioactivity data across 88 specific targets, our A4C proxy relies solely on rule-based structural alerts, which carry well-known high false-positive rates and lack target specificity." |
| Secondary | Section 5.7, Future Work (expert review) | Incorporating activity-linked validation akin to Helmke's approach | "A natural extension would follow the paradigm of Helmke et al. [Helmke2025] by grounding the computational review proxy in specific off-target bioactivity data rather than generic structural alerts, enabling target-specific risk stratification of proposed replacements." |

---

## 4. DeepBioisostere 2025 v2 (Kim et al., arXiv 2403.02706v2)

### One-Paragraph Description for Related Work

DeepBioisostere v2 (Kim et al., 2025) presents an end-to-end deep generative model for autonomous bioisosteric fragment selection and substitution, eliminating the need for pre-established substitution rules. The model simultaneously controls multiple molecular properties -- logP, drug-likeness (QED), synthetic accessibility (SA) score, and molecular weight -- during fragment replacement, enabling multi-objective optimization within a single generative framework. Applied to the optimization of SARS-CoV-2 Mpro inhibitors targeting the E166V mutant, the method demonstrates practical utility in a challenging drug-design context. By removing the requirement for pre-defined substitution vocabularies, DeepBioisostere v2 represents a fully generative alternative to rule-based and retrieval-based approaches.

### Specific Claim/Findings to Cite

- End-to-end generative model: learns fragment selection AND substitution jointly (no pre-defined rules)
- Multi-property optimization: logP, drug-likeness (QED), SA score, MW controlled simultaneously
- Applied to SARS-CoV-2 Mpro (E166V mutant) inhibitor optimization -- real-world case study
- Revision v2 (Dec 2025) likely includes improvements over the original 2024 version
- No pre-established substitution vocabulary needed (unlike our closed-vocabulary approach)

### Relationship to Our Work

| Dimension | DeepBioisostere v2 | Our Work |
|-----------|--------------------|----------|
| Paradigm | Generative (end-to-end fragment proposal) | Ranking (closed-vocabulary candidate selection) |
| Vocabulary | Open -- proposes novel fragments | Closed -- 150-152 fragments from training data |
| Rule dependency | None (learns substitution autonomously) | MMP-derived rules as supervision labels |
| Property control | Multi-objective (logP, QED, SA, MW) | Not addressed (ranking only) |
| Validation | Case study (SARS-CoV-2 Mpro) | Systematic benchmark (13,347 queries, bootstrap CI) |
| Leakage control | Not specified | Transform-heldout split + secondary blind protocol |

**Differences:** DeepBioisostere v2 solves a fundamentally different and arguably harder problem: it must generate novel fragments and decide where to attach them, without a pre-defined vocabulary. Our task is more constrained but more rigorously evaluated. DeepBioisostere's property control capability is entirely absent from our system. Conversely, our leakage-controlled evaluation and statistical rigor go beyond what DeepBioisostere reports.

**Similarities:** Both systems aim to support medicinal chemistry decision-making. Both recognize that properties beyond structural similarity matter (DeepBioisostere via multi-objective control, our work via the A4C proxy). Both can be seen as addressing different parts of the same workflow -- generative methods propose, ranking methods prioritize.

### Citation Placement

| Where | Section | Context | Citation Text |
|-------|---------|---------|--------------|
| **Primary** | Section 2, Related Work, paragraph 3 (learning-based methods) | As the leading generative approach to bioisosteric replacement | "DeepBioisostere v2 [Kim2025] represents a fully generative alternative, employing an end-to-end deep model that learns both fragment selection and substitution without pre-defined rules, and enables multi-objective control over physicochemical properties during replacement. Applied to SARS-CoV-2 Mpro optimization, it demonstrates practical utility but does not provide the leakage-controlled evaluation that is the focus of the present work." |
| Secondary | Section 5.6, Limitations (no direct comparison) | Explicitly named as a method with a different task formulation | "DeepBioisostere [Kim2025] addresses open-vocabulary fragment generation rather than closed-vocabulary ranking, precluding direct comparison on our benchmark." |
| Secondary | Section 5.7, Future Work (open-vocabulary extension) | As a candidate generative model for the scorer to re-rank | "The candidate-level scorer could serve as a re-ranking filter on candidates generated by models such as DeepBioisostere [Kim2025], combining the generative coverage of deep learning with the ranking precision of our feature-based scorer." |
| Secondary | Section 5.7, Future Work (multi-objective extension) | As inspiration for incorporating property control | "Incorporating multi-objective property control similar to DeepBioisostere [Kim2025] -- e.g., filtering or re-ranking candidates by logP, QED, and SA score targets -- would extend the scorer from a ranking tool to a design tool." |

**Note:** The paper currently cites the earlier version as "Kim2024: Kim H, et al. 'DeepBioisostere: discovering bioisosteres with deep learning.' Chemical Science, 2024." The v2 update (arXiv 2403.02706v2, Dec 2025) should be cited as the current version. Verify whether the journal publication has been updated or if the arXiv version is the appropriate reference. Consider citing both: "Kim et al. (2024/2025)".

---

## Summary: Suggested Citation Changes to the Current Draft

| Section | Current Text | Proposed Change |
|---------|-------------|-----------------|
| Section 2, para 2 | SwissBioisostere and CReM only | Add NeBULA as third database-driven tool |
| Section 2, para 3 | "deep generative models that select fragments for removal and insertion [Kim2024]" | Update to Kim2025 (v2); add GraphBioisostere GNN detail |
| Section 2 | (no paragraph on risk assessment) | Add new paragraph 5 with Helmke 2025 |
| Section 3.7 | PAINS + Brenk alerts only | Add Helmke 2025 citation as motivation for risk stratification |
| Section 5.6 | "no direct comparison to public methods" | Explicitly name DeepBioisostere (generative) and GraphBioisostere (GNN) |
| Section 5.7 | "activity-linked reconstruction" and "richer representations" | Reference DeepBioisostere for multi-objective and GraphBioisostere for GNN features |
| References | Kim2024 (original Chemical Science) | Update to Kim2025 (arXiv v2); verify publication status |
| References | Masunaga2026 (JCIM as venue) | Update to Journal of Supercomputing, Vol. 82, 2026; verify first author name |

## Ordering for Related Work Section

The expanded Section 2 should present these references as follows:

1. **Paragraph 1 (MMP foundations):** Patani1996, Hussain2010, Griffen2011, Zdrazil2023 -- unchanged
2. **Paragraph 2 (database/rule-based tools):** SwissBioisostere, CReM, **NeBULA** (new -- most comprehensive database)
3. **Paragraph 3 (learning-based methods):** Ertl2020, **DeepBioisostere v2** (generative), **GraphBioisostere** (GNN)
4. **Paragraph 4 (rank fusion):** Dwork2001, Cormack2009 -- unchanged
5. **Paragraph 5 (risk/alert assessment -- NEW):** Baell2010, Brenk2008, **Helmke2025** (data-driven off-target assessment)
