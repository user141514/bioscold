# Cover Letter

**Date**: [DATE]

**To**: Editor-in-Chief  
*Journal of Chemical Information and Modeling*  
ACS Publications

**RE**: Submission of "Learned Candidate–Fragment Compatibility Ranking for Fragment Replacement"

---

Dear Editor,

We submit the manuscript "Learned Candidate–Fragment Compatibility Ranking for Fragment Replacement" for consideration in the *Journal of Chemical Information and Modeling*.

Fragment replacement ranking is a core computational task in bioisosteric design, directly relevant to JCIM's scope of chemical information and molecular modeling. Our work presents **LBC-Ranker**, a compact logistic regression model that learns the balance between a candidate's replacement track record and its substructure compatibility with the query old fragment.

**Key contributions**:

1. We demonstrate that learned feature combination (freq + bit_corr + Morgan + physicochemical deltas) substantially outperforms both hand-crafted content similarity and K-nearest-OF retrieval under fair, cross-validated baseline tuning.

2. Across 123 old fragments from ChEMBL 36 under a 10-seed repeated OF-level split protocol, LBC-Ranker achieves an OF-macro Hit@10 of 0.852 ± 0.032 and improves over the strongest tuned non-LBC baseline by a paired mean difference of +0.130 (exact p = 0.00195).

3. A matched-capacity gradient-boosted tree (HGB) reaches essentially identical performance, confirming that the eight-feature representation — not model capacity — drives ranking quality.

4. Feature ablation identifies three independently contributing signals (frequency, bit-level correlation, Morgan similarity), and a complete tuning ledger is provided in the Supporting Information.

5. The model is compact (9 parameters), deterministic, CPU-trainable, and directly interpretable, offering a practical tool for fragment replacement prioritization.

We confirm that this manuscript has not been published elsewhere and is not under consideration by another journal. All authors have approved the manuscript and agree with its submission to JCIM. The authors declare no competing financial interests. The data and code are available at `https://github.com/user141514/bioscold`.

We suggest the following potential reviewers (no conflicts of interest):  
[Optional: list 3-5 reviewer suggestions with affiliations]

Sincerely,  
[Corresponding Author Name]  
[Affiliation]  
[Email]

---

**Co-authors**:  
[Author 2 Name], [Affiliation]  
[Author 3 Name], [Affiliation]
