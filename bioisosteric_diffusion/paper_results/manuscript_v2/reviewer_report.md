# Simulated Peer Review: LBC-Ranker Manuscript

**Target Journal**: Journal of Chemical Information and Modeling (JCIM)
**Review Date**: 2026-06-07
**Recommendation**: **Minor Revision**

---

## Overall Assessment

This manuscript presents LBC-Ranker, a compact logistic regression model for ranking fragment replacement candidates. The method combines eight hand-crafted features (Morgan similarity, bit-level fingerprint correlation, five physicochemical deltas, and train-only candidate frequency) with learned weights. The evaluation protocol is unusually thorough: 10 repeated OF-level splits with inner 3-fold cross-validation for baseline hyperparameter tuning, an exact paired sign-flip test, and a full one-drop feature ablation. The writing is clear, the claims are appropriately scoped, and the limitations section is honest.

The primary strength is the evaluation rigor. Tuning baselines rather than using default parameters, reporting per-seed results rather than only aggregates, and providing a complete tuning ledger in the Supporting Information sets a standard that most JCIM submissions do not meet. The frank acknowledgment that frequency is the dominant feature (not bit_corr) and that LR matches HGB (not exceeds it) indicates intellectual honesty.

The primary weakness is that the eight-feature representation, while interpretable, is not deeply novel. Morgan similarity, physicochemical deltas, and candidate frequency are individually well-precedented features. The bit_corr feature is the only genuinely new element, and it ranks second in the ablation. The paper's contribution is therefore primarily in the rigorous demonstration that learned feature combination improves over hand-crafted and retrieval baselines, rather than in a new representation or algorithm.

---

## Major Concerns

### M1. Problem Formulation (line 33) contains an unrevised phrase

> "candidate c is a known successful replacement"

This is inconsistent with the rest of the manuscript, which uses "labeled positive replacement" (after an earlier round of editing). The phrase "successful replacement" appears only here. Please harmonize to "labeled positive replacement" throughout.

**Required**: Replace "successful replacement" with "labeled positive replacement" on line 33.

### M2. The "Best non-LBC" row in Table 2 needs explicit definition

The "Best non-LBC (per-seed)" row aggregates a metric whose definition depends on per-seed selection of the best baseline. A reader unfamiliar with the evaluation protocol may not immediately understand that "0.723" is the mean across seeds of the per-seed best baseline Hit@10, not a single method's score. The Methods section (§2.6) explains this, but the table should be self-contained.

**Required**: Add a footnote to Table 2: "Mean across seeds of the per-seed best non-LBC baseline Hit@10. The best baseline is C3F-style retrieval in 7 seeds and CA in 3 seeds."

### M3. Reference [7] (Papadatos & Brown, 2013) should be verified

Reference [7] cites WIREs Comput. Mol. Sci. 2013, 3 (5), 449-458. While this is likely a real paper, please verify the exact title, volume, and page numbers before submission. JCIM reviewers will check references.

**Required**: Verify all 9 references for exact DOI, volume, and page numbers before submission.

---

## Minor Concerns

### m1. bit_corr definition could be more precise

Line 50: "Pearson correlation between L1-normalized candidate and OF fingerprint bit vectors"

For binary (0/1) fingerprints of length 2048, L1-normalization produces vectors where most entries are zero and non-zero entries are 1/(number of bits set). The Pearson correlation of two such sparse binary vectors has a known relationship to the Jaccard/Tanimoto coefficient. The manuscript would benefit from a brief note (possibly in SI) clarifying whether bit_corr provides information that is mathematically independent of Morgan Tanimoto, or whether it captures the same overlap signal in a normalized form.

**Suggested**: Add one sentence to §2.3: "For binary fingerprints, bit_corr is related to but distinct from Tanimoto similarity: it measures the linear association of bit intensities after normalization rather than the fractional overlap of set bits."

### m2. LBC-Ranker weights should be reported

The Discussion states (line 154) that "each weight w_i quantifies the marginal contribution of its corresponding feature." However, the actual learned weights are never reported. For a paper whose primary methodological claim is interpretability, readers should see the weights.

**Suggested**: Add a table or figure in SI showing the mean and std of each w_i across the 10 seeds.

### m3. The 14 zero-positive OFs are mentioned but not analyzed

Line 39 notes that 14 OFs "have zero positive-label support in the current archive." This is an important practical limitation. The manuscript could strengthen its cold-start discussion by reporting what LBC-Ranker predicts for these OFs (e.g., top-ranked candidates, distribution of scores) even though no ground-truth evaluation is possible.

**Suggested**: Add a brief paragraph to SI analyzing LBC-Ranker predictions on the 14 zero-positive OFs.

### m4. Figure references are placeholders

Line 138: "The per-OF Hit@10 distribution (Figure X, Supporting Information)..."

The actual figure number should be assigned.

**Suggested**: Replace "Figure X" with the actual SI figure number once figures are finalized.

### m5. Query-weighted Hit@10 should be in Table 2

The query-weighted Hit@10 values are mentioned in §3.5 but not tabulated. Given that they serve as a guardrail metric and differ meaningfully from the macro values, they should appear alongside the macro values in a supplementary table or in Table 2.

**Suggested**: Add query-weighted columns to Table S3 in the SI.

---

## Verification Summary

| Reference | Authors | Venue | Verified? |
|-----------|---------|-------|-----------|
| [1] | Meanwell (2011) | J. Med. Chem. | Well-known, likely correct |
| [2] | Brown (2012) | Wiley-VCH book | Well-known, likely correct |
| [3] | Ertl (2007) | Curr. Opin. Drug Discov. Dev. | Well-known, likely correct |
| [4] | Wagener & Lommerse (2006) | JCIM | Well-known, likely correct |
| [5] | Rogers & Hahn (2010) | JCIM | ECFP paper, correct |
| [6] | Bajusz et al. (2015) | J. Cheminform. | Well-known, likely correct |
| [7] | Papadatos & Brown (2013) | WIREs Comput. Mol. Sci. | Needs verification |
| [8] | Ertl & Schuffenhauer (2009) | J. Cheminform. | SA_Score paper, correct |
| [9] | Gaulton et al. (2017) | Nucleic Acids Res. | ChEMBL paper, correct |

**All 9 references appear to be real, well-cited papers in established venues.**

---

## Recommendation

**Minor Revision.** The manuscript presents a methodologically sound study with rigorous evaluation, honest reporting of results, and well-scoped claims. The major concerns (one inconsistent phrase, one table footnote, and one reference verification) are trivially fixable. The minor concerns would improve clarity and completeness but do not affect the core findings.

This paper makes a solid contribution to the fragment replacement ranking literature by demonstrating (a) that learned feature combination outperforms both hand-crafted and retrieval baselines under a fair tuning protocol, (b) that a simple linear model suffices when the feature representation is appropriate, and (c) that candidate track record, structural compatibility, and property matching each contribute independent information. I recommend acceptance after minor revision.
