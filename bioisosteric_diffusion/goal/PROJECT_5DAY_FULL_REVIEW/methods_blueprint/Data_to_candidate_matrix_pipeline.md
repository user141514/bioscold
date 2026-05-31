# Data → Candidate Matrix Pipeline

## Source
ChEMBL37K (~37,000 drug-like molecules from ChEMBL33)

## Pipeline Steps

### 1. MMP Extraction
- Recursive decomposition at acyclic single bonds → fragment-scaffold pairs
- Any two molecules sharing identical scaffold + different substituent at same attachment point → MMP
- Observed substitution (f_old → c*) = positive example

### 2. Quality Filters
- Salt stripping, stereochemistry normalization
- Remove PAINS/Brenk alert molecules
- Fragment MW: 15–250 Da

### 3. Vocabulary Construction
- V_train: all unique replacement fragments from training MMPs
- Closed vocabulary: only fragments in V_train are eligible candidates
- Typical size: 150–152 fragments

### 4. Candidate Matrix Per Query
- Query q_i = (f_i^old, σ_i)
- Candidates C_i = {c ∈ V_train : χ(c,σ_i) = 1}
- Attachment compatibility χ: valence + bond-order constraints
- Each row = (query_id, candidate_id, features...)

### 5. Decoy Repair
- 1:1 positive:decoy ratio for balanced training
- Wrong-positive removal: decoys in P_i excluded
- Deduplication
- 1:5 diagnostic set (class imbalance stress test, excluded from primary eval)

### 6. Label Assignment
- y_ic = 1 if (f_i^old, σ_i, c) ∈ M (observed MMP)
- y_ic = 0 otherwise (decoy or non-observed)
- Multi-positive: queries can have |P_i| > 1

### 7. Split Design
- Transform-heldout: partition unique (f_old, σ) pairs → zero overlap train/eval
- Development/calibration: held-out train partition for architecture/hyperparameter selection
- Secondary blind: 13,347 queries, zero transform overlap with train
- Canonical analysis: 21,052 queries, robustness only
