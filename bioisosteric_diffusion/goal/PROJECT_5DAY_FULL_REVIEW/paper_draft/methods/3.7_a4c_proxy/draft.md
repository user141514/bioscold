### 3.7 Computational Review Proxy (A4C)

Bioisosteric replacement, even when chemically conservative, can produce
unexpected off-target effects: recent experimental profiling across 88
safety-relevant targets showed that an ester-to-secondary-amide substitution
increases CHRM2 binding affinity approximately 10-fold (Helmke et al., 2025),
demonstrating that structural familiarity is no guarantee of selectivity. This
motivates a lightweight computational triage layer that flags potentially
problematic proposals before synthesis. We therefore adopt a dual-mode workflow
under an A4C computational review proxy that categorizes each candidate
proposal by its provenance within the ranking pipeline and by the presence of
medicinal-chemistry structural alerts. The A4C annotation layer applies
rule-based filters -- PAINS alerts (Baell and Holloway, 2010) and Brenk alerts
(Brenk et al., 2008) -- to flag structural features associated with assay
interference, reactivity, or unfavorable pharmacokinetics. We additionally
record shifts in key molecular properties (molecular weight, logP, TPSA, and
hydrogen-bond donor and acceptor counts) relative to the query fragment. The
framework is exclusively a computational screening aid and does not constitute
a medicinal-chemistry truth standard; its purpose is to stratify proposals into
risk tiers for qualitative assessment, providing a proxy for the type of
first-pass triage a medicinal chemist would perform when reviewing output
lists.

#### 3.7.1 Dual-Mode Workflow Design

The dual-mode workflow comprises two operating modes that reflect different
positions on the exploration-exploitation spectrum, each generating proposals
through a distinct ranking route.

**Conservative Mode** draws proposals exclusively from the HGB ranker
(Section 3.3.3). Because HGB relies on training-set fragment frequencies and
molecular descriptors, its top-ranked proposals are biased toward chemically
well-characterized replacements that align with historical preferences in the
training data. This mode produces chemically conservative suggestions that are
likely to be synthetically accessible and structurally plausible, but it may
fail to propose less common but equally valid replacements.

**Exploration Mode** draws proposals from the Borda(DE, HGB) fused ranking
(Section 3.3.4), which combines the Dual Encoder's learned molecular similarity
with HGB's frequency-based signal. The Dual Encoder component enables the fused
ranking to propose replacements that differ from training-set frequency priors,
potentially identifying novel or rare fragments that HGB alone would rank low.
Exploration Mode thus trades some chemical conservatism for broader coverage of
the fragment space.

#### 3.7.2 Provenance Labels and Alert Stratification

Within Exploration Mode, each proposal receives a provenance label that
indicates its origin within the fused ranking relative to the individual base
rankers. These labels define three groups with distinct alert-rate profiles,
summarized in Table 7.

---

**Table 7.** Provenance groups, definitions, and A4C alert rates. Here
$\mathcal{K}_{m}$ denotes the set of top-$K$ candidates according to ranker
$m \in \{\mathrm{HGB}, \mathrm{DE}\}$, and $\mathcal{K}_{\mathrm{Borda}}$
denotes the top-$K$ set of the Borda fused ranking.

| Group | Definition | A4C Alert Rate | Interpretation |
|-------|-----------|----------------|----------------|
| $\mathrm{G4}$ | In $\mathcal{K}_{\mathrm{HGB}} \cap \mathcal{K}_{\mathrm{Borda}}$ | 0.99% | Low-alert reference; broad consensus between frequency and similarity signals |
| $\mathrm{G3}$ | In $\mathcal{K}_{\mathrm{Borda}} \setminus \mathcal{K}_{\mathrm{HGB}}$ where DE drives rank elevation | 9.67% | Moderate expansion; similarity-supported novelty beyond frequency priors |
| $\mathrm{G2}$ | In $\mathcal{K}_{\mathrm{Borda}} \setminus (\mathcal{K}_{\mathrm{HGB}} \cup \mathcal{K}_{\mathrm{DE}})$ | 46.85% | Highest novelty; combined ranker agreement with elevated structural-alert burden |

---

**$\mathrm{G4}$: Shared candidates.** Fragments appearing in the top $K$ of
both the HGB and Borda rankings represent the intersection of frequency-based
and similarity-based recommendation. The near-zero A4C alert rate (0.99%)
establishes these as a low-alert reference set: proposals on which both ranking
methods agree carry minimal structural flags and are suitable as first-pass
suggestions.

**$\mathrm{G3}$: DE-elevated candidates.** Fragments in the Borda top $K$ that
fall outside the HGB top $K$ represent expansions beyond frequency-based
priors. Since DE is the non-HGB component of the Borda fusion, these proposals
are supported by learned molecular similarity -- the Dual Encoder identifies
them as chemically related to the query -- but they lack the support of
historical frequency. The moderate alert rate (9.67%) indicates that this
expansion carries some additional structural risk but remains manageable for
most applications.

**$\mathrm{G2}$: Borda-emergent candidates.** Fragments appearing in the Borda
top $K$ but absent from the individual top-$K$ lists of both HGB and DE
represent the highest novelty tier. These proposals emerge specifically from
the *combined* effect of the two rankers -- they are not strongly recommended
by either ranker alone, but their aggregate Borda score places them among the
top candidates. The substantially elevated alert rate (46.85%) indicates that
nearly half of these proposals carry structural alerts. $\mathrm{G2}$ candidates
therefore require individual expert medicinal-chemistry review before any
synthetic commitment.

This provenance-based stratification provides an actionable risk structure:
a reviewer can allocate attention according to alert burden, treating
$\mathrm{G4}$ proposals as low-risk starting points, $\mathrm{G3}$ proposals as
moderate-risk candidates suitable for closer inspection, and $\mathrm{G2}$
proposals as high-novelty items that demand expert evaluation.

#### 3.7.3 Scope and Limitations

We emphasize that the A4C framework is used exclusively for workflow and risk
interpretation. It does not support any primary performance claim -- the Top-10
accuracy results that establish the scorer's effectiveness rest solely on the
secondary blind protocol (Section 3.6.4). The alert rates reported here are
computational screening signals derived from published rule-based filters and
simple property-shift thresholds; they have not been validated through
medicinal-chemistry expert review, and they do not constitute a determination
of synthetic accessibility, assay interference, or toxicity. Indeed, as the
Helmke et al. (2025) finding illustrates, off-target effects can arise from
replacements that pass all computational filters, underscoring that no
rule-based proxy can replace experimental profiling. The A4C framework is a
computational screening aid, not a substitute for domain expertise or
experimental validation.

Consistent with standard practice in computational fragment replacement, we
recommend the following protocol: $\mathrm{G4}$ proposals may be used as
initial suggestions without further computational triage; $\mathrm{G3}$
proposals should be reviewed for structural flags before synthesis; and
$\mathrm{G2}$ proposals require comprehensive expert medicinal-chemistry
review. This conservative interpretation ensures that the computational proxy
informs rather than replaces human judgment, and it guards against
over-interpretation of computational alert signals in the absence of
experimental confirmation.
