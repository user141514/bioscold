# BindingDB Hard-Negative 10k Diagnostic v2

Date: 2026-06-10

Purpose: run the hard-negative 10k diagnostic before any full 10-seed benchmark, comparing Frequency, CA, fixed Blend, FullBlend, LBC, and LBC no-freq.

## Methods

- Frequency: train-source positive candidate frequency.
- CA: hand-crafted content-aware fragment similarity.
- Blend: fixed z-scored frequency/content blend.
- FullBlend: ridge learned-content scalar blended with frequency; alpha selected from `[0.0, 0.25, 0.5, 0.75, 1.0]` on training OFs.
- LBC: logistic regression on seven content features plus frequency.
- LBC no-freq: logistic regression on the seven content features only.

## 3-Seed Results

| Method | Query Hit@10 mean | Query Hit@10 sd | OF-macro Hit@10 mean | OF-macro Hit@10 sd |
|---|---:|---:|---:|---:|
| Frequency | 0.848522 | 0.011983 | 0.783168 | 0.004893 |
| CA | 0.665755 | 0.005970 | 0.819187 | 0.005570 |
| Blend | 0.749667 | 0.008432 | 0.861194 | 0.003745 |
| FullBlend | 0.956750 | 0.002341 | 0.952495 | 0.002688 |
| LBC | 0.954230 | 0.006078 | 0.948716 | 0.002473 |
| LBC no-freq | 0.947693 | 0.005058 | 0.948654 | 0.002967 |

Mean evaluation coverage: 3,376.0 queries and 1,009 old fragments per seed.

FullBlend selected alpha `0.25` for all three seeds.

## Decision

The precondition `LBC > FullBlend > Frequency / CA` is not met. FullBlend is slightly above LBC, while LBC no-freq is essentially tied with LBC on OF-macro Hit@10.

Do not promote a full 10-seed BindingDB-hard run as an LBC-superiority benchmark yet. The revised interpretation is that learned transferable content support dominates the hard-negative ranking, with frequency contributing but not explaining the full signal.
