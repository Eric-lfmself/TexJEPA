# Historical manuscript tables

We report these aggregate results in our unsubmitted TexJEPA manuscript. We preserve the displayed values, historical model names, and missing cells. Protocol and interpretation tables are identified separately.

Source: private manuscript texjepa-manuscript (not distributed). SHA-256: `ca4a03a5f94dbb5b60396f76834d68d54b39eb760e73ef2d02c913dd74dc9ab7`.

## Table I. Diagnostic evaluation axes used in this study.

PDF page 3 · experimental_protocol · [CSV](csv/table_i.csv)

| Axis | Measurement | Purpose |
| --- | --- | --- |
| Clean utility | Macro AUROC | Standard downstream accuracy |
| Perturbation robustness | AUROC under noise and frequency corruption | Prediction stability |
| Feature stability | Cosine representation drift | Encoder-level sensitivity |
| Lesion sensitivity | Delta Logit Drop | Use of annotated pathology |
| Adaptability | MLP and partial fine-tuning | Head vs. encoder diagnosis |
| Post-training | v4/v5/v6 comparison | Robustness–semantics trade-off |

- This table describes our evaluation axes; it does not contain measured numeric outcomes.

## Table II. Texture-aware JEPA variants and diagnostic roles.

PDF page 4 · experimental_protocol · [CSV](csv/table_ii.csv)

| Variant | Source | Diagnostic role |
| --- | --- | --- |
| TexJEPA-N | v4 | Missing noise-invariance signal |
| TexJEPA-R | v5 | Register-token readout hygiene and tight-mask texture prediction |
| TexJEPA-C | v6 | Patch-token feature decorrelation and high-noise smoothing |


## Table III. Downstream evaluation protocols.

PDF page 4 · experimental_protocol · [CSV](csv/table_iii.csv)

| Protocol | Trainable part | Key settings |
| --- | --- | --- |
| Linear probe | Head | SGD, cosine schedule, 100 epochs |
| MLP probe | Head | Hidden 512, GELU, dropout |
| Partial FT | Last two ViT blocks + head | AdamW, 15 epochs |

- Numbers in Key settings specify training configuration, not measured outcomes. We preserve the concise table wording; Section IV.A also specifies the final normalization layer for partial fine-tuning.

## Table IV. Gaussian-noise robustness under frozen linear probing (AUROC).

PDF page 5 · numeric_results · [CSV](csv/table_iv.csv)

| Model | Clean | 0.05 | 0.10 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- | --- |
| I-JEPA-H/300 | 0.910 | 0.641 | 0.589 | 0.571 | 0.553 |
| MAE-H/300 | 0.891 | 0.777 | 0.618 | 0.490 | 0.485 |


## Table V. Cosine representation drift under Gaussian noise.

PDF page 6 · numeric_results · [CSV](csv/table_v.csv)

| Model | 0.05 | 0.10 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- |
| I-JEPA-H/300 | 0.755 | 0.895 | 0.871 | 0.824 |
| MAE-H/300 | 0.169 | 0.319 | 0.339 | 0.337 |

- Cosine drift is 1 - cosine similarity, as defined in Equation (2).

## Table VI. Frequency sensitivity of I-JEPA-H/300 under linear probing.

PDF page 7 · numeric_results · [CSV](csv/table_vi.csv)

| Condition | AUROC | Drop | Drift |
| --- | --- | --- | --- |
| Clean | 0.910 | – | – |
| Low-pass cutoff 0.15 | 0.670 | −0.240 | 0.437 |
| Band corrupt 0.00–0.20 | 0.847 | −0.063 | 0.110 |
| Band corrupt 0.20–0.45 | 0.594 | −0.317 | 0.752 |
| Band corrupt 0.45–1.00 | 0.618 | −0.292 | 0.622 |

- Frequency cutoffs and band limits are normalized frequencies, as specified in Section IV.C.
- Drop follows Equation (1): perturbed AUROC minus clean AUROC. Negative values indicate degradation.
- The mid-band row prints Drop = -0.317, whereas 0.594 - 0.910 from the displayed AUROCs is -0.316. We preserve -0.317 without inferring additional precision.
- The Clean row uses dashes for Drop and Drift. These remain null; we do not insert zeros.

## Table VII. Probe capacity and partial fine-tuning under Gaussian noise. Drift is measured at noise standard deviation 0.05.

PDF page 8 · numeric_results · [CSV](csv/table_vii.csv)

| Model | Protocol | Clean | 0.05 | 0.10 | 0.20 | 0.30 | Drift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I-JEPA-H/300 | Linear probe | 0.910 | 0.641 | 0.589 | 0.571 | 0.553 | 0.755 |
| I-JEPA-H/300 | MLP probe | 0.927 | 0.678 | 0.573 | 0.582 | 0.571 | – |
| I-JEPA-H/300 | Partial FT | 0.935 | 0.625 | 0.587 | 0.555 | 0.534 | 0.780 |
| MAE-H/300 | Linear probe | 0.891 | 0.777 | 0.618 | 0.490 | 0.485 | 0.169 |
| MAE-H/300 | MLP probe | 0.925 | 0.786 | 0.568 | 0.471 | 0.464 | – |
| MAE-H/300 | Partial FT | 0.949 | 0.820 | 0.588 | 0.462 | 0.485 | 0.689 |

- The Drift column applies only to σ = 0.05. The two MLP drift cells are dashes in the source and remain null.

## Table VIII. Lesion sensitivity measured by Delta Logit Drop (Δ_lesion).

PDF page 8 · numeric_results · [CSV](csv/table_viii.csv)

| Model | Δ_lesion | 95% CI |
| --- | --- | --- |
| I-JEPA-H/300 | 0.222 | [0.199, 0.247] |
| I-JEPA-H/201 | 0.195 | [0.167, 0.223] |
| I-JEPA-H/95 | 0.190 | [0.161, 0.220] |
| MAE-H/97 | 0.066 | [0.053, 0.078] |
| MAE-H/300 | 0.039 | [0.025, 0.053] |

- Δ_lesion is lesion logit drop minus the matched-control logit drop, as defined in Equation (3).
- The source reports 95% confidence intervals. Figure 7 describes them as bootstrap confidence intervals; no bootstrap sample count is added here.
- The CSV and numeric rows split the printed 95% CI into lower and upper bounds; reported_rows preserves the printed combined interval.

## Table IX. Post-training robustness comparison under Gaussian noise (AUROC).

PDF page 8 · numeric_results · [CSV](csv/table_ix.csv)

| Model | Clean | σ=0.05 | σ=0.10 | σ=0.20 | σ=0.30 |
| --- | --- | --- | --- | --- | --- |
| v3.1 (Baseline) | 0.916 | 0.643 | 0.596 | 0.586 | 0.588 |
| v4 (+noise) | 0.918 | 0.906 | 0.883 | 0.713 | 0.513 |
| v5 (+reg) | 0.930 | 0.920 | 0.897 | 0.556 | 0.550 |
| v6 (+vicreg) | 0.916 | 0.906 | 0.889 | 0.802 | 0.529 |
| MAE-H/300 | 0.891 | 0.777 | 0.618 | 0.490 | 0.485 |
| EVA-X-B/300 | 0.933 | 0.836 | 0.645 | 0.575 | 0.563 |
| RAD-DINO-B/300 | 0.935 | 0.871 | 0.672 | 0.512 | 0.554 |

- We preserve the historical checkpoint labels printed in this table. Section III.B identifies v3.1 with I-JEPA-H/201; it is distinct from the I-JEPA-H/300 baseline in Tables IV-VII.
- Table XII repeats these numeric rows; it is not an independent experiment.

## Table X. Post-training cosine drift under Gaussian noise.

PDF page 9 · numeric_results · [CSV](csv/table_x.csv)

| Model | σ=0.05 | σ=0.10 | σ=0.20 | σ=0.30 |
| --- | --- | --- | --- | --- |
| v3.1 (Baseline) | 0.872 | 0.963 | 0.957 | 0.956 |
| v4 (+noise) | 0.017 | 0.049 | 0.489 | 0.777 |
| v5 (+reg) | 0.016 | 0.052 | 0.781 | 0.811 |
| v6 (+vicreg) | 0.010 | 0.017 | 0.064 | 0.212 |
| MAE-H/300 | 0.169 | 0.319 | 0.339 | 0.337 |
| EVA-X-B/300 | 0.125 | 0.237 | 0.285 | 0.299 |
| RAD-DINO-B/300 | 0.164 | 0.464 | 0.701 | 0.711 |


## Table XI. Robustness–semantics trade-off among post-training variants.

PDF page 10 · mixed_results_and_interpretation · [CSV](csv/table_xi.csv)

| Model | Δ_lesion | Best low-noise AUROC | Drift at 0.20 | Interpretation |
| --- | --- | --- | --- | --- |
| v3.1 baseline | 0.195 | 0.643 | 0.957 | Strong lesion sensitivity but fragile at light noise |
| v4 + noise | 0.126 | 0.906 | 0.489 | Balanced improvement from clean-target noise invariance |
| v5 + register | 0.177 | 0.920 | 0.781 | Best clean/low-noise readout, but cliff at stronger noise |
| v6 + var/cov | 0.040 | 0.906 | 0.064 | Smoothest feature space, but lesion sensitivity largely lost |
| MAE-H/300 | 0.039 | 0.777 | 0.339 | Stable reconstruction baseline with weak lesion-specific readout |

- We retain the source heading Best low-noise AUROC. Its entries match the σ = 0.05 column in Table IX; the export does not recompute a maximum.
- No confidence intervals are printed here for the post-training Δ_lesion results. None are inferred.
- The Interpretation column records our qualitative interpretation rather than a separate measurement.

## Table XII. Gaussian Noise Robustness Curve of Golden Checkpoints measured by Macro AUROC.

PDF page 10 · numeric_results · [CSV](csv/table_xii.csv)

| Model | Clean | σ=0.05 | σ=0.10 | σ=0.20 | σ=0.30 |
| --- | --- | --- | --- | --- | --- |
| v3.1 (Baseline) | 0.916 | 0.643 | 0.596 | 0.586 | 0.588 |
| v4 (+noise) | 0.918 | 0.906 | 0.883 | 0.713 | 0.513 |
| v5 (+reg) | 0.930 | 0.920 | 0.897 | 0.556 | 0.550 |
| v6 (+vicreg) | 0.916 | 0.906 | 0.889 | 0.802 | 0.529 |
| MAE-H/300 | 0.891 | 0.777 | 0.618 | 0.490 | 0.485 |
| EVA-X-B/300 | 0.933 | 0.836 | 0.645 | 0.575 | 0.563 |
| RAD-DINO-B/300 | 0.935 | 0.871 | 0.672 | 0.512 | 0.554 |

- The source repeats the values in Table IX. We preserve this separate table with explicit duplicate metadata.

## Table XIII. Interpretation of additional diagnostic interventions.

PDF page 11 · qualitative_interpretation · [CSV](csv/table_xiii.csv)

| Intervention | Observed behavior | Interpretation |
| --- | --- | --- |
| Token drift | Noise changes many tokens; lesion occlusion is more localized | I-JEPA combines local lesion sensitivity with global noise instability |
| Input smoothing | Helps only slightly under very light noise | Fragility is not solved by simple denoising and may overlap with lesion texture |
| Consistency adapter | Aligns noisy features but can reduce clean I-JEPA performance | Downstream smoothing suppresses discriminative lesion directions |

- This table contains qualitative observations and interpretations, not numeric measurements.

## Caption, figure-annotation, and prose values

We keep the explicitly printed Figure 7 annotations and approximate NCA/preprocessing values in [supplementary.json](supplementary.json) and [supplementary.csv](csv/supplementary.csv). Their precision and source locations are recorded per value.

The printed Figure 4 heatmap annotations are stored separately in [figure_annotations.json](figure_annotations.json).
