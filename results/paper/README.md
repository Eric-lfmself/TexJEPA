# Historical results from our TexJEPA manuscript

We release the aggregate results from our historical experiments alongside our implementation for studying texture sensitivity in chest X-ray representations. The source is our **unsubmitted manuscript**, *When Texture Becomes the World: Texture-aware JEPA for Chest X-ray Representation Learning*. Our manuscript is not distributed in this repository.

We transcribe every table, including the experimental protocols and qualitative interpretations, and preserve the precision displayed in the manuscript. These files contain aggregate results; they do not contain patient images, patient-level predictions, or model weights.

## Files

| File | Contents |
| --- | --- |
| [tables.json](tables.json) | Canonical structured transcription of Tables I-XIII, with numeric values, original display strings, units, source locators, and missing-value notes |
| [TABLES.md](TABLES.md) | Readable export of all 13 tables |
| [csv/](csv/) | One CSV per table, plus supplementary values; every row includes source provenance |
| [supplementary.json](supplementary.json) | 16 printed Figure 7 heatmap values, 9 Section V.I values, and the approximate smoothing range in the Figure 8 caption |
| [figure_annotations.json](figure_annotations.json) | Separately recorded numeric annotations printed in the Figure 4 robustness matrix |
| [source.sha256](source.sha256) | Fingerprint of the private source manuscript used for these transcriptions |

The private source manuscript has 13 pages. Its SHA-256 is `ca4a03a5f94dbb5b60396f76834d68d54b39eb760e73ef2d02c913dd74dc9ab7`.

## Table coverage

| Table | Manuscript page | Rows | Content type | Subject |
| --- | ---: | ---: | --- | --- |
| I | 3 | 6 | Experimental protocol | Evaluation axes |
| II | 4 | 3 | Experimental protocol | TexJEPA variant names and roles |
| III | 4 | 3 | Experimental protocol | Downstream protocols |
| IV | 5 | 2 | Numeric results | Baseline Gaussian-noise AUROC |
| V | 6 | 2 | Numeric results | Baseline cosine drift |
| VI | 7 | 5 | Numeric results | Frequency sensitivity |
| VII | 8 | 6 | Numeric results | Linear probe, MLP probe, and partial fine-tuning |
| VIII | 8 | 5 | Numeric results | Lesion sensitivity and 95% confidence intervals |
| IX | 8 | 7 | Numeric results | Post-training Gaussian-noise AUROC |
| X | 9 | 7 | Numeric results | Post-training cosine drift |
| XI | 10 | 5 | Numeric results and interpretation | Robustness and lesion sensitivity |
| XII | 10 | 7 | Numeric results | Repeated checkpoint AUROC table |
| XIII | 11 | 3 | Qualitative interpretation | Additional interventions |

There are **61 table rows, 193 numeric cells, and 4 null cells**. The count includes repeated values wherever they appear in separate manuscript tables. In particular, Table XII repeats Table IX and is marked `duplicate_of: "table_ix"`; it is not an independent experimental run.

## Schema and units

Both canonical JSON files use `schema_version: "1.0.0"`, `publication_status: "unsubmitted_manuscript"`, `evidence: "paper_reported"`, and `dataset_type: "aggregate_experimental_results"`. The top-level `source` object records `path: "texjepa-manuscript"` and `availability: "not_distributed"`. We also attach evidence and source metadata to individual tables or supplementary records.

Each table has:

- `id` and `table`: a stable identifier such as `table_iv` and the manuscript's Roman table number.
- `table_kind`: `experimental_protocol`, `numeric_results`, `mixed_results_and_interpretation`, or `qualitative_interpretation`. Numeric training settings embedded in Table III remain protocol text.
- `source`: the manuscript identifier `texjepa-manuscript` in the existing `path` field, a one-based manuscript page number, and table number. The identifier is a provenance label, not a file path or download location; source availability is declared in the document's top-level metadata.
- `columns`: keys, original labels, data types, units, and decimal precision. Noise columns also record their condition and sigma.
- `rows`: one dictionary per printed row. Measurements are JSON numbers, text remains text, and unreported cells are `null`.
- `reported_headers` and `reported_rows`: the printed table layout and values as strings. We join line wraps, remove line-breaking hyphenation, and retain meaningful wording, signs, and decimal places. These strings retain trailing zeros that a numeric parser may discard.
- `missing_cells`: zero-based row indices, column keys, original dash markers, and the reason a cell is null.
- `notes`: source-specific definitions and ambiguities.

All numeric table measurements are displayed to **three decimal places**. AUROC is a dimensionless fraction, not a percentage. Cosine drift is `1 - cosine(clean, perturbed)`, a dimensionless quantity. Delta Logit Drop is the lesion-region logit drop minus the matched-control logit drop, in logit-difference units. Frequency cutoffs and band boundaries are normalized frequencies. Sigma labels retain the manuscript's Gaussian-noise standard deviations and should be interpreted with the evaluation preprocessing; we do not assign physical acquisition units.

Table VIII's printed `95% CI` is split into `ci95_lower` and `ci95_upper` in numeric rows and CSV. Its original combined interval is retained in `reported_rows`. The manuscript describes these as bootstrap confidence intervals. We do not infer intervals for the post-training lesion scores in Table XI.

CSV uses UTF-8 with a header row. Empty numeric fields correspond to JSON `null`; zero remains `0.000`. Per-row metadata columns are `evidence`, `dataset_type`, `source_pdf`, `source_pdf_page`, and `source_table`. The existing `source_pdf` column contains the provenance identifier `texjepa-manuscript`; it does not point to a distributed PDF.

Supplementary records retain their individual `reported` strings and `precision` labels. The 16 Figure 7 values are the numbers printed in the heatmap, at two decimal places; the full model names are paired with their displayed `JEPA300` and `MAE300` labels. Approximate prose values remain marked `approximate` or `contextually_approximate`. A reported range stores `value: null`, `range_lower`, and `range_upper`; we do not invent a midpoint. Supplementary CSV includes the source location and precision for each record.

## Source details that affect interpretation

We retain the manuscript's historical model names. The post-training `v3.1 (Baseline)` corresponds to I-JEPA-H/201 in Section III.B; it is distinct from I-JEPA-H/300 in the main baseline comparison. The `v4`, `v5`, and `v6` names are retained where printed, alongside the TexJEPA-N/R/C mapping in Table II.

Table VI reports a mid-band AUROC of `0.594`, clean AUROC of `0.910`, and Drop of `-0.317`. Subtracting the displayed AUROCs gives `-0.316`. We preserve the printed Drop without assuming unreported precision or silently correcting the source. The two clean-condition dashes in that table remain null. The two MLP drift dashes in Table VII also remain null.

Equation (1) and Table VI use **perturbed minus clean** for AUROC difference. Some figure panels display the opposite **clean minus perturbed** convention. We preserve source-specific signs rather than combining the two conventions.

Section V.I reports raw-encoder drift of `0.46` and `0.60` within the NCA experiment, whereas Table V reports `0.755` and `0.895` at the corresponding noise levels for the main baseline experiment. We keep those values in their own source contexts and do not substitute one for the other. Section V.I describes clean NCA AUROC as about `0.875`; the Figure 8 caption rounds it to about `0.88`. We preserve the prose value with its approximate status.

## Validate and regenerate

From the repository checkout, using Python 3.10 or later:

```bash
python scripts/export_paper_results.py --check
python scripts/export_paper_results.py
```

The first command checks table coverage, data types, displayed precision, confidence-interval order, missing cells, repeated-table consistency, and the committed CSV/Markdown exports. It does not require the manuscript PDF. The second regenerates those exports from the canonical JSON. Neither command runs training or evaluates patient data.

The exporter uses only Python's standard library. The results are repository assets rather than wheel package data, so these commands require the source checkout. `--root` accepts another checkout directory. If we provide an authorized local copy of the private manuscript with `--source-pdf`, the exporter additionally verifies its SHA-256. The default validation and regeneration commands require only the committed results.

## Summary figures

We plot the reported values directly from `tables.json`:

- [Main results](figures/main_results.png): Tables IV, V and VIII; [PDF](figures/main_results.pdf), [SVG](figures/main_results.svg).
- [Post-training results](figures/post_training_results.png): Tables IX, X and XI; [PDF](figures/post_training_results.pdf), [SVG](figures/post_training_results.svg).

The main figure compares noise AUROC, representation drift, and lesion sensitivity with the reported 95% intervals. The post-training figure compares robustness, drift, and reported lesion point estimates; no unreported intervals are added. Source and output fingerprints are retained in [figures/manifest.json](figures/manifest.json).

```sh
python -m scripts.plot_paper_results --output outputs/paper_figures
```
