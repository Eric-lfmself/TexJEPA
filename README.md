<h1 align="center">TexJEPA</h1>
<p align="center"><strong>Texture-aware JEPA for chest X-ray representation learning</strong></p>

<p align="center">
  <a href="https://github.com/Eric-lfmself/TexJEPA/actions/workflows/tests.yml"><img src="https://github.com/Eric-lfmself/TexJEPA/actions/workflows/tests.yml/badge.svg?branch=main" alt="CPU tests"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11 or newer"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.5 or newer, below 3"></a>
</p>

<p align="center">
  <a href="#main-results">Results</a> ·
  <a href="#texjepa-post-training">Method</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/EXECUTION.md">Experiments</a> ·
  <a href="#documentation">Documentation</a>
</p>

**Clean accuracy tells only part of the story.** We study how chest X-ray encoders use radiographic texture: sensitivity to fine lesion detail can also make a representation fragile to image noise. TexJEPA combines controlled diagnostics with three I-JEPA post-training variants to examine the balance between classification, noise robustness, and lesion sensitivity.

![TexJEPA study overview: compare I-JEPA and MAE, measure classification, feature drift and lesion sensitivity, then evaluate three I-JEPA post-training variants.](docs/assets/overview.svg)

We provide the training and evaluation code, aggregate experimental results, and a small CPU demonstration that requires no images or checkpoints.

## Main results

We compare **MIMIC-CXR-pretrained I-JEPA-H/300 and MAE-H/300** using frozen linear probing on a deterministic **12,000 / 3,000 split of VinDr-CXR/VinBigData**, with 15 binary labels.

| Encoder | Clean AUROC ↑ | Noisy AUROC ↑ | Cosine drift ↓ | Lesion sensitivity ↑ |
|---|---:|---:|---:|---:|
| I-JEPA-H/300 | **0.910** | 0.641 | 0.755 | **0.222** |
| MAE-H/300 | 0.891 | **0.777** | **0.169** | 0.039 |

Noisy AUROC and cosine drift are measured at Gaussian σ = 0.05. Lesion sensitivity is the target-class logit drop from lesion occlusion minus the mean drop from matched non-lesion controls. Sources: [Table IV](results/paper/csv/table_iv.csv), [Table V](results/paper/csv/table_v.csv), and [Table VIII](results/paper/csv/table_viii.csv).

**I-JEPA has stronger clean discrimination and lesion sensitivity, while MAE is more stable under light noise.** This motivates evaluating these properties together.

<details>
<summary><strong>View the noise curves, representation drift, and lesion confidence intervals</strong></summary>

![Reported Gaussian-noise AUROC and cosine drift for I-JEPA and MAE, with 95% bootstrap confidence intervals for lesion sensitivity.](results/paper/figures/main_results.png)

The lesion plot also includes earlier checkpoints. Download the figure as [PDF](results/paper/figures/main_results.pdf) or [SVG](results/paper/figures/main_results.svg).

</details>

## TexJEPA post-training

We start from **I-JEPA-H/201 (`v3.1`)** and train a noisy context branch to predict a clean EMA target. TexJEPA-N introduces this asymmetric noise objective; TexJEPA-R and TexJEPA-C branch from its epoch-50 checkpoint.

| Variant | Post-training change | Clean AUROC ↑ | AUROC at σ = 0.05 ↑ | Lesion sensitivity ↑ |
|---|---|---:|---:|---:|
| I-JEPA-H/201 (`v3.1`) | Baseline | 0.916 | 0.643 | **0.195** |
| **TexJEPA-N** (`v4`) | Asymmetric context noise | 0.918 | 0.906 | 0.126 |
| **TexJEPA-R** (`v5`) | Noise + four register tokens + tighter masking | **0.930** | **0.920** | 0.177 |
| **TexJEPA-C** (`v6`) | Noise + patch variance/covariance regularization | 0.916 | 0.906 | 0.040 |

These results use the post-training comparison's checkpoint selections, distinct from the H/300 comparison above. Sources: [Table XI](results/paper/csv/table_xi.csv) and [Table XII](results/paper/csv/table_xii.csv).

TexJEPA-R gives the strongest clean and light-noise readout among these variants. TexJEPA-C has the lowest drift at σ = 0.20, but substantially lower lesion sensitivity. The [method guide](docs/METHODS.md#texjepa-post-training) gives the objectives, masking changes, and checkpoint lineage.

<details>
<summary><strong>View the full post-training comparison</strong></summary>

![Reported post-training AUROC, cosine drift, and lesion sensitivity for v3.1, v4, v5, and v6.](results/paper/figures/post_training_results.png)

Download the figure as [PDF](results/paper/figures/post_training_results.pdf) or [SVG](results/paper/figures/post_training_results.svg). All 13 manuscript tables, supplementary values, and Figure 4 annotations are available in the [results package](results/paper/README.md).

</details>

## Quick start

**Linux or macOS · Python 3.11+ · no dataset or pretrained weights needed**

### 1. Get the code and inspect the results

```sh
git clone https://github.com/Eric-lfmself/TexJEPA.git
cd TexJEPA
python3 scripts/export_paper_results.py --check
```

This first check uses only the Python standard library. It validates 13 tables, 26 supplementary records, 90 figure annotations, and their exports.

### 2. Install and run the CPU demonstration

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m scripts.run_all --dry-run --profile smoke --output outputs/smoke
```

The demo runs the diagnostic workflow with synthetic images and small randomly initialized models. It writes `outputs/smoke/results.json` and a linked table index at `outputs/smoke/tables/INDEX.md`. Demo outputs are labeled `synthetic_smoke`; our experimental measurements are in [`results/paper/`](results/paper/README.md).

### 3. Explore figures or run the tests

```sh
python -m scripts.plot_paper_results --output outputs/paper_figures
python -m pytest
```

Figure generation reads the committed experimental values. Neither command downloads medical images or model weights. For GPU experiments, install a PyTorch build appropriate to the execution machine before installing the package.

## Run an experiment

We support native **I-JEPA / MAE pretraining**, **TexJEPA post-training**, and downstream evaluation through local **I-JEPA, MAE, EVA-X, RAD-DINO, and custom backbone adapters**. Evaluation includes linear and MLP probes, partial fine-tuning, image and token drift, lesion occlusion, frequency interventions, input smoothing, and noise-consistency adapters.

1. Prepare the images, label manifest, and checkpoints on the execution machine using the [data contract](docs/DATA_CONTRACT.md) and [backbone recipes](docs/BACKBONES.md).
2. Edit [`configs/experiment.example.json`](configs/experiment.example.json). Replace the `/srv/texjepa/...` placeholders, select the device and preprocessing, and enable the desired protocols.
3. Plan, validate, and run:

```sh
python -m scripts.run_experiments --config configs/experiment.example.json --plan
python -m scripts.run_experiments --config configs/experiment.example.json --validate-inputs
python -m scripts.run_experiments --config configs/experiment.example.json --execute
```

`--plan` lists the work without loading images, models, or a GPU. `--validate-inputs` checks manifests, splits, and file availability. Execution writes resumable checkpoints and complete result bundles. See the [execution guide](docs/EXECUTION.md) for pretraining, resume, evaluation-only runs, and output locations.

For external backbones, install `python -m pip install -e '.[backbones]'` and follow the source-specific dependency instructions. For downstream evaluation only, set `post_training.enabled` to `false`. Native post-training requires a compatible **full native I-JEPA checkpoint**, including the predictor and target encoder.

## Documentation

| I want to… | Start here |
|---|---|
| Understand the objectives and metrics | [Methods](docs/METHODS.md) |
| Configure, train, resume, or evaluate | [Execution guide](docs/EXECUTION.md) · [Example configuration](configs/experiment.example.json) |
| Prepare data and annotated lesions | [Data contract](docs/DATA_CONTRACT.md) |
| Load a pretrained encoder | [Backbone recipes](docs/BACKBONES.md) |
| Inspect the experimental measurements | [Results guide](results/paper/README.md) · [All tables](results/paper/TABLES.md) · [JSON](results/paper/tables.json) |
| Find dataset and upstream code links | [Sources](docs/SOURCES.md) |

<details>
<summary><strong>Code structure</strong></summary>

```text
configs/         Experiment settings and validation
models/          Encoders, predictors, checkpoint adapters, and post-training
training/        Optimization, checkpoint state, and resume
probing/         Linear, MLP, and partial fine-tuning protocols
perturbations/   Spatial and frequency interventions
metrics/         Classification and representation metrics
evaluation/      Robustness, lesion, and token diagnostics
scripts/         Training, experiment, and export entry points
report/          Tables and figures from saved measurements
results/paper/   Aggregate experimental results and figures
tests/          Synthetic CPU tests
```

</details>

## Availability and citation

We release code and aggregate experimental results for *When Texture Becomes the World: Texture-aware JEPA for Chest X-ray Representation Learning*. The manuscript remains private and unsubmitted. The results package preserves the displayed values, source locations, and precision; it contains aggregate measurements rather than per-image predictions or training logs. Chest radiographs and pretrained checkpoints are not bundled; access links are listed in [Sources](docs/SOURCES.md).

A provisional title-based citation is available in [CITATION.bib](CITATION.bib). For questions about the code or results, [open an issue](https://github.com/Eric-lfmself/TexJEPA/issues).
