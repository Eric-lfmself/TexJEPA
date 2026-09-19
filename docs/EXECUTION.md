# Experiment execution

We provide separate entry points for synthetic checks, native pretraining, and experiments with local data and checkpoints. Run the commands below from the repository root after installing the package as described in the [README](../README.md).

## Configure an experiment

[experiment.example.json](../configs/experiment.example.json) includes the model list, probe protocols, optional post-training, and report settings. Replace every placeholder path with an existing path on the execution machine.

| Configuration | Required choice |
|---|---|
| `runtime` | Device, batch size, seed, and a dedicated output directory |
| `data.manifest`, `data.image_root` | Labeled image manifest and image directory |
| `data.split.path` | Explicit split-ID file, or `null` to create a deterministic split |
| `data.image_size`, `data.intensity` | Image size and an explicit intensity policy |
| `models` | Checkpoint, source implementation, normalization, and pooling contract for each encoder |
| `post_training` | Enable or disable native post-training and specify its full parent checkpoint |
| `protocols` | Any of `linear`, `mlp`, and `partial_ft` |

Relative paths are resolved against the configuration file's directory. We recommend a new output directory whenever the code, configuration, or inputs change. Labeled downstream data and unlabeled pretraining data have separate manifests; see the [data contract](DATA_CONTRACT.md).

The example uses a common 224-pixel input and identity normalization. These are explicit example settings. Select preprocessing appropriate to the checkpoints and experimental comparison; the manuscript does not specify every normalization or augmentation parameter. The RAD-DINO recipe itself defaults to 518 pixels, so overriding it to 224 pixels is a recorded experimental choice.

## Plan and validate

```sh
python -m scripts.run_experiments --config configs/experiment.example.json --plan
python -m scripts.run_experiments --config configs/experiment.example.json --validate-inputs
```

`--plan` lists tasks and required assets without decoding images, constructing models, or initializing a GPU. `--validate-inputs` additionally checks manifests, split membership, and file availability. Weight contents and tensor compatibility are checked during model loading.

For evaluation through external backbone adapters, set `post_training.enabled` to `false` unless a compatible full native parent checkpoint is also available. An official I-JEPA encoder checkpoint does not contain the full native predictor/encoder training state required by the post-training runner. [Backbones](BACKBONES.md) describes the loading contracts.

## Execute, resume, or evaluate

```sh
python -m scripts.run_experiments --config configs/experiment.example.json --execute
python -m scripts.run_experiments --config configs/experiment.example.json --resume
python -m scripts.run_experiments --config configs/experiment.example.json --eval-only
```

The configured workflow can include post-training, probing, perturbation tests, lesion occlusion with five controls, smoothing and augmentation comparisons, noise-consistency adaptation, token drift, and gradient-times-activation diagnostics. Diagnostics that require a linear readout are generated only when the linear protocol is selected. Optional stages can be disabled in the configuration.

Checkpoints store completed-epoch model, optimizer, scheduler, random-state, and sampler state. Resume requires the same experiment and input/source fingerprints. Evaluation-only mode requires completed probe and adapter checkpoints. A process lock prevents two jobs from writing to the same output directory; a remaining lock file alone does not indicate an active process.

The output directory contains:

```text
run_state.json            # Current execution status
jobs/                     # Probe and diagnostic training artifacts
post_training/            # Native variant checkpoints, when enabled
publications/run-*/       # Completed result, table, and figure bundles
CURRENT.json              # Pointer to the latest complete bundle
```

`CURRENT.json` advances only after the result bundle is complete. A failed run retains the previous complete bundle. The runner records configuration, source and manifest hashes, checkpoint hashes, and image file size/modification-time fingerprints. Image fingerprints are file metadata, not content hashes of every image.

New experiment measurements use `measured_local`; bounded fixture runs use `synthetic_smoke`. These labels describe how a run was produced. They do not associate a new checkpoint or run with the aggregate values in [results/paper](../results/paper/README.md).

## Native I-JEPA and MAE pretraining

Edit [pretrain.example.json](../configs/pretrain.example.json) for an unlabeled image manifest, output directory, device, and training schedule:

```sh
python -m scripts.pretrain --config configs/pretrain.example.json
python -m scripts.pretrain --config configs/pretrain.example.json --execute
python -m scripts.pretrain --config configs/pretrain.example.json --execute --resume
```

The first command inspects the unlabeled manifest and prints a plan. Set `architecture` to `mae` to use MAE. The default I-JEPA example uses `variant: "v3.1"`; its completed epoch-201 checkpoint is the parent for `v4`. Both `v5` and `v6` branch from `v4` at epoch 50. The runner checks parent lineage and completed epochs.

This path uses our native encoder and predictor structures. External repository checkpoints require matching architecture and weight mappings before they can serve as full native training checkpoints. Selecting an external encoder for downstream evaluation does not convert its predictor or optimizer state.

To export an encoder from a full native checkpoint:

```sh
python -m scripts.export_encoder \
  --checkpoint /srv/texjepa/checkpoints/native_ijepa/epoch_0201.pt \
  --component context_encoder \
  --output /srv/texjepa/checkpoints/encoder_0201.pt
```

Use `--component encoder` for MAE, or `target_encoder` for the I-JEPA target branch. Configure the exported file with `backend: "native"` and `checkpoint.state_key: "model_state_dict"`. Export preserves provenance and completed-epoch information and refuses to overwrite an existing file.

## Export figures

```sh
python -m report.build_figures path/to/results.json --output path/to/new_figures
```

Figure generation reads saved measurements. Missing AUROC values remain gaps; lesion intervals use the saved image-cluster bootstrap estimates. Figures distinguish probe protocols and use a shared `[0, 2]` scale for token cosine drift. The exports provide diagnostic layouts in PNG, PDF, and SVG.

## Synthetic checks

```sh
python -m pytest
python -m scripts.run_all --dry-run --profile smoke --output outputs/smoke
```

The test suite and smoke workflow use synthetic inputs and small CPU models. They cover metric definitions, loading contracts, data validation, checkpoint/resume behavior, and reporting. Real-data jobs use `runtime.profile: "experiment"`; the separate `fixture` profile imposes small model, image, batch, and epoch limits. Experiment mode rejects known synthetic checkpoint provenance.
