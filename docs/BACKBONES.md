# Backbone adapters and checkpoints

We evaluate locally available encoders through a common `BackboneAdapter`. Model construction does not fetch source repositories, install packages, or download weights. Use the optional `backbones` extra for timm, Transformers, and safetensors, then match external dependency versions to the selected source repository.

## Shared interface

`models.backbones.build_backbone(spec)` builds a CPU model from a JSON-compatible dictionary. `validate_backbone_spec(spec, check_files=False)` normalizes and validates a specification without importing optional libraries or allocating model tensors; `check_files=True` additionally checks paths. Checkpoint contents are validated when the model is loaded.

The adapter exposes:

- `forward(images)`: image representations shaped `[B, D]`.
- `forward_tokens(images, indices=None)`: spatially ordered patch tokens shaped `[B, N, D]`, excluding CLS and register tokens.
- `embed_dim`, `blocks`, `norm`, `image_size`, `patch_size`, `grid_size`, `num_patches`, `num_register_tokens`, and `supports_masked_tokens`.
- `architecture_metadata()` and `provenance`.

Inputs are floating tensors `[B, C, H, W]`, ordinarily containing raw pixels in `[0, 1]`. The adapter applies `(pixels − mean) / std` only. Image resizing belongs to the dataset, and the adapter preserves out-of-range values produced by Fourier interventions.

| Pooling | Image representation |
|---|---|
| `mean_patch` | Mean of the final patch tokens |
| `cls` | Declared CLS token |
| `model_head` | `forward_head(..., pre_logits=True)`, preserving model-specific pooling and normalization |

`blocks` and `norm` reference the source modules so that partial fine-tuning updates the selected encoder parameters. EVA-X exposes both `norm` and `fc_norm`.

## Native models

The `native` backend uses our `MiniViT` implementation, which accepts both small test dimensions and larger configured dimensions. A small random example is:

```json
{
  "backend": "native",
  "family": "native",
  "use_random_init": true,
  "image_size": 32,
  "patch_size": 8,
  "in_channels": 3,
  "kwargs": {"embed_dim": 64, "depth": 2, "num_heads": 4, "num_register_tokens": 0},
  "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]}
}
```

Random initialization is explicit and recorded as `mock: true`. To load an unwrapped native encoder, replace `use_random_init` with:

```json
"checkpoint": {"path": "/srv/texjepa/encoder.pt", "state_key": "model_state_dict"}
```

Native encoder exports from `scripts.export_encoder` use this format. Full native I-JEPA training checkpoints also contain a predictor, target encoder, and training state; load them with the matching native training model. A saved `BackboneAdapter` uses `model.*` keys and includes normalization buffers, so it must be restored into the matching wrapper rather than treated as an unwrapped encoder.

The native training path and external repository adapters have different checkpoint contracts. External encoder loading supports downstream diagnostics; full native post-training requires a compatible native parent or an explicit, verified architecture/weight conversion.

## I-JEPA, MAE, and EVA-X from local repositories

Recipes select an existing source repository, constructor, token interface, and pooling method:

| Recipe | Source | Construction and pooling |
|---|---|---|
| `ijepa_huge_patch14` | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | `vit_huge`; mean patch; no CLS token |
| `mae_huge_patch14` | [facebookresearch/mae](https://github.com/facebookresearch/mae) | `mae_vit_huge_patch14`; full MAE checkpoint; mean patch |
| `eva_x_base_patch16` | [hustvl/EVA-X](https://github.com/hustvl/EVA-X) | `EVA_X` base architecture; model pooling and `fc_norm` |

These use `backend: "local_factory"`. Set `repo_path` and `trust_local_code: true` because the selected Python source executes in the current process. Use a fresh process when local repository module names conflict. Normalization must be supplied explicitly.

For I-JEPA:

```json
{
  "recipe": "ijepa_huge_patch14",
  "repo_path": "/srv/texjepa/code/ijepa",
  "trust_local_code": true,
  "checkpoint": {
    "path": "/srv/texjepa/checkpoints/ijepa.pth.tar",
    "state_key": "target_encoder",
    "strip_prefix": "module."
  },
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

The normalization above illustrates the ImageNet convention; choose values appropriate to the actual checkpoint. Select `encoder` or `target_encoder` to match the intended branch and use an empty prefix when the checkpoint has no `module.` prefix. Key selection and prefix removal are explicit; every selected key must match the declared prefix.

**MAE.** Use `recipe: "mae_huge_patch14"` and the actual state container, commonly `model`. The full checkpoint, including decoder tensors, is loaded strictly. Downstream features use the encoder with raster-order patch tokens. The adapter follows patch embedding, position embedding, visible-token selection, CLS insertion, blocks, and final normalization without MAE's random masking shuffle. Loading keeps `weights_only=True`; the standard-library `argparse.Namespace` used by official MAE argument metadata is explicitly allowed, while unsupported objects fail. The source repository may require an older timm API.

**EVA-X.** Use `recipe: "eva_x_base_patch16"` and `checkpoint.conversion: "eva_x_official"` for the supported MIM checkpoint format. Declare the actual `state_key` and `strip_prefix`. The adapter constructs `EVA_X` directly and checks loading itself. Conversion maps MLP/attention keys, separate query/value biases, and MIM `norm` to `fc_norm`. Discarded mask/head tensors and fixed RoPE keys are recorded individually. Remaining keys and shapes must match exactly; patch and position tensors are not silently interpolated.

Recipe constructor defaults can be overridden through `kwargs` and `interface`, subject to validation. Declared image and patch dimensions must match the constructor. Successful loading establishes tensor compatibility with the selected architecture; comparison with manuscript results also depends on data, preprocessing, and checkpoint lineage.

## RAD-DINO from a local model snapshot

```json
{
  "recipe": "rad_dino",
  "model_path": "/srv/texjepa/models/rad-dino",
  "normalization": {
    "mean": [0.5307, 0.5307, 0.5307],
    "std": [0.2583, 0.2583, 0.2583]
  }
}
```

The recipe defaults to 518-pixel inputs, 14-pixel patches, and CLS pooling. Resize/crop is controlled by the dataset pipeline. An override, such as the common 224-pixel input in the experiment example, is a distinct preprocessing choice.

`model_path` must contain local configuration and weight artifacts. The adapter uses Transformers' built-in DINOv2 or DINOv2-with-registers model with local-only loading, remote code disabled, and weight-only deserialization. Custom `auto_map` configurations and nonempty missing/unexpected/mismatched loading reports are rejected. Register count is read from the model configuration; CLS and registers are removed only from patch-token outputs. Configuration and weight files are hashed for provenance.

Use the [RAD-DINO model card](https://huggingface.co/microsoft/rad-dino) and the processor configuration packaged with the selected snapshot to establish its input contract and checkpoint provenance.

## Custom backbones

For another local architecture, declare `backend: "local_factory"`, `module`, `factory`, `kwargs`, and `interface`. The default interface expects `forward_features(x)` to return `[B, prefix + patches, D]`, plus `embed_dim`, `blocks`, and `norm` attributes.

Supported interface fields include `kind`, `method`, `output_key`, `prefix_tokens`, `cls_index`, `blocks_path`, `norm_paths`, `embed_dim_path`, `mask_argument`, and `mask_as_list`. Select a token dictionary entry through `output_key` when needed. Constructor loading/download arguments, such as `pretrained` and `checkpoint_path`, are rejected.

For a generic timm model, set `backend: "timm"`, `family: "custom"`, a local registry `architecture`, explicit token layout and normalization, and a compatible local checkpoint. Construction uses `pretrained=False`; hub-prefixed architecture names are rejected. Use the dedicated local recipe for the named model families above.

## Masking and provenance

Native and official I-JEPA interfaces select visible patches before attention. The MAE adapter also selects patches before its transformer blocks. Other adapters accept masked indices only when their declared contract supports pre-attention masking. Selecting tokens after full-image attention would change the training objective and is not used as a substitute.

Provenance records the effective architecture configuration, pooling, normalization, checkpoint SHA-256, explicit key conversions, and local source-file/tree hashes or model-snapshot artifact hashes. Optional packages are imported only by the corresponding backends. Tests exercise these contracts with tiny CPU models and controlled fixtures; use the experiment runner to record the actual environment and artifacts for each new measurement.
