# Methods and implementation choices

Our study evaluates chest X-ray representations along three main axes: clean discrimination, stability under perturbation, and sensitivity to annotated lesions. The implementation exposes these measurements separately so that a robustness gain is not mistaken for a gain in lesion sensitivity.

## Study design

The main comparison uses MIMIC-CXR-pretrained I-JEPA-H/300 and MAE-H/300, matched in backbone family and pretraining duration. I-JEPA uses a ViT-Huge/14 encoder with a 1,280-dimensional image representation. Downstream evaluation uses 15 binary labels and a deterministic seed-42, 12,000/3,000 split of VinDr-CXR/VinBigData.

| Protocol | Trainable parameters | Manuscript settings |
|---|---|---|
| Linear probe | Batch normalization and linear head | 100 epochs; SGD, momentum 0.9; cosine schedule |
| MLP probe | MLP head | Hidden width 512; GELU and dropout |
| Partial fine-tuning | Last two encoder blocks, final normalization, and MLP head | 15 epochs; AdamW; lower learning rate for the encoder |

We use binary cross-entropy with logits and compute positive-class weights from the training split. Bounding boxes are used for lesion diagnostics, not for probe supervision.

## Robustness and drift

Gaussian-noise evaluation uses standard deviations `0.05`, `0.10`, `0.20`, and `0.30`. Additional diagnostics apply Gaussian blur, brightness shifts, contrast scaling, and frequency-domain interventions.

For a clean image `x`, perturbed image `x̃`, and encoder `f`, we measure:

```text
ΔAUROC = AUROC(perturbed) − AUROC(clean)
drift(x, x̃) = 1 − cosine(f(x), f(x̃))
```

A negative ΔAUROC indicates a performance loss. Cosine drift ranges from 0 to 2; smaller values indicate more stable representations. Patch-token drift uses the same metric at corresponding spatial token positions, excluding CLS and register tokens.

Frequency tests include low-pass filtering at cutoff `0.15` and the bands `[0.00, 0.20]`, `[0.20, 0.45]`, and `[0.45, 1.00]`. The implementation supports band-stop and band-pass interventions. By default, radial frequency is normalized by the two-dimensional Nyquist corner; bands are lower-inclusive and upper-exclusive, except that the highest band includes 1. Inverse Fourier outputs are not clipped or rescaled. These conventions are explicit implementation choices because the manuscript does not fully specify the Fourier operator.

## Classification-aligned lesion occlusion

For each eligible image–class pair, we measure the target-class logit drop after occluding its annotated lesion. Five area-matched non-lesion control regions estimate the effect of occlusion alone:

```text
Drop(region) = logit(clean) − logit(occluded region)
Δlesion = Drop(lesion) − mean[Drop(control)]
```

A positive score means that removing annotated lesion evidence suppresses the disease logit more than removing matched control regions. We use a neutral fill of `0.5` by default. Cases without feasible controls are explicitly skipped. Confidence intervals use an image-cluster bootstrap, keeping contributions from the same image together. Fill, control sampling, and bootstrap settings are recorded with the run.

## TexJEPA post-training

The variants form a shared lineage:

```text
I-JEPA-H/201 (v3.1)
└── TexJEPA-N (v4): asymmetric context noise
    └── v4 epoch 50
        ├── TexJEPA-R (v5): register tokens + tighter masking
        └── TexJEPA-C (v6): patch variance/covariance regularization
```

**TexJEPA-N (`v4`).** A noisy context branch predicts a clean target branch. The target encoder follows the context encoder by exponential moving average, with stop-gradient targets and a Smooth-L1 prediction loss. Context perturbations include Gaussian noise, Poisson noise, and JPEG compression.

**TexJEPA-R (`v5`).** We retain asymmetric noise and add four learnable register tokens. The context scale changes from `[0.85, 1.0]` to `[0.65, 0.85]`, the number of target blocks increases from four to six, and target aspect ratios widen from `[0.75, 1.5]` to `[0.5, 2.0]`. Predictor depth remains 12. Register tokens participate in attention and are excluded from patch pooling.

**TexJEPA-C (`v6`).** We retain asymmetric prediction and regularize context patch tokens `z`:

```text
L = Lasymmetric + 1.00 × Lvariance(z) + 0.04 × Lcovariance(z)
```

There is no separate two-view VICReg invariance term. The implementation uses a standard-deviation hinge with target 1 and epsilon `1e-4`; covariance uses an `M − 1` denominator and the sum of squared off-diagonal entries divided by feature dimension.

## Additional diagnostics

Input smoothing and training augmentation test whether robustness can be improved before or during probe training. The noise-consistency adapter (NCA) keeps the encoder frozen and trains a residual feature adapter with feature consistency, prediction consistency, and optional supervised classification terms. Its architecture and loss weights are configurable. Gradient-times-activation maps provide a supplementary spatial diagnostic; they are not lesion annotations or causal evidence by themselves.

## Configuration and result interpretation

We distinguish settings stated in the manuscript from choices exposed by the implementation:

| Area | Explicit implementation choices |
|---|---|
| Native architecture | Positional embeddings, predictor width, MAE decoder settings, and constructor dimensions |
| Preprocessing | Resize, intensity conversion, channel normalization, and augmentation parameters |
| Optimization | Learning rates, dropout, weight decay, EMA schedule, and other settings not fully enumerated in the manuscript |
| Frequency tests | Radial normalization, operator, endpoint convention, and inverse-transform handling |
| Lesion analysis | Control placement, fill value, and image-cluster bootstrap |
| Post-training and NCA | Noise probabilities and strengths, adapter architecture, and loss settings |

The native I-JEPA implementation averages Smooth-L1 loss across target blocks with `beta=1`. The native MAE default normalizes patch pixels using population variance. Both are configurable implementations of the respective objectives; external checkpoint loading follows the separate contracts in [Backbones](BACKBONES.md).

[results/paper](../results/paper/README.md) contains aggregate values transcribed from our manuscript. New experiment outputs and synthetic test outputs have their own provenance. We preserve source-specific values when a figure and table do not establish identical evaluation settings; in particular, Figure 8's raw-drift values are not substituted for Table V's Gaussian-noise drift values. Dataset mapping, split membership, preprocessing, checkpoint lineage, and metric conventions must all be considered when comparing a new run with the reported results.
