# Reading the TexJEPA framework

The framework is a schematic of our implementation. Image tiles and chest outlines illustrate transformations and masking; they are not patient images or experimental attribution maps.

- **Feature flow:** the clean image splits into a corrupted context view and a clean target view. Context target patches are removed before context-encoder attention. The predictor uses visible context tokens and target-position queries.
- **EMA target:** the target encoder processes the full clean image. We normalize its features, select target blocks, and stop their gradients. The target parameters follow the context encoder through an exponential moving average after the optimizer step.
- **Objectives:** predicted and target embeddings meet at the mean Smooth-L1 objective. TexJEPA-C adds variance and covariance penalties on visible context patch tokens before the predictor.
- **Lineage:** I-JEPA-H/201 initializes TexJEPA-N. TexJEPA-R and TexJEPA-C both branch from the TexJEPA-N epoch-50 checkpoint. Registers and tighter masks belong to R; patch variance/covariance regularization belongs to C.
- **Evaluation:** classification and lesion readouts use a probe; representation drift is measured directly on encoder features. Lesion sensitivity compares lesion occlusion with five area-matched non-lesion controls.

See [Methods](../METHODS.md) for full objective definitions, mask settings, metric conventions, and implementation choices. The [framework](framework.svg) is editable vector artwork. Our palette follows the [NPG-inspired color family](https://nanx.me/ggsci/reference/pal_npg.html): muted blue for the online branch, teal for the clean target, and coral accents for corruption and auxiliary regularization. In the model lineage and result gallery, N is cyan, R is teal, and C is coral. The method geometry and labels encode the meaning in addition to color.
