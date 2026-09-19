# Study provenance and upstream sources

## TexJEPA

Our unpublished manuscript, *When Texture Becomes the World: Texture-aware JEPA for Chest X-ray Representation Learning*, describes the study design, methods, and reported measurements. Our manuscript is not distributed in this repository. The [reported-results package](../results/paper/README.md) links aggregate values to their manuscript locations. [Methods](METHODS.md) documents the implementation's metric conventions and configurable choices.

## Datasets

- **MIMIC-CXR:** [dataset paper](https://doi.org/10.1038/s41597-019-0322-0) and [PhysioNet access page](https://physionet.org/content/mimic-cxr/).
- **VinDr-CXR:** [dataset paper](https://doi.org/10.1038/s41597-022-01498-w) and [PhysioNet access page](https://physionet.org/content/vindr-cxr/).

Dataset images and annotations are not bundled. Each dataset's access requirements and terms apply. The [data contract](DATA_CONTRACT.md) explains how to represent authorized local data without inferring class mappings or reader aggregation.

## Model implementations

- **I-JEPA:** [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa).
- **MAE:** [facebookresearch/mae](https://github.com/facebookresearch/mae).
- **EVA-X:** [hustvl/EVA-X](https://github.com/hustvl/EVA-X).
- **RAD-DINO:** [Microsoft model card](https://huggingface.co/microsoft/rad-dino).
- **DINOv2:** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
- **VICReg:** [facebookresearch/vicreg](https://github.com/facebookresearch/vicreg).

The external backbone adapters use existing local source trees or model snapshots. Their source and weight provenance are recorded per run. See [Backbones](BACKBONES.md) for supported formats and compatibility requirements.
