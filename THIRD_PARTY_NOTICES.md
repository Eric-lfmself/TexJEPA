# Third-party notices

Our [proprietary notice](LICENSE) reserves only the rights we own. The components
identified below retain their separate licenses. Those licenses do not grant
rights to unrelated TexJEPA materials.

## EVA-X checkpoint conversion

The `_eva_x_convert` compatibility helper in `models/backbones.py` adapts the
key-conversion portion of EVA-X's `checkpoint_filter_fn`. We retain Apache-2.0
terms for this helper and preserve the upstream license in
[licenses/EVA-X-Apache-2.0.txt](licenses/EVA-X-Apache-2.0.txt).

- Upstream source: [hustvl/EVA-X, eva_x.py](https://github.com/hustvl/EVA-X/blob/eedd642c85339b25682ceb574b644cb9afe88e62/eva_x.py).
- Reviewed revision: `eedd642c85339b25682ceb574b644cb9afe88e62`.
- Source attribution: “by Jingfeng Yao”; “from HUST-VL”. The upstream file also
  credits [baaivision/EVA](https://github.com/baaivision/EVA) and
  [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models).
- Our modifications: retain key conversion without interpolation or resampling;
  record renamed and discarded keys; reject conversion collisions. Shape
  compatibility remains enforced by strict checkpoint loading.

The full upstream license is preserved verbatim, including its existing
copyright text. This notice does not assert that we own the upstream work.

## DejaVu fonts in exported figures

The PDF figures under `results/paper/figures/` contain embedded subsets of
DejaVu Sans fonts supplied with Matplotlib. The font software has separate
Bitstream Vera / DejaVu and applicable Arev notices, preserved in full in
[licenses/DejaVu.txt](licenses/DejaVu.txt). The DejaVu changes are identified as
public domain in that notice. Font permissions concern the font software and
do not grant permission to reuse our original figure content.

## External dependencies and resources

Libraries installed as dependencies, locally supplied backbone repositories,
datasets, and pretrained checkpoints remain subject to their own license and
access terms. Our repository does not relicense them. Consult the notices
provided with each resource and the links in [Sources](docs/SOURCES.md).
