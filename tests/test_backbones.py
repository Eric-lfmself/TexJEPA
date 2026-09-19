"""Tiny CPU contract checks. Optional libraries and official weights are never downloaded."""
from argparse import Namespace
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.backbones import (BackboneAdapter, build_backbone, validate_backbone_spec,
                              _eva_x_convert)
from models.checkpoints import CheckpointError
from models.vit_mini import MiniViT, gather_tokens
from probing.trainer import ProbeConfig, ProbeModel


NORM = {"mean": [.5, .5, .5], "std": [.25, .25, .25]}


def native(**updates):
    spec = {"backend": "native", "family": "native", "use_random_init": True,
            "image_size": 8, "patch_size": 4,
            "kwargs": {"embed_dim": 8, "depth": 3, "num_heads": 2}, "normalization": NORM}
    spec.update(updates)
    return spec


class TinyTokens(nn.Module):
    """Distinct CLS/register/patch outputs and a differentiable real tiny encoder."""
    def __init__(self, registers=0):
        super().__init__()
        self.embed_dim = 8
        self.core = MiniViT(image_size=8, patch_size=4, embed_dim=8, depth=3, num_heads=2)
        self.num_register_tokens = registers
        self.fc_norm = nn.LayerNorm(8)

    @property
    def blocks(self):
        return self.core.blocks

    @property
    def norm(self):
        return self.core.norm

    def forward_features(self, x):
        self.last_input = x.detach().clone()
        patch = self.core.forward_tokens(x)
        cls = patch.mean(1, keepdim=True) + 10
        reg = torch.full((len(x), self.num_register_tokens, 8), 100., device=x.device)
        return torch.cat([cls, reg, patch], 1)

    def forward_head(self, tokens, pre_logits=False):
        assert pre_logits
        return self.fc_norm(tokens[:, 1 + self.num_register_tokens:].mean(1))


def token_spec(**updates):
    spec = {"backend": "timm", "family": "custom", "architecture": "tiny_test",
            "use_random_init": True, "image_size": 8, "patch_size": 4,
            "normalization": NORM, "pooling": "mean_patch"}
    spec.update(updates)
    return spec


def test_native_general_shape_metadata_and_no_alias_state():
    model = build_backbone(native())
    x = torch.rand(2, 3, 8, 8)
    assert model(x).shape == (2, 8)
    assert model.forward_tokens(x).shape == (2, 4, 8)
    assert model.grid_size == 2 and model.num_patches == 4
    assert model.provenance["is_paper_pretrained_model"] is False
    assert model.provenance["mock"] is True
    assert not any(k.startswith("blocks.") or k.startswith("norm.") for k in model.state_dict())
    assert build_backbone(native(kwargs={"embed_dim": 24, "depth": 4, "num_heads": 3})).embed_dim == 24


def test_native_masked_context_does_not_see_removed_patch():
    model = build_backbone(native()).eval()
    x = torch.rand(2, 3, 8, 8)
    changed = x.clone()
    changed[:, :, 4:, 4:] += 100
    indices = torch.tensor([[0, 1], [0, 1]])
    assert torch.equal(model.forward_tokens(x, indices), model.forward_tokens(changed, indices))


@pytest.mark.parametrize("update", [
    {"family": "eva_x"}, {"image_size": True}, {"normalization": {"mean": [0, 0, 0], "std": [1, 0, 1]}},
    {"kwargs": {"pretrained": True}}, {"use_random_init": "yes"}, {"typo": 1},
    {"kwargs": {"embed_dim": 7, "num_heads": 2}}, {"pooling": "cls"},
    {"interface": {"prefix_tokens": 1}}, {"interface": {"kind": "huggingface"}},
    {"architecture": "official_ijepa"}, {"kwargs": {"img_size": 224}},
])
def test_static_validation_rejects_invalid_specs(update):
    with pytest.raises(ValueError):
        validate_backbone_spec(native(**update))


def test_dry_validation_is_allocation_and_file_free(monkeypatch):
    monkeypatch.setattr(torch.nn, "Linear", lambda *a, **k: pytest.fail("allocation during validation"))
    spec = validate_backbone_spec({"recipe": "ijepa_huge_patch14", "repo_path": "/missing/ijepa",
                                  "trust_local_code": True, "checkpoint": {"path": "/missing/weights.pth"},
                                  "normalization": NORM})
    assert spec["factory"] == "vit_huge" and spec["interface"]["kind"] == "ijepa"


def test_normalization_preserves_fft_negative_values():
    underlying = TinyTokens()
    model = BackboneAdapter(underlying, validate_backbone_spec(token_spec()), {})
    x = torch.full((2, 3, 8, 8), -.25)
    model(x)
    assert torch.equal(underlying.last_input, torch.full_like(x, -3))
    with pytest.raises(ValueError, match="floating"):
        model(torch.zeros(2, 3, 8, 8, dtype=torch.uint8))


def test_prefix_tokens_cls_pooling_and_unsupported_masks():
    underlying = TinyTokens(registers=2)
    spec = token_spec(interface={"prefix_tokens": 3, "cls_index": 0})
    model = BackboneAdapter(underlying, validate_backbone_spec(spec), {})
    x = torch.rand(2, 3, 8, 8)
    assert model.forward_tokens(x).shape == (2, 4, 8)
    assert torch.allclose(model(x), model.forward_tokens(x).mean(1))
    model.spec["pooling"] = "cls"
    assert torch.allclose(model(x), model.forward_tokens(x).mean(1) + 10)
    with pytest.raises(ValueError, match="leak"):
        model.forward_tokens(x, torch.tensor([[0], [0]]))


def test_partial_ft_final_blocks_and_both_norms_receive_gradients():
    underlying = TinyTokens()
    spec = token_spec(pooling="model_head", interface={"norm_paths": ["norm", "fc_norm"]})
    model = BackboneAdapter(underlying, validate_backbone_spec(spec), {})
    probe = ProbeModel(model, ProbeConfig(protocol="partial_ft", num_classes=2, hidden_dim=6, dropout=0))
    assert not next(model.blocks[0].parameters()).requires_grad
    assert next(model.blocks[-1].parameters()).requires_grad
    assert underlying.fc_norm.weight.requires_grad and underlying.norm.weight.requires_grad
    probe(torch.rand(2, 3, 8, 8)).square().sum().backward()
    assert next(model.blocks[-1].parameters()).grad is not None
    assert underlying.fc_norm.weight.grad is not None
    assert not model.training and not model.blocks[0].training and model.blocks[-1].training


def test_strict_local_weight_selection_and_hash(tmp_path):
    base = MiniViT(image_size=8, patch_size=4, embed_dim=8, depth=3, num_heads=2)
    weights = tmp_path / "tiny.pt"
    torch.save({"encoder": {"module." + k: v for k, v in base.state_dict().items()}}, weights)
    cp = {"path": str(weights), "state_key": "encoder", "strip_prefix": "module."}
    model = build_backbone(native(checkpoint=cp, use_random_init=False))
    assert model.provenance["mock"] is False
    assert len(model.provenance["checkpoint_sha256"]) == 64
    assert torch.equal(model.model.pos_embed, base.pos_embed)
    for change, message in (({"sha256": "0" * 64}, "SHA256"), ({"state_key": "missing"}, "state_key"),
                            ({"strip_prefix": "bad."}, "strip_prefix")):
        with pytest.raises(CheckpointError, match=message):
            build_backbone(native(checkpoint={**cp, **change}, use_random_init=False))
    bad = dict(base.state_dict()); bad.pop("norm.weight")
    torch.save(bad, weights)
    with pytest.raises(CheckpointError, match="missing"):
        build_backbone(native(checkpoint=str(weights), use_random_init=False))
    torch.save({"model": base.state_dict(), "args": Namespace(epochs=300, device="cuda")}, weights)
    restored = build_backbone(native(checkpoint={"path": str(weights), "state_key": "model"}, use_random_init=False))
    assert torch.equal(restored.model.pos_embed, base.pos_embed)


def test_local_factory_uses_declared_local_source_and_ijepa_pre_attention_mask(tmp_path):
    name = "local_ijepa_" + tmp_path.name.replace("-", "_")
    source = tmp_path / (name + ".py")
    source.write_text('''from models.vit_mini import MiniViT
class Encoder(MiniViT):
    def forward(self, x, masks=None):
        return self.forward_tokens(x, None if masks is None else masks[0])
def make():
    return Encoder(image_size=8, patch_size=4, embed_dim=8, depth=3, num_heads=2)
''')
    spec = {"backend": "local_factory", "family": "ijepa", "repo_path": str(tmp_path),
            "module": name, "factory": "make", "trust_local_code": True, "use_random_init": True,
            "image_size": 8, "patch_size": 4, "normalization": NORM, "interface": {"kind": "ijepa"}}
    model = build_backbone(spec).eval()
    assert model.provenance["source_file"] == str(source)
    assert len(model.provenance["source_python_tree_sha256"]) == 64
    x = torch.rand(2, 3, 8, 8); changed = x.clone(); changed[:, :, 4:, 4:] += 10
    ix = torch.tensor([[0, 1], [0, 1]])
    assert torch.equal(model.forward_tokens(x, ix), model.forward_tokens(changed, ix))
    with pytest.raises(ValueError, match="trust_local_code"):
        build_backbone({**spec, "trust_local_code": False})


def test_mae_encoder_preserves_patch_order_and_masks_before_attention():
    class Patch(nn.Module):
        def forward(self, x):
            return x.unfold(2, 4, 4).unfold(3, 4, 4).mean((-1, -2)).flatten(2).transpose(1, 2)
    model = nn.Module()
    model.patch_embed = Patch()
    model.pos_embed = nn.Parameter(torch.zeros(1, 5, 3))
    model.cls_token = nn.Parameter(torch.ones(1, 1, 3) * 9)
    model.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    model.norm = nn.Identity()
    spec = token_spec(interface={"kind": "mae", "embed_dim_path": "pos_embed"})
    wrapped = BackboneAdapter(model, validate_backbone_spec(spec), {})
    x = torch.arange(4.).reshape(1, 1, 2, 2).repeat_interleave(4, 2).repeat_interleave(4, 3).expand(2, 3, 8, 8)
    expected = torch.tensor([-2., 2., 6., 10.])
    assert torch.equal(wrapped.forward_tokens(x)[0, :, 0], expected)
    ix = torch.tensor([[2, 0], [3, 1]])
    assert torch.equal(wrapped.forward_tokens(x, ix), gather_tokens(wrapped.forward_tokens(x), ix))


def test_timm_forces_no_pretrained_and_rejects_official_label(monkeypatch):
    calls = []
    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        return TinyTokens()
    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=create_model, __version__="mock"))
    model = build_backbone(token_spec())
    assert calls == [("tiny_test", {"pretrained": False})]
    assert model(torch.rand(2, 3, 8, 8)).shape == (2, 8)
    for update in ({"architecture": "hf-hub:user/repo"}, {"kwargs": {"pretrained": True}}, {"family": "eva_x"}):
        with pytest.raises(ValueError):
            build_backbone(token_spec(**update))


def test_huggingface_forces_offline_and_checks_missing_weights(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "dinov2_with_registers", "num_register_tokens": 2}))
    (tmp_path / "model.safetensors").write_bytes(b"mock only, not model weights")
    model = TinyTokens(registers=2)
    model.config = SimpleNamespace(hidden_size=8)
    model.encoder = SimpleNamespace(layer=model.blocks)
    model.layernorm = model.norm
    model.forward = lambda pixel_values, return_dict: SimpleNamespace(last_hidden_state=model.forward_features(pixel_values))
    calls, info = [], {}
    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return model, info
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=from_pretrained)))
    spec = {"backend": "huggingface", "family": "rad_dino", "model_path": str(tmp_path),
            "image_size": 8, "patch_size": 4, "normalization": NORM}
    wrapped = build_backbone(spec)
    assert wrapped.forward_tokens(torch.rand(2, 3, 8, 8)).shape == (2, 4, 8)
    assert wrapped.interface["prefix_tokens"] == 3
    options = calls[0][1]
    assert options["local_files_only"] is True and options["trust_remote_code"] is False
    assert options["force_download"] is False and options["ignore_mismatched_sizes"] is False
    assert options["weights_only"] is True and options["token"] is False
    info["missing_keys"] = ["encoder.layer.0.weight"]
    with pytest.raises(CheckpointError, match="strictly"):
        build_backbone(spec)
    with pytest.raises(FileNotFoundError, match="Missing local"):
        build_backbone({**spec, "model_path": str(tmp_path / "missing")})


def test_eva_official_key_mapping_is_explicit_and_detects_collision():
    value = torch.ones(3)
    state, audit = _eva_x_convert({"mask_token": value, "lm_head.weight": value,
                                 "norm.weight": value, "blocks.0.mlp.w12.weight": value,
                                 "blocks.0.attn.q_proj.weight": value, "blocks.0.attn.q_bias": value})
    assert "fc_norm.weight" in state and "blocks.0.mlp.fc1.weight" in state
    assert "blocks.0.attn.q_proj.bias" in state
    assert audit["discarded_pretraining_or_rope_keys"] == ["lm_head.weight", "mask_token"]
    assert audit["interpolation"] is False
    with pytest.raises(CheckpointError, match="collision"):
        _eva_x_convert({"blocks.0.mlp.w1.weight": value, "blocks.0.mlp.fc1_g.weight": value})
