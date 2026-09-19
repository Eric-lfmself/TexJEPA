"""Behavioral tests for latent prediction, pixel reconstruction and checkpoint safety."""
import copy
import pytest
import torch
from models import (MiniViT, IJEPA, MAE, save_checkpoint, load_checkpoint,
                    create_encoder, warm_start, CheckpointError)


@pytest.fixture(autouse=True)
def reproducible_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def indices(batch=2):
    return torch.tensor([[0, 1, 4, 5]]).expand(batch, -1), torch.tensor([[10, 11, 14, 15]]).expand(batch, -1)


@pytest.mark.parametrize("size,patch", [(32, 8), (64, 8)])
def test_vit_shapes_and_registers_excluded_from_pooling(size, patch):
    model = MiniViT(image_size=size, patch_size=patch, num_register_tokens=4)
    x = torch.randn(2, 3, size, size)
    counts = []
    hook = model.blocks[0].register_forward_pre_hook(lambda module, args: counts.append(args[0].shape[1]))
    tokens = model.forward_tokens(x)
    pooled = model(x)
    assert tokens.shape == (2, (size // patch)**2, 64)
    assert counts == [(size // patch)**2 + 4] * 2
    torch.testing.assert_close(pooled, tokens.mean(1))
    hook.remove()
    pooled.square().sum().backward()
    assert model.register_tokens.grad is not None
    assert model.register_tokens.grad.abs().sum() > 0


def test_jepa_target_stop_gradient_context_only_and_ema():
    model = IJEPA(MiniViT())
    x = torch.randn(2, 3, 32, 32)
    ctx, tgt = indices()
    result = model(x, ctx, tgt)
    assert result["predictions"].shape == result["targets"].shape == (2, 4, 64)
    assert not result["targets"].requires_grad
    result["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    noisy = x + torch.randn_like(x) * 0.1
    perturbed = model(x, ctx, tgt, context_images=noisy)
    torch.testing.assert_close(perturbed["targets"], result["targets"])
    assert not torch.allclose(perturbed["context_tokens"], result["context_tokens"])
    context = next(model.context_encoder.parameters())
    target = next(model.target_encoder.parameters())
    before = target.clone()
    with torch.no_grad():
        context.add_(0.2)
    model.update_ema(0.9)
    torch.testing.assert_close(target, before * 0.9 + context * 0.1)
    model.train()
    assert not model.target_encoder.training


def test_jepa_target_pixels_do_not_enter_context_and_target_positions_matter():
    model = IJEPA(MiniViT()).eval()
    x = torch.randn(2, 3, 32, 32)
    ctx, tgt = indices()
    result = model(x, ctx, tgt)
    changed = x.clone()
    changed[:, :, 16:, 16:] += 10
    other = model(changed, ctx, tgt)
    torch.testing.assert_close(result["context_tokens"], other["context_tokens"])
    torch.testing.assert_close(result["predictions"], other["predictions"])
    assert not torch.allclose(result["targets"], other["targets"])
    predictions = model.predictor(result["context_tokens"], ctx, torch.tensor([[2, 3, 6, 7]]).expand(2, -1))
    assert not torch.allclose(predictions, result["predictions"])


def test_jepa_multiple_blocks_and_invalid_overlap():
    model = IJEPA(MiniViT())
    ctx, tgt = indices()
    result = model(torch.randn(2, 3, 32, 32), ctx, [tgt[:, :2], tgt[:, 2:]])
    assert len(result["predictions"]) == 2
    assert torch.isfinite(result["loss"])
    with pytest.raises(ValueError, match="overlap"):
        model(torch.randn(2, 3, 32, 32), ctx, ctx)


def test_mae_patch_roundtrip_masked_loss_and_gradients():
    model = MAE(MiniViT())
    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(model.unpatchify(model.patchify(x)), x)
    ctx, _ = indices()
    result = model(x, visible_indices=ctx)
    assert result["predictions"].shape == (2, 16, 192)
    assert result["mask"].sum().item() == 24
    reference = (result["predictions"] - result["targets"]).square().mean(-1)[result["mask"]].mean()
    torch.testing.assert_close(result["loss"], reference)
    result["loss"].backward()
    assert model.encoder.patch_embed.weight.grad.abs().sum() > 0
    assert model.decoder_pred.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="masked"):
        model(x, visible_indices=torch.arange(16).expand(2, -1))


def test_checkpoint_strict_load_roundtrip_and_provenance(tmp_path):
    model = MiniViT()
    path = save_checkpoint(tmp_path / "local.pt", model, {"variant": "v3.1", "epoch": 201, "mock": True})
    restored = MiniViT()
    meta = load_checkpoint(path, restored, expected_metadata={"variant": "v3.1"})
    assert meta["epoch"] == 201
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    with pytest.raises(CheckpointError, match="provenance"):
        load_checkpoint(path, restored, expected_metadata={"variant": "v4"})
    with pytest.raises(CheckpointError, match="mismatch"):
        load_checkpoint(path, MiniViT(embed_dim=32))
    with pytest.raises(FileNotFoundError):
        create_encoder("ijepa", checkpoint=tmp_path / "missing.pt")
    with pytest.raises(CheckpointError, match="local checkpoint"):
        create_encoder("mini")
    with pytest.raises(CheckpointError, match="factory"):
        create_encoder("eva_x", checkpoint=path)
    assert create_encoder("rad_dino", use_random_init=True).provenance["mock"] is True


def test_warm_start_lineage_and_only_register_keys_added(tmp_path):
    base = IJEPA(MiniViT())
    path = save_checkpoint(tmp_path / "v4.pt", base, {"variant": "v4", "epoch": 50, "mock": True})
    reg_model = IJEPA(MiniViT(num_register_tokens=4))
    result = warm_start(path, reg_model, "v5")
    assert set(result["new_parameters"]) == {"context_encoder.register_tokens", "target_encoder.register_tokens"}
    torch.testing.assert_close(reg_model.context_encoder.register_tokens, reg_model.target_encoder.register_tokens)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(reg_model.state_dict()[key], value)
    with pytest.raises(CheckpointError, match="provenance"):
        warm_start(path, IJEPA(MiniViT()), "v4")
    save_checkpoint(path, base, {"variant": "v4", "epoch": 49})
    with pytest.raises(CheckpointError, match="epoch"):
        warm_start(path, IJEPA(MiniViT()), "v6")


@pytest.mark.parametrize("epoch", [None, 0, 200, 202])
def test_v4_rejects_missing_or_wrong_v31_parent_epoch_without_mutation(tmp_path, epoch):
    source, destination = IJEPA(MiniViT()), IJEPA(MiniViT())
    metadata = {"variant": "v3.1", "mock": True}
    if epoch is not None:
        metadata["epoch"] = epoch
    path = save_checkpoint(tmp_path / "wrong_parent.pt", source, metadata)
    before = copy.deepcopy(destination.state_dict())
    with pytest.raises(CheckpointError, match="epoch"):
        warm_start(path, destination, "v4")
    for key, value in before.items():
        torch.testing.assert_close(value, destination.state_dict()[key])


def test_v4_accepts_explicit_synthetic_v31_epoch201_fixture(tmp_path):
    source, destination = IJEPA(MiniViT()), IJEPA(MiniViT())
    path = save_checkpoint(tmp_path / "fixture.pt", source,
                           {"variant": "v3.1", "epoch": 201, "mock": True,
                            "evidence": "synthetic_smoke", "actual_training_steps": 0})
    provenance = warm_start(path, destination, "v4")
    assert provenance["warm_start"]["epoch"] == 201
    assert provenance["warm_start"]["mock"] is True
    assert provenance["warm_start"]["actual_training_steps"] == 0
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, destination.state_dict()[key])


def test_bad_keys_fail_before_any_model_mutation(tmp_path):
    model = MiniViT()
    before = copy.deepcopy(model.state_dict())
    path = save_checkpoint(tmp_path / "bad.pt", model, {"mock": True})
    payload = torch.load(path, weights_only=True)
    payload["model_state_dict"]["unrecognized.weight"] = torch.ones(1)
    payload["model_state_dict"]["pos_embed"].fill_(999)
    torch.save(payload, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, model)
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)
