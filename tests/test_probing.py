"""Verify actual updates and freeze boundaries for all three Table III protocols."""
import torch
import pytest
from data import DummyCXRDataset, collate_cxr
from models.vit_mini import MiniViT
from probing import ProbeConfig, ProbeModel, fit_probe, positive_class_weights
from probing.trainer import build_optimizer

@pytest.mark.parametrize("protocol", ["linear", "mlp", "partial_ft"])
def test_update_scope(protocol):
    encoder = MiniViT()
    before = {n:p.detach().clone() for n,p in encoder.named_parameters()}
    ds=DummyCXRDataset(4)
    batches=[collate_cxr([ds[0],ds[1]]),collate_cxr([ds[2],ds[3]])]
    model, result = fit_probe(encoder,batches,ProbeConfig(protocol=protocol,epochs=1))
    assert result["weight_source"] == "train_split_only"
    assert torch.isfinite(torch.tensor(result["history"][0]["train_loss"]))
    changed=[n for n,p in encoder.named_parameters() if not torch.equal(p,before[n])]
    if protocol == "partial_ft":
        assert changed and all(n.startswith(("blocks.","norm.")) for n in changed)
        assert not encoder.patch_embed.weight.requires_grad
    else:
        assert changed == [] and all(not p.requires_grad for p in encoder.parameters())
    assert model(torch.stack([ds[0]["image"],ds[1]["image"]])).shape == (2,15)


def test_partial_lr_and_positive_weights():
    model=ProbeModel(MiniViT(),ProbeConfig(protocol="partial_ft"))
    optimizer,_=build_optimizer(model)
    assert optimizer.param_groups[0]["lr"] < optimizer.param_groups[1]["lr"]
    weights,undefined=positive_class_weights(torch.tensor([[0,1,0],[1,1,0],[0,1,0]]))
    assert weights.tolist()==[2,1,1] and undefined==[1,2]
    with pytest.raises(ValueError,match="binary"):
        positive_class_weights([[float("nan")]])
