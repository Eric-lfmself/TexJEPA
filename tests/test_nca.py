"""CPU numerical checks for Sec. V-I preprocessing and frozen NCA training."""

import torch
from torch import nn

from evaluation.mitigation import median_filter, gaussian_smoothing, evaluate_mitigations
from evaluation.robustness import PerturbationSpec
from models.nca import NoiseConsistencyAdapter, ResidualAdapter, nca_objective, train_nca


def test_smoothing_impulse_and_signed_constants():
    x = torch.full((2, 1, 32, 32), 0.5, device="cpu")
    x[:, :, 16, 16] = 1
    median = median_filter(x, 3)
    assert torch.equal(median, torch.full_like(x, 0.5))
    gaussian = gaussian_smoothing(x, 1)
    assert 0.5 < gaussian[0, 0, 16, 16].item() < 1
    assert abs((gaussian - 0.5).sum().item() - 1.0) < 1e-4
    signed = torch.full_like(x, -0.2)
    assert torch.allclose(gaussian_smoothing(signed), signed, atol=1e-6)
    assert torch.equal(median_filter(signed), signed)


def test_adapter_identity_and_loss_numeric():
    adapter = ResidualAdapter(4, 2).cpu()
    clean = torch.tensor([[1., 2, 3, 4], [2., 3, 4, 5]], device="cpu")
    assert torch.equal(adapter(clean), clean)
    loss = nca_objective(clean, clean + 1, torch.zeros(2, 1), torch.zeros(2, 1),
                         supervised_weight=0)
    assert loss["alignment"].item() == 1 and loss["consistency"].item() == 0
    assert loss["loss"].item() == 1


def test_nca_training_changes_adapter_head_but_freezes_encoder_and_buffers():
    torch.manual_seed(7)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 8), nn.BatchNorm1d(8)).cpu()
    model = NoiseConsistencyAdapter(encoder, 8, num_classes=2, hidden_dim=4).cpu()
    before_encoder = {key: value.clone() for key, value in encoder.state_dict().items()}
    before_adapter = {key: value.clone() for key, value in model.adapter.state_dict().items()}
    before_head = {key: value.clone() for key, value in model.head.state_dict().items()}
    batch = {"image": torch.rand(4, 1, 32, 32),
             "labels": torch.tensor([[0., 0], [1., 1], [0., 1], [1., 0]])}
    history = train_nca(model, [batch], epochs=2, max_steps=2)
    assert len(history) == 2 and all(torch.isfinite(torch.tensor(row["loss"])) for row in history)
    assert all(torch.equal(before_encoder[key], value) for key, value in encoder.state_dict().items())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    assert not encoder.training
    assert any(not torch.equal(before_adapter[key], value) for key, value in model.adapter.state_dict().items())
    assert any(not torch.equal(before_head[key], value) for key, value in model.head.state_dict().items())


def test_mitigation_robustness_includes_paired_clean_reference():
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 4)).cpu()
    head = nn.Linear(4, 1).cpu()
    batch = {"image": torch.rand(2, 1, 32, 32), "labels": torch.tensor([[0], [1]])}
    specs = [PerturbationSpec("gaussian_noise", 0.05, {"sigma": 0.05})]
    rows = evaluate_mitigations(encoder, head, [batch], specs, model_name="tiny")
    assert len(rows) == 6 and {row["mitigation"] for row in rows} == {"none", "median", "gaussian"}
    assert all(row["delta_auroc"] == 0 for row in rows if row["perturbation"] == "clean")
