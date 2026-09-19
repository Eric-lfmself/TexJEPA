"""CPU numerical tests of Sec. V-H drift maps and weak spatial saliency."""

import torch
from torch import nn

from evaluation.token_diagnostics import (compare_token_drift, gradient_activation_saliency,
                                          token_cosine_drift)
from models.vit_mini import MiniViT


class LocalPatchEncoder(nn.Module):
    grid_size = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_tokens(self, x):
        means = torch.nn.functional.avg_pool2d(x, 8).flatten(2).transpose(1, 2)
        return torch.cat([means * self.scale, torch.ones_like(means)], dim=-1)


def test_token_drift_geometry_and_localization():
    clean = torch.tensor([[[1., 0], [1., 0], [0., 0], [1., 0]]])
    changed = torch.tensor([[[1., 0], [0., 1], [0., 0], [-1., 0]]])
    values = token_cosine_drift(clean, changed, grid_size=2)
    assert values[0, 0, 0] == 0 and values[0, 0, 1] == 1 and values[0, 1, 1] == 2
    assert torch.isnan(values[0, 1, 0])
    images = torch.zeros(2, 1, 32, 32, device="cpu")
    altered = images.clone()
    altered[:, :, :8, :8] = 1
    result = compare_token_drift(LocalPatchEncoder(), images, altered)
    assert result["drift_map"].shape == (2, 4, 4)
    assert (result["drift_map"] > 1e-8).sum().item() == 2
    assert result["valid_tokens"] == 32


def test_gradient_times_activation_known_value_and_no_parameter_mutation():
    encoder = LocalPatchEncoder().cpu()
    head = nn.Linear(2, 2, bias=False).cpu()
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[2., 0], [-3., 0]]))
    encoder.train()
    encoder.scale.grad = torch.tensor(9.)  # Existing training gradients must remain untouched.
    image = torch.ones(2, 1, 32, 32, device="cpu")
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    result = gradient_activation_saliency(encoder, head, image, [0, 1])
    assert torch.equal(result["saliency_map"][0], torch.full((4, 4), 2 / 16))
    assert torch.equal(result["saliency_map"][1], torch.full((4, 4), -3 / 16))
    assert torch.allclose(result["input_gradients"][0], torch.full((1, 32, 32), 2 / 1024))
    assert encoder.training and encoder.scale.requires_grad and encoder.scale.grad.item() == 9
    assert all(torch.equal(before[key], value) for key, value in encoder.state_dict().items())
    assert "not_causal" in result["interpretation"]


def test_mini_vit_saliency_excludes_registers_and_preserves_weights():
    encoder = MiniViT(image_size=32, patch_size=8, in_channels=1, embed_dim=8,
                      depth=2, num_heads=2, num_register_tokens=4).cpu()
    head = nn.Linear(8, 2).cpu()
    image = torch.rand(2, 1, 32, 32, device="cpu")
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    result = gradient_activation_saliency(encoder, head, image, 0)
    assert result["saliency_map"].shape == (2, 4, 4)
    assert torch.isfinite(result["input_gradients"]).all()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(torch.equal(before[key], value) for key, value in encoder.state_dict().items())
