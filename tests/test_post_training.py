"""Post-training mechanisms: asymmetric views, spatial masks and patch statistics."""
import pytest
import torch
from models import (MiniViT, IJEPA, PostTrainingObjective, BlockMaskSampler,
                    context_noise, patch_variance_covariance)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("noise_kind", ["gaussian", "poisson", "jpeg"])
def test_each_noise_applies_to_context_and_leaves_clean_targets(noise_kind):
    images = torch.rand(2, 3, 32, 32)
    original = images.clone()
    model = IJEPA(MiniViT())
    masks = BlockMaskSampler(4)(2)
    clean = model(images, masks["context_indices"], masks["target_indices"])
    objective = PostTrainingObjective("noise")
    result = objective(model, images, masks["context_indices"], masks["target_indices"],
                       noise_kind=noise_kind)
    assert result["context_images"].shape == images.shape
    assert not torch.equal(result["context_images"], images)
    assert result["context_images"].min() >= 0 and result["context_images"].max() <= 1
    torch.testing.assert_close(images, original)
    torch.testing.assert_close(result["target_images"], original)
    for target, reference in zip(result["targets"], clean["targets"]):
        torch.testing.assert_close(target, reference)
        assert not target.requires_grad
    torch.testing.assert_close(result["loss"], result["asymmetric_loss"])
    result["loss"].backward()
    assert model.context_encoder.patch_embed.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.target_encoder.parameters())


@pytest.mark.parametrize("kind", ["gaussian", "poisson"])
def test_noise_generator_reproducibility_and_domain(kind):
    x = torch.full((2, 1, 32, 32), 0.5)
    a = context_noise(x, kind, generator=torch.Generator().manual_seed(9))
    b = context_noise(x, kind, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="unnormalized"):
        context_noise(x - 1, kind)


def test_patch_vicreg_correct_sample_axis_and_no_extra_invariance_term():
    tokens = torch.tensor([[[0., 2.], [2., 0.]], [[4., 3.], [1., 1.]]], requires_grad=True)
    terms = patch_variance_covariance(tokens)
    flat = tokens.reshape(4, 2)
    expected_var = torch.relu(1 - torch.sqrt(flat.var(dim=0, unbiased=True) + 1e-4)).mean()
    centered = flat - flat.mean(0)
    covariance = centered.T @ centered / 3
    expected_cov = (covariance[0, 1]**2 + covariance[1, 0]**2) / 2
    torch.testing.assert_close(terms["variance"], expected_var)
    torch.testing.assert_close(terms["covariance"], expected_cov)
    assert terms["num_samples"] == 4 and terms["feature_dim"] == 2
    (terms["variance"] + .04*terms["covariance"]).backward()
    assert torch.isfinite(tokens.grad).all()
    collapsed = patch_variance_covariance(torch.zeros(2, 4, 8))
    torch.testing.assert_close(collapsed["variance"], torch.tensor(.99))
    assert collapsed["covariance"] == 0
    with pytest.raises(ValueError, match="two patch"):
        patch_variance_covariance(torch.zeros(1, 1, 8))


def test_v6_objective_matches_weighted_formula():
    model = IJEPA(MiniViT())
    objective = PostTrainingObjective("vicreg")
    masks = objective.make_mask_sampler(4)(2)
    result = objective(model, torch.rand(2, 3, 32, 32), masks["context_indices"],
                       masks["target_indices"], noise_kind="gaussian")
    expected = result["asymmetric_loss"] + result["variance_loss"] + .04*result["covariance_loss"]
    torch.testing.assert_close(result["loss"], expected)
    assert result["regularization"]["num_samples"] == result["context_tokens"].shape[0] * result["context_tokens"].shape[1]
    result["loss"].backward()
    assert torch.isfinite(model.context_encoder.patch_embed.weight.grad).all()


def test_v5_register_requirement_and_paper_mask_schedule():
    objective = PostTrainingObjective("register")
    baseline = BlockMaskSampler(16, variant="baseline")
    sampler = objective.make_mask_sampler(16)
    assert baseline.context_scale == (.85, 1.) and baseline.num_target_blocks == 4
    assert baseline.target_aspect_ratio == (.75, 1.5)
    assert sampler.context_scale == (.65, .85) and sampler.num_target_blocks == 6
    assert sampler.target_aspect_ratio == (.5, 2.)
    masks = objective.make_mask_sampler(4)(2)
    with pytest.raises(ValueError, match="four register"):
        objective(IJEPA(MiniViT()), torch.rand(2, 3, 32, 32),
                  masks["context_indices"], masks["target_indices"])
    model = IJEPA(MiniViT(num_register_tokens=4))
    result = objective(model, torch.rand(2, 3, 32, 32), masks["context_indices"],
                       masks["target_indices"], noise_kind="poisson")
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.context_encoder.register_tokens.grad.abs().sum() > 0


@pytest.mark.parametrize("grid,variant", [(4, "v5"), (8, "v4"), (16, "v5")])
def test_spatial_target_rectangles_no_target_visible_and_finite_fallback(grid, variant):
    sampler = BlockMaskSampler(grid, variant=variant, max_attempts=2, seed=10)
    result = sampler(3)
    context = result["context_indices"]
    assert context.shape[0] == 3 and context.shape[1] > 0
    for block_number, targets in enumerate(result["target_indices"]):
        assert not (context[:, :, None] == targets[:, None, :]).any()
        for b in range(3):
            top, left, h, w = result["metadata"]["geometry"][b]["target_rectangles"][block_number]
            expected = torch.tensor([r*grid+c for r in range(top, top+h) for c in range(left, left+w)])
            torch.testing.assert_close(targets[b], expected)
    if grid == 4:
        assert result["metadata"]["fallback_used"]
    repeated = BlockMaskSampler(grid, variant=variant, max_attempts=2, seed=10)(3)
    torch.testing.assert_close(result["context_indices"], repeated["context_indices"])


def test_pathological_mask_parameters_report_fallback_instead_of_hanging():
    sampler = BlockMaskSampler(2, target_scale=(1., 1.), context_scale=(1., 1.),
                               num_target_blocks=6, max_attempts=1)
    result = sampler(2)
    assert result["metadata"]["fallback_used"]
    assert result["context_indices"].shape == (2, 3)
    for target in result["target_indices"]:
        assert target.shape == (2, 1)
        assert not (result["context_indices"][:, :, None] == target[:, None, :]).any()
    with pytest.raises(ValueError, match="2x2"):
        BlockMaskSampler(1)


def test_normalization_runs_after_corruptions_for_both_branches():
    objective = PostTrainingObjective("noise", normalizer=lambda x: (x - .5) / .25)
    model = IJEPA(MiniViT())
    masks = objective.make_mask_sampler(4)(2)
    x = torch.rand(2, 3, 32, 32)
    output = objective(model, x, masks["context_indices"], masks["target_indices"], noise_kind="gaussian")
    reference = model((x - .5)/.25, masks["context_indices"], masks["target_indices"],
                      context_images=(output["context_images"] - .5)/.25)
    torch.testing.assert_close(output["loss"], reference["loss"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.])
@pytest.mark.parametrize("parameter", ["gaussian_sigma", "poisson_peak", "lambda_var", "lambda_cov",
                                       "variance_gamma", "variance_eps"])
def test_post_training_rejects_nonfinite_or_negative_configuration(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        PostTrainingObjective("vicreg", **{parameter: value})


@pytest.mark.parametrize("parameter", ["poisson_peak", "variance_gamma", "variance_eps"])
def test_post_training_requires_positive_peak_gamma_epsilon(parameter):
    with pytest.raises(ValueError, match=parameter):
        PostTrainingObjective("vicreg", **{parameter: 0.})


@pytest.mark.parametrize("quality", [0, 101, 75.5, 75., True, float("nan"), float("inf")])
def test_jpeg_quality_requires_bounded_integer(quality):
    with pytest.raises(ValueError, match="integer"):
        PostTrainingObjective(jpeg_quality=quality)
    with pytest.raises(ValueError, match="integer"):
        context_noise(torch.full((2, 1, 32, 32), .5), "jpeg", jpeg_quality=quality)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_direct_noise_and_vicreg_helpers_reject_invalid_parameters(value):
    images = torch.full((2, 1, 32, 32), .5)
    with pytest.raises(ValueError, match="sigma"):
        context_noise(images, "gaussian", sigma=value)
    with pytest.raises(ValueError, match="poisson_peak"):
        context_noise(images, "poisson", poisson_peak=value)
    with pytest.raises(ValueError, match="eps"):
        patch_variance_covariance(torch.rand(2, 4, 8), eps=value)
    with pytest.raises(ValueError, match="gamma"):
        patch_variance_covariance(torch.rand(2, 4, 8), gamma=value)


def test_zero_noise_and_zero_auxiliary_weights_remain_valid():
    images = torch.full((2, 1, 32, 32), .5)
    torch.testing.assert_close(context_noise(images, "gaussian", sigma=0.), images)
    objective = PostTrainingObjective("vicreg", gaussian_sigma=0., lambda_var=0., lambda_cov=0.)
    assert objective.gaussian_sigma == objective.lambda_var == objective.lambda_cov == 0
