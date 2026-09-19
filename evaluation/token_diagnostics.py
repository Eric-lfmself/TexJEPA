"""Token diagnostics, TexJEPA Sec. V-H and Table XIII.

Patch-to-patch cosine drift extends Eq. 2 spatially. Gradient-times-activation
is a WEAK spatial diagnostic, not causal evidence of lesion use. The paper
does not specify saliency pooling/normalization; we explicitly use the signed
sum of gradient*activation across features, with no display normalization.
The default head input is mean patch pooling, matching this project's MiniViT.
Supply the actual pooling function when evaluating another backbone/readout.
"""

from contextlib import contextmanager

import torch

from metrics import cosine_drift
from .robustness import evaluation_mode


def _grid(grid_size, token_count):
    if isinstance(grid_size, int):
        height, width = grid_size, grid_size
    elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
        height, width = grid_size
    else:
        raise ValueError("Specify an integer or (height,width) patch grid")
    if not isinstance(height, int) or not isinstance(width, int) or min(height, width) < 1:
        raise ValueError("Patch grid dimensions must be positive integers")
    if height * width != token_count:
        raise ValueError("Grid must match patch tokens exactly; remove CLS/register tokens before analysis")
    return height, width


def token_cosine_drift(clean_tokens, perturbed_tokens, *, grid_size):
    """Return [B,grid_h,grid_w], with undefined zero-token angles as NaN."""
    if clean_tokens.ndim != 3 or clean_tokens.shape != perturbed_tokens.shape:
        raise ValueError("Expected matching BxNxD patch-token tensors")
    batch, count, dimension = clean_tokens.shape
    height, width = _grid(grid_size, count)
    return cosine_drift(clean_tokens.reshape(-1, dimension),
                        perturbed_tokens.reshape(-1, dimension)).reshape(batch, height, width)


def compare_token_drift(encoder, clean_images, perturbed_images, *, grid_size=None):
    """Compare spatially corresponding full patch tokens without weight updates.

    ``forward_tokens`` must return patch tokens in raster order, excluding all
    non-image tokens. Pair images must have identical shape and alignment.
    """
    if clean_images.ndim != 4 or clean_images.shape != perturbed_images.shape:
        raise ValueError("Token drift requires matching BCHW image batches")
    if not hasattr(encoder, "forward_tokens"):
        raise ValueError("Encoder must expose forward_tokens for patch diagnostics")
    grid = getattr(encoder, "grid_size", None) if grid_size is None else grid_size
    with evaluation_mode(encoder):
        clean = encoder.forward_tokens(clean_images)
        perturbed = encoder.forward_tokens(perturbed_images)
        values = token_cosine_drift(clean, perturbed, grid_size=grid)
    finite = torch.isfinite(values)
    return {"drift_map": values.detach(), "valid_tokens_per_image": finite.sum(dim=(1, 2)),
            "mean_drift": values[finite].mean().item() if finite.any() else float("nan"),
            "total_tokens": values.numel(), "valid_tokens": int(finite.sum())}


@contextmanager
def _frozen_differentiable_eval(*models):
    """Freeze parameter gradients without disabling activation/input gradients."""
    parameters = {parameter: parameter.requires_grad for model in models for parameter in model.parameters()}
    modules = {module: module.training for model in models for module in model.modules()}
    try:
        for model in models:
            model.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        with torch.enable_grad():
            yield
    finally:
        for parameter, original in parameters.items():
            parameter.requires_grad_(original)
        for module, training in modules.items():
            module.training = training


def gradient_activation_saliency(encoder, head, images, class_index, *,
                                 grid_size=None, pooling=None):
    """Return signed patch saliency and input gradients for target class logits.

    ``class_index`` can be one class for the batch, or a [B] index tensor.
    autograd.grad leaves existing parameter .grad buffers untouched. All
    parameter trainability flags and train/eval modes are restored afterwards.
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("Expected floating BCHW images")
    grid = getattr(encoder, "grid_size", None) if grid_size is None else grid_size
    pool = (lambda tokens: tokens.mean(dim=1)) if pooling is None else pooling
    with _frozen_differentiable_eval(encoder, head):
        inputs = images.detach().clone().requires_grad_(True)
        tokens = encoder.forward_tokens(inputs)
        if tokens.ndim != 3 or not tokens.requires_grad:
            raise ValueError("forward_tokens must retain differentiability from the input")
        height, width = _grid(grid, tokens.shape[1])
        logits = head(pool(tokens))
        if logits.ndim != 2 or logits.shape[0] != images.shape[0] or not torch.isfinite(logits).all():
            raise ValueError("Head must return finite BxC logits")
        if isinstance(class_index, int):
            indices = torch.full((images.shape[0],), class_index, dtype=torch.long, device=images.device)
        else:
            indices = torch.as_tensor(class_index, device=images.device)
            if indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("Target classes must be integer indices")
            indices = indices.long()
        if indices.shape != (images.shape[0],) or (indices < 0).any() or (indices >= logits.shape[1]).any():
            raise ValueError("Each image requires a valid target-class index")
        selected = logits.gather(1, indices[:, None]).sum()
        token_gradients, input_gradients = torch.autograd.grad(selected, (tokens, inputs))
        attribution = (tokens * token_gradients).sum(dim=-1).reshape(-1, height, width)
    return {"saliency_map": attribution.detach(), "input_gradients": input_gradients.detach(),
            "class_indices": indices.detach(), "target_logits": logits.detach().gather(1, indices[:, None]).squeeze(1),
            "interpretation": "weak_spatial_diagnostic_not_causal_evidence"}
