"""I-JEPA latent prediction: clean EMA targets and Smooth-L1 (paper Eq. 4).

The predictor attends to visible context with its original patch positions and
learnable target queries with their own positions. No target-image pixels enter
the context encoder or predictor. update_ema() is called after an optimizer step.
"""
from __future__ import annotations

import copy
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .vit_mini import TransformerBlock, gather_tokens, validate_indices


class JEPAPredictor(nn.Module):
    def __init__(self, encoder_dim, num_patches, predictor_dim=64, depth=2, num_heads=4):
        super().__init__()
        self.input_projection = nn.Linear(encoder_dim, predictor_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.blocks = nn.ModuleList(TransformerBlock(predictor_dim, num_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_projection = nn.Linear(predictor_dim, encoder_dim)
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, context_tokens, context_indices, target_indices):
        positions = self.position_embeddings.expand(context_tokens.shape[0], -1, -1)
        context = self.input_projection(context_tokens) + gather_tokens(positions, context_indices)
        queries = self.mask_token.expand(context.shape[0], target_indices.shape[1], -1)
        queries = queries + gather_tokens(positions, target_indices)
        tokens = torch.cat([context, queries], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_projection(self.norm(tokens[:, context.shape[1]:]))


class IJEPA(nn.Module):
    def __init__(self, encoder, predictor_dim=64, predictor_depth=2, num_heads=4):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder).requires_grad_(False).eval()
        self.predictor = JEPAPredictor(encoder.embed_dim, encoder.num_patches,
                                       predictor_dim, predictor_depth, num_heads)

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_ema(self, momentum: float = 0.996):
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be in [0, 1]")
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.lerp_(context.detach(), 1 - momentum)
        for target, context in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            if target.is_floating_point():
                target.lerp_(context.detach(), 1 - momentum)
            else:
                target.copy_(context)

    def forward(self, x: Tensor, context_indices: Tensor, target_indices,
                context_images: Tensor | None = None) -> dict:
        context_indices = validate_indices(context_indices, x.shape[0], self.context_encoder.num_patches,
                                           device=x.device)
        multiple = isinstance(target_indices, (list, tuple))
        blocks = list(target_indices) if multiple else [target_indices]
        if not blocks:
            raise ValueError("At least one target block is required")
        blocks = [validate_indices(t, x.shape[0], self.context_encoder.num_patches, device=x.device)
                  for t in blocks]
        for target in blocks:
            if bool((context_indices[:, :, None] == target[:, None, :]).any()):
                raise ValueError("Context and target patches must not overlap")
        if context_images is not None and context_images.shape != x.shape:
            raise ValueError("Clean target and noisy context images must have matching shapes")
        context = self.context_encoder.forward_tokens(x if context_images is None else context_images,
                                                       context_indices)
        with torch.no_grad():
            full_targets = self.target_encoder.forward_tokens(x)
            full_targets = F.layer_norm(full_targets, (full_targets.shape[-1],))
            targets = [gather_tokens(full_targets, block).detach() for block in blocks]
        predictions = [self.predictor(context, context_indices, block) for block in blocks]
        loss = torch.stack([F.smooth_l1_loss(p, t) for p, t in zip(predictions, targets)]).mean()
        return {"loss": loss, "predictions": predictions if multiple else predictions[0],
                "targets": targets if multiple else targets[0], "context_tokens": context}
