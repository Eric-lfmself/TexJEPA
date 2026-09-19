"""MAE pixel reconstruction counterpart to latent I-JEPA (paper Sec. II/III).

Only visible patches enter the encoder. Decoder predictions cover all patches;
the loss averages masked patches only. Pixel normalization is configurable.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from .vit_mini import TransformerBlock, validate_indices


class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim=64, decoder_depth=2, num_heads=4, norm_pix_loss=True):
        super().__init__()
        self.encoder = encoder
        self.norm_pix_loss = norm_pix_loss
        self.decoder_embed = nn.Linear(encoder.embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, encoder.num_patches, decoder_dim))
        self.decoder_blocks = nn.ModuleList(TransformerBlock(decoder_dim, num_heads)
                                            for _ in range(decoder_depth))
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, encoder.patch_size ** 2 * encoder.in_channels)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def patchify(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        p = self.encoder.patch_size
        if h != w or h % p:
            raise ValueError("Images must be square with dimensions divisible by patch_size")
        return x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p*p*c)

    def unpatchify(self, patches: Tensor) -> Tensor:
        b, n, _ = patches.shape
        g, p, c = self.encoder.grid_size, self.encoder.patch_size, self.encoder.in_channels
        if n != g*g or patches.shape[-1] != p*p*c:
            raise ValueError("Invalid patch dimensions")
        return patches.reshape(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, g*p, g*p)

    def forward(self, x: Tensor, visible_indices: Tensor | None = None, mask_ratio=0.75,
                generator: torch.Generator | None = None) -> dict:
        b, n = x.shape[0], self.encoder.num_patches
        if visible_indices is None:
            if not 0 < mask_ratio < 1:
                raise ValueError("mask_ratio must be strictly between zero and one")
            keep = max(1, min(n - 1, int(n * (1 - mask_ratio))))
            visible_indices = torch.rand(b, n, device=x.device, generator=generator).argsort(dim=1)[:, :keep]
        visible_indices = validate_indices(visible_indices, b, n, device=x.device)
        if visible_indices.shape[1] >= n:
            raise ValueError("MAE requires at least one masked patch")
        visible = self.encoder.forward_tokens(x, visible_indices)
        projected = self.decoder_embed(visible)
        tokens = self.mask_token.expand(b, n, -1).clone()
        tokens.scatter_(1, visible_indices[:, :, None].expand(-1, -1, projected.shape[-1]), projected)
        tokens = tokens + self.decoder_pos_embed
        for block in self.decoder_blocks:
            tokens = block(tokens)
        predictions = self.decoder_pred(self.decoder_norm(tokens))
        targets = self.patchify(x)
        if self.norm_pix_loss:
            targets = (targets - targets.mean(dim=-1, keepdim=True)) / (
                targets.var(dim=-1, unbiased=False, keepdim=True) + 1e-6).sqrt()
        mask = torch.ones(b, n, device=x.device, dtype=torch.bool)
        mask.scatter_(1, visible_indices, False)
        per_patch = (predictions - targets).square().mean(dim=-1)
        return {"loss": per_patch[mask].mean(), "predictions": predictions, "targets": targets,
                "mask": mask, "visible_indices": visible_indices, "context_tokens": visible}
