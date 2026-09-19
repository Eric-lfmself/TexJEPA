"""Small, offline ViT used to validate the paper's Sec. III--IV mechanics.

The defaults are a CPU surrogate, not the paper's pretrained ViT-Huge/14.
Patch tokens have learned absolute positions. Optional register tokens participate
in every attention block but are excluded from returned tokens and mean pooling.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def validate_indices(indices: Tensor, batch_size: int, num_patches: int, *, device=None) -> Tensor:
    if not isinstance(indices, Tensor) or indices.ndim != 2 or indices.shape[0] != batch_size:
        raise ValueError("Patch indices must be a tensor with shape (batch, selected_patches)")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Patch indices must have integer dtype")
    if indices.shape[1] == 0 or bool((indices < 0).any()) or bool((indices >= num_patches).any()):
        raise ValueError("Patch indices must be nonempty and within the image patch grid")
    if indices.shape[1] > 1 and bool((indices.sort(dim=1).values.diff(dim=1) == 0).any()):
        raise ValueError("Repeated patch indices are not allowed")
    return indices.to(device=device, dtype=torch.long)


def gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
    indices = validate_indices(indices, tokens.shape[0], tokens.shape[1], device=tokens.device)
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(embed_dim * mlp_ratio), embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class MiniViT(nn.Module):
    def __init__(self, image_size=32, patch_size=8, in_channels=3, embed_dim=64,
                 depth=2, num_heads=4, num_register_tokens=0):
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if depth < 1 or num_heads < 1 or embed_dim < 1 or embed_dim % num_heads:
            raise ValueError("depth must be positive and embed_dim divisible by num_heads")
        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be nonnegative")
        self.image_size, self.patch_size = image_size, patch_size
        self.in_channels, self.embed_dim = in_channels, embed_dim
        self.num_register_tokens = num_register_tokens
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        if num_register_tokens:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            nn.init.normal_(self.register_tokens, std=0.02)
        else:
            self.register_parameter("register_tokens", None)
        self.blocks = nn.ModuleList(TransformerBlock(embed_dim, num_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward_tokens(self, x: Tensor, indices: Tensor | None = None) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(f"Expected (B,{self.in_channels},{self.image_size},{self.image_size}) images")
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        if indices is not None:
            tokens = gather_tokens(tokens, indices)
        if self.register_tokens is not None:
            tokens = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)[:, self.num_register_tokens:]

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_tokens(x).mean(dim=1)

    def architecture_metadata(self) -> dict:
        return {"architecture": "MiniViT", "image_size": self.image_size,
                "patch_size": self.patch_size, "embed_dim": self.embed_dim,
                "depth": len(self.blocks), "num_register_tokens": self.num_register_tokens,
                "is_paper_pretrained_model": False}
