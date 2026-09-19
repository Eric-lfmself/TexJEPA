"""Spatial block masking for Sec. IV-E, with finite and reported tiny-grid fallback.

Context scale denotes the sampled context rectangle BEFORE removing every target
patch, as in I-JEPA block masking. Target blocks may overlap each other; no target
patch is visible to the context encoder. Target scale is an implementation
assumption because this paper does not report it. Batch context lengths are
equalized by uniformly subsampling only already-visible context patches.
"""
from __future__ import annotations

import math
import torch


class BlockMaskSampler:
    def __init__(self, grid_size: int, *, variant="baseline", context_scale=None,
                 target_scale=(0.15, 0.20), num_target_blocks=None,
                 target_aspect_ratio=None, max_attempts=32, seed=42):
        if grid_size < 2:
            raise ValueError("A grid of at least 2x2 is required for disjoint context and target patches")
        if variant not in ("baseline", "v3.1", "v4", "v5", "v6", "noise", "register", "vicreg"):
            raise ValueError(f"Unknown masking variant {variant!r}")
        registers = variant in ("v5", "register")
        self.grid_size = grid_size
        self.context_scale = tuple(context_scale or ((0.65, 0.85) if registers else (0.85, 1.0)))
        self.target_scale = tuple(target_scale)
        self.num_target_blocks = num_target_blocks if num_target_blocks is not None else (6 if registers else 4)
        self.target_aspect_ratio = tuple(target_aspect_ratio or ((0.5, 2.0) if registers else (0.75, 1.5)))
        self.max_attempts = max_attempts
        if self.num_target_blocks < 1 or max_attempts < 1:
            raise ValueError("num_target_blocks and max_attempts must be positive")
        for label, bounds in (("context_scale", self.context_scale), ("target_scale", self.target_scale)):
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1] <= 1:
                raise ValueError(f"{label} must be an increasing pair within (0,1]")
        if len(self.target_aspect_ratio) != 2 or not 0 < self.target_aspect_ratio[0] <= self.target_aspect_ratio[1]:
            raise ValueError("target_aspect_ratio must be an increasing positive pair")
        self.generator = torch.Generator().manual_seed(seed)
        self.last_metadata = None

    @staticmethod
    def _uniform(bounds, generator):
        return bounds[0] + float(torch.rand((), generator=generator)) * (bounds[1] - bounds[0])

    def _shape(self, scale, aspect, generator):
        """Return an integer rectangle satisfying area/aspect bounds when possible."""
        g = self.grid_size
        candidates = [(h, w) for h in range(1, g + 1) for w in range(1, g + 1)
                      if scale[0] <= h*w/(g*g) <= scale[1] and aspect[0] <= w/h <= aspect[1]]
        if candidates:
            wanted_area = self._uniform(scale, generator) * g*g
            wanted_ratio = self._uniform(aspect, generator)
            h, w = min(candidates, key=lambda hw: abs(hw[0]*hw[1] - wanted_area)/(g*g)
                       + abs(math.log((hw[1]/hw[0])/wanted_ratio)))
            return h, w, False
        # Coarse CPU grids can have no integer rectangle inside the continuous
        # paper ranges. Explicitly report the nearest feasible geometry.
        desired_area = self._uniform(scale, generator) * g*g
        desired_ratio = self._uniform(aspect, generator)
        all_shapes = [(h, w) for h in range(1, g + 1) for w in range(1, g + 1)
                      if h*w < g*g]
        h, w = min(all_shapes, key=lambda hw: abs(hw[0]*hw[1] - desired_area)/(g*g)
                   + abs(math.log((hw[1]/hw[0])/desired_ratio)))
        return h, w, True

    def _rectangle(self, shape, generator):
        h, w = shape
        top = int(torch.randint(self.grid_size - h + 1, (), generator=generator))
        left = int(torch.randint(self.grid_size - w + 1, (), generator=generator))
        yy, xx = torch.meshgrid(torch.arange(top, top+h), torch.arange(left, left+w), indexing="ij")
        return (yy*self.grid_size + xx).flatten(), [top, left, h, w]

    def sample(self, batch_size: int, *, device="cpu", generator=None) -> dict:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        generator = self.generator if generator is None else generator
        if str(generator.device) != "cpu":
            raise ValueError("Block geometry uses a CPU generator; returned indices can be moved to any device")
        n = self.grid_size ** 2
        fallbacks = []
        target_shapes = []
        for block in range(self.num_target_blocks):
            h, w, fallback = self._shape(self.target_scale, self.target_aspect_ratio, generator)
            target_shapes.append((h, w))
            if fallback:
                fallbacks.append({"kind": "integer_target_geometry", "block": block,
                                  "shape": [h, w], "actual_scale": h*w/n, "actual_aspect": w/h})
        # A context rectangle uses a square-ish aspect as an explicit assumption.
        ch, cw, context_fallback = self._shape(self.context_scale, (0.75, 1.5), generator)
        if context_fallback:
            fallbacks.append({"kind": "integer_context_geometry", "shape": [ch, cw], "actual_scale": ch*cw/n})
        contexts, all_targets, geometries = [], [], []
        for image in range(batch_size):
            accepted = False
            for attempt in range(self.max_attempts):
                target_rectangles = [self._rectangle(shape, generator) for shape in target_shapes]
                raw_context, context_box = self._rectangle((ch, cw), generator)
                target_union = torch.unique(torch.cat([pair[0] for pair in target_rectangles]))
                context = raw_context[~torch.isin(raw_context, target_union)]
                if context.numel():
                    accepted = True
                    break
            if not accepted:
                # Keep every target a true rectangle. Reposition them together
                # to leave the final grid cell for context; never loop forever.
                target_rectangles = []
                for h, w in target_shapes:
                    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                    target_rectangles.append(((yy*self.grid_size + xx).flatten(), [0, 0, h, w]))
                union = torch.unique(torch.cat([pair[0] for pair in target_rectangles]))
                context = torch.arange(n)[~torch.isin(torch.arange(n), union)]
                if not context.numel():
                    # Only possible for extreme caller-selected scales. Preserve
                    # a valid smoke-test objective and disclose changed targets.
                    target_rectangles = [(torch.tensor([0]), [0, 0, 1, 1]) for _ in target_shapes]
                    context = torch.arange(1, n)
                context_box = [0, 0, self.grid_size, self.grid_size]
                fallbacks.append({"kind": "finite_retry_exhausted", "image": image,
                                  "attempts": self.max_attempts,
                                  "detail": "co-located targets and residual context; original context scale relaxed"})
            contexts.append(context)
            all_targets.append([pair[0] for pair in target_rectangles])
            geometries.append({"context_rectangle": context_box,
                               "target_rectangles": [pair[1] for pair in target_rectangles],
                               "visible_before_equalization": context.numel()})
        # Extreme fallback may alter target sizes for one image. Apply its tiny
        # geometry consistently over the batch, keeping each block rectangular.
        if any(len({all_targets[b][i].numel() for b in range(batch_size)}) != 1
               for i in range(self.num_target_blocks)):
            all_targets = [[torch.tensor([0]) for _ in target_shapes] for _ in range(batch_size)]
            contexts = [torch.arange(1, n) for _ in range(batch_size)]
            geometries = [{"context_rectangle": [0, 0, self.grid_size, self.grid_size],
                           "target_rectangles": [[0, 0, 1, 1] for _ in target_shapes],
                           "visible_before_equalization": n-1} for _ in range(batch_size)]
            fallbacks.append({"kind": "batch_geometry_fallback", "detail": "single-patch targets for all images"})
        keep = min(context.numel() for context in contexts)
        context_indices = torch.stack([context[torch.randperm(context.numel(), generator=generator)[:keep]].sort().values
                                       for context in contexts]).to(device)
        target_indices = [torch.stack([all_targets[b][i] for b in range(batch_size)]).to(device)
                          for i in range(self.num_target_blocks)]
        metadata = {"context_scale_requested": list(self.context_scale),
                    "target_scale_requested": list(self.target_scale),
                    "target_scale_is_assumption": True,
                    "target_aspect_ratio_requested": list(self.target_aspect_ratio),
                    "num_target_blocks": self.num_target_blocks, "grid_size": self.grid_size,
                    "context_patches": keep, "visible_fraction": keep/n,
                    "target_blocks_can_overlap_each_other": True,
                    "context_target_overlap": False, "geometry": geometries,
                    "fallback_used": bool(fallbacks), "fallbacks": fallbacks}
        self.last_metadata = metadata
        return {"context_indices": context_indices, "target_indices": target_indices, "metadata": metadata}

    def __call__(self, batch_size, **kwargs):
        return self.sample(batch_size, **kwargs)


def sample_block_masks(grid_size, batch_size, *, variant="baseline", seed=42, device="cpu", **kwargs):
    return BlockMaskSampler(grid_size, variant=variant, seed=seed, **kwargs)(batch_size, device=device)
