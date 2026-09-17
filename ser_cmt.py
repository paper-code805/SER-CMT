"""SER-CMT backbone components.

SER-CMT keeps the CMT-Ti hierarchy and stage depths unchanged.  DALFE replaces
the local perception unit in Stages 1 and 2, while every Stage 3 block uses the
unified DALFE + ECSA + IRFFN design.  A semantic edge prior is generated once
from the completed Stage 2 feature and shared by all ten Stage 3 blocks.  ECSA
keeps Q on the unmodified feature and conditions K/V through a zero-initialized
residual path; the original CMT reduction and relative-position formulation
are preserved.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

import cmt


STAGE_DIMS = (46, 92, 184, 368)


def deform_conv2d_flop_jit(inputs, outputs):
    """fvcore FLOP handle for the deformable sampling convolution."""
    from fvcore.nn.jit_handles import conv_flop_count, get_shape

    return Counter(
        {
            "deform_conv": conv_flop_count(
                get_shape(inputs[0]),
                get_shape(inputs[1]),
                get_shape(outputs[0]),
                transposed=False,
            )
        }
    )


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch, channels, height, width = x.shape
    if channels % groups:
        raise ValueError(f"{channels} channels cannot be shuffled into {groups} groups")
    x = x.reshape(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).contiguous().reshape(batch, channels, height, width)


class AdaptiveGeometryBranch(nn.Module):
    """Depth-wise modulated deformable branch used by the validated DALFE."""

    def __init__(self, channels: int):
        super().__init__()
        self.offset_mask = nn.Conv2d(channels, 27, kernel_size=3, padding=1, bias=True)
        self.deform = DeformConv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.norm = nn.BatchNorm2d(channels, eps=1e-5)
        self.act = nn.GELU()
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x: Tensor) -> Tensor:
        offset_mask = self.offset_mask(x)
        offset, mask_logits = offset_mask[:, :18], offset_mask[:, 18:]
        return self.act(self.norm(self.deform(x, offset, torch.sigmoid(mask_logits))))


class DALFE(nn.Module):
    """Degradation-Adaptive Local Feature Enhancement (validated version)."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.input_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5),
            nn.GELU(),
        )
        self.texture = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels, eps=1e-5),
            nn.GELU(),
        )
        self.geometry = AdaptiveGeometryBranch(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5),
        )

    def forward(self, x: Tensor) -> Tensor:
        projected = self.input_projection(x)
        branches = (
            self.texture(projected),
            self.context(projected),
            self.geometry(projected),
        )
        fused = self.fuse(channel_shuffle(torch.cat(branches, dim=1), groups=3))
        return x + fused


class DALFEBlock(nn.Module):
    """CMT block with LPU replaced by DALFE and LMHSA/IRFFN unchanged."""

    def __init__(self, source_block: nn.Module, dim: int):
        super().__init__()
        self.norm1 = source_block.norm1
        self.attn = source_block.attn
        self.drop_path = source_block.drop_path
        self.norm2 = source_block.norm2
        self.mlp = source_block.mlp
        self.proj = DALFE(dim)

    def forward(self, x: Tensor, height: int, width: int, relative_pos: Tensor) -> Tensor:
        batch, _, channels = x.shape
        feature = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.proj(feature).flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), height, width, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


class SemanticEdgePrior(nn.Module):
    """Stage2 semantic projection followed by fixed Sobel/Laplacian edges."""

    def __init__(self, channels: int, use_valid_mask: bool = True, eps: float = 1e-6):
        super().__init__()
        self.semantic_proj = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.semantic_proj.weight, 1.0 / float(channels))
        nn.init.zeros_(self.semantic_proj.bias)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).reshape(1, 1, 3, 3)
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=True)
        self.register_buffer("sobel_y", sobel_y, persistent=True)
        self.register_buffer("laplacian", laplacian, persistent=True)
        self.edge_logits = nn.Parameter(torch.log(torch.tensor([0.7, 0.3])))
        self.use_valid_mask = bool(use_valid_mask)
        self.eps = float(eps)
        self.last_maps = {}

    def _normalize(self, x: Tensor) -> Tensor:
        return x / x.amax(dim=(-2, -1), keepdim=True).clamp_min(self.eps)

    def forward(
        self,
        feature2: Tensor,
        valid_mask: Optional[Tensor],
        target_size: Sequence[int],
    ) -> Tensor:
        p2 = torch.sigmoid(self.semantic_proj(feature2))
        gx = F.conv2d(p2, self.sobel_x, padding=1)
        gy = F.conv2d(p2, self.sobel_y, padding=1)
        sobel = self._normalize(torch.sqrt(gx.square() + gy.square() + self.eps))
        laplacian = self._normalize(F.conv2d(p2, self.laplacian, padding=1).abs())
        alpha_beta = torch.softmax(self.edge_logits, dim=0)
        edge_raw = alpha_beta[0] * sobel + alpha_beta[1] * laplacian
        edge2 = edge_raw * p2

        mask2 = None
        if self.use_valid_mask and valid_mask is not None:
            mask2 = F.interpolate(valid_mask, size=feature2.shape[-2:], mode="nearest")
            # One-cell erosion removes the artificial content/padding seam from
            # the edge support, not only the constant padding interior.
            mask2 = 1.0 - F.max_pool2d(1.0 - mask2, kernel_size=3, stride=1, padding=1)
            edge2 = edge2 * mask2

        pooled = F.max_pool2d(edge2, kernel_size=2, stride=2)
        edge3 = (
            pooled
            if tuple(pooled.shape[-2:]) == tuple(target_size)
            else F.interpolate(pooled, size=target_size, mode="bilinear", align_corners=False)
        )
        self.last_maps = {
            "p2": p2.detach(),
            "sobel": sobel.detach(),
            "laplacian": laplacian.detach(),
            "edge_raw": edge_raw.detach(),
            "edge2": edge2.detach(),
            "edge3": edge3.detach(),
            "valid2": mask2.detach() if mask2 is not None else None,
        }
        return edge3

    def fusion_weights(self) -> Tuple[float, float]:
        weights = torch.softmax(self.edge_logits.detach(), dim=0).cpu()
        return float(weights[0]), float(weights[1])


class ECSA(nn.Module):
    """Edge-Conditioned Structural Attention with original CMT Q/K/V path."""

    def __init__(self, dim: int, source_attention: nn.Module):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(source_attention.num_heads)
        self.scale = source_attention.scale
        self.qk_dim = int(source_attention.qk_dim)
        self.sr_ratio = int(source_attention.sr_ratio)

        # Names and shapes intentionally match LMHSA so a DALFE checkpoint can
        # initialize every original attention parameter without remapping.
        self.q = source_attention.q
        self.k = source_attention.k
        self.v = source_attention.v
        self.attn_drop = source_attention.attn_drop
        self.proj = source_attention.proj
        self.proj_drop = source_attention.proj_drop
        if self.sr_ratio > 1:
            self.sr = source_attention.sr

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        nn.init.zeros_(self.spatial_conv.bias)
        self.eta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_structural_attention = None

    def forward(
        self,
        x: Tensor,
        height: int,
        width: int,
        relative_pos: Tensor,
        edge_prior: Tensor,
    ) -> Tensor:
        batch, tokens, channels = x.shape
        feature = x.transpose(1, 2).reshape(batch, channels, height, width)
        avg_map = feature.mean(dim=1, keepdim=True)
        max_map = feature.amax(dim=1, keepdim=True)
        spatial_semantic = self.spatial_conv(torch.cat([avg_map, max_map], dim=1))
        if tuple(edge_prior.shape[-2:]) != (height, width):
            edge_prior = F.interpolate(
                edge_prior, size=(height, width), mode="bilinear", align_corners=False
            )
        structural = torch.sigmoid(spatial_semantic + self.eta * edge_prior)
        enhanced_feature = feature * (1.0 + self.gamma * structural)
        enhanced_tokens = enhanced_feature.flatten(2).transpose(1, 2)
        self.last_structural_attention = structural.detach()

        # Occluded positions retain their complete query representation.
        q = self.q(x).reshape(
            batch, tokens, self.num_heads, self.qk_dim // self.num_heads
        ).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            reduced = self.sr(enhanced_feature).flatten(2).transpose(1, 2)
        else:
            reduced = enhanced_tokens
        reduced_tokens = reduced.shape[1]
        k = self.k(reduced).reshape(
            batch, reduced_tokens, self.num_heads, self.qk_dim // self.num_heads
        ).permute(0, 2, 1, 3)
        v = self.v(reduced).reshape(
            batch, reduced_tokens, self.num_heads, channels // self.num_heads
        ).permute(0, 2, 1, 3)

        attention = ((q @ k.transpose(-2, -1)) * self.scale + relative_pos).softmax(dim=-1)
        attention = self.attn_drop(attention)
        output = (attention @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class SERBlock(nn.Module):
    """Semantic edge-guided reasoning block: DALFE + ECSA + IRFFN."""

    def __init__(self, source_block: nn.Module, dim: int):
        super().__init__()
        self.norm1 = source_block.norm1
        self.attn = ECSA(dim, source_block.attn)
        self.drop_path = source_block.drop_path
        self.norm2 = source_block.norm2
        self.mlp = source_block.mlp
        self.proj = DALFE(dim)

    def forward(
        self,
        x: Tensor,
        height: int,
        width: int,
        relative_pos: Tensor,
        edge_prior: Tensor,
    ) -> Tensor:
        batch, _, channels = x.shape
        feature = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.proj(feature).flatten(2).transpose(1, 2)
        x = x + self.drop_path(
            self.attn(self.norm1(x), height, width, relative_pos, edge_prior)
        )
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        return x


def build_ser_cmt(
    img_size: int = 320,
    drop_path_rate: float = 0.1,
    use_valid_mask: bool = True,
) -> nn.Module:
    """Build Full SER-CMT on the established CMT-Ti detector backbone."""
    model = cmt.cmt_ti(
        pretrained=False,
        img_size=img_size,
        num_classes=0,
        drop_path_rate=drop_path_rate,
    )
    model.blocks_a = nn.ModuleList(DALFEBlock(block, STAGE_DIMS[0]) for block in model.blocks_a)
    model.blocks_b = nn.ModuleList(DALFEBlock(block, STAGE_DIMS[1]) for block in model.blocks_b)
    model.semantic_edge_prior = SemanticEdgePrior(
        STAGE_DIMS[1], use_valid_mask=use_valid_mask
    )

    model.blocks_c = nn.ModuleList(
        SERBlock(block, STAGE_DIMS[2]) for block in model.blocks_c
    )
    model.ser_variant = "full_ser_cmt"
    return model


class SERCMTFeatures(nn.Module):
    """Four-stage feature wrapper that propagates the shared E3 prior."""

    def __init__(self, model: nn.Module, infer_valid_mask: bool = True):
        super().__init__()
        self.model = model
        self.infer_valid_mask = bool(infer_valid_mask)
        self.last_valid_mask = None

    @staticmethod
    def _valid_region_mask(x: Tensor, tolerance: float = 1e-7) -> Tensor:
        # Deterministic letterbox padding consists of exactly constant full
        # rows/columns, even after color jitter and RetinaNet normalization.
        row_span = x.amax(dim=(1, 3)) - x.amin(dim=(1, 3))
        col_span = x.amax(dim=(1, 2)) - x.amin(dim=(1, 2))
        invalid_rows = row_span <= tolerance
        invalid_cols = col_span <= tolerance
        valid = (~invalid_rows).unsqueeze(1).unsqueeze(3) & (~invalid_cols).unsqueeze(1).unsqueeze(2)
        return valid.to(dtype=x.dtype)

    @staticmethod
    def _stage(
        x: Tensor,
        patch: nn.Module,
        blocks: Iterable[nn.Module],
        relative_pos: Tensor,
    ) -> Tensor:
        batch = x.shape[0]
        tokens, (height, width) = patch(x)
        for block in blocks:
            tokens = block(tokens, height, width, relative_pos)
        return tokens.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: Tensor) -> List[Tensor]:
        m = self.model
        valid = self._valid_region_mask(x) if self.infer_valid_mask else None
        self.last_valid_mask = valid.detach() if valid is not None else None
        x = m.stem_norm1(m.stem_relu1(m.stem_conv1(x)))
        x = m.stem_norm2(m.stem_relu2(m.stem_conv2(x)))
        x = m.stem_norm3(m.stem_relu3(m.stem_conv3(x)))
        stage1 = self._stage(x, m.patch_embed_a, m.blocks_a, m.relative_pos_a)
        stage2 = self._stage(stage1, m.patch_embed_b, m.blocks_b, m.relative_pos_b)

        batch = stage2.shape[0]
        tokens3, (height3, width3) = m.patch_embed_c(stage2)
        edge3 = m.semantic_edge_prior(stage2, valid, (height3, width3))
        for block in m.blocks_c:
            tokens3 = block(tokens3, height3, width3, m.relative_pos_c, edge3)
        stage3 = tokens3.reshape(batch, height3, width3, -1).permute(0, 3, 1, 2).contiguous()
        stage4 = self._stage(stage3, m.patch_embed_d, m.blocks_d, m.relative_pos_d)
        return [stage1, stage2, stage3, stage4]


def architecture_manifest(model: nn.Module) -> dict:
    """Auditable placement and initialization manifest."""
    stages = {}
    for stage_index, name in enumerate(("blocks_a", "blocks_b", "blocks_c", "blocks_d"), 1):
        rows = []
        for index, block in enumerate(getattr(model, name)):
            rows.append(
                {
                    "block": index,
                    "local": "DALFE" if isinstance(block.proj, DALFE) else "LPU",
                    "attention": "ECSA" if isinstance(block.attn, ECSA) else "LMHSA",
                    "irffn": type(block.mlp).__name__,
                }
            )
        stages[f"stage{stage_index}"] = rows
    alpha, beta = model.semantic_edge_prior.fusion_weights()
    return {
        "model": "SER-CMT",
        "base": "CMT-Ti",
        "stages": stages,
        "stage3_depth": len(model.blocks_c),
        "stage3_ser_blocks": sum(isinstance(block, SERBlock) for block in model.blocks_c),
        "semantic_edge_prior": {
            "source": "post-Stage2 DALFE-enhanced feature",
            "projection": "1x1 Conv + Sigmoid",
            "operators": ["fixed Sobel", "fixed Laplacian"],
            "semantic_gate": "Eraw * P2",
            "valid_region_mask": bool(model.semantic_edge_prior.use_valid_mask),
            "downsample": "MaxPool2d, then bilinear fallback only on shape mismatch",
            "alpha_initial": alpha,
            "beta_initial": beta,
        },
        "ecsa": {
            "map": "sigmoid(S3 + eta * E3)",
            "enhancement": "X_hat = X3 * (1 + gamma * Mec)",
            "q_source": "X3",
            "kv_source": "X_hat",
            "eta_initial": 0.0,
            "gamma_initial": 0.0,
            "hard_gating": False,
            "attention_logit_bias": False,
        },
    }


def ecsa_modules(model: nn.Module) -> List[Tuple[str, ECSA]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, ECSA)]
