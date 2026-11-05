"""
models.py
=============================================================================
Standardized Model Architecture Registry for CheXpert Plus SOTA.
=============================================================================
Features:
- TriBackboneFusion: Direct multi-scale fusion of DenseNet-121, ConvNeXt-Small,
  and EfficientNetV2-M with BatchNorm1d and 2-stage MLP projection head.
- Output bias initialization to empirical class log-odds, eliminating wasted
  epochs discovering dataset priors.
- Discriminative parameter grouping for separate backbone and head optimization.
- Support for single-backbone baselines (e.g., DenseNet-121).
"""

from __future__ import annotations

import os
from typing import Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
import timm

from config import ALL_LABELS


class TriBackboneFusion(nn.Module):
    """
    Tri-Backbone Direct Multi-Scale Fusion Architecture for CheXpert Plus.

    Feature extraction:
      - DenseNet-121     (ImageNet pretrained) -> Global-Average-Pool -> [B, 1024]
      - ConvNeXt-Small   (ImageNet pretrained) -> Global-Average-Pool -> [B,  768]
      - EfficientNetV2-M (ImageNet pretrained) -> Global-Average-Pool -> [B, 1280]
      Total concatenated representation: [B, 3072]

    Classification head:
      - BatchNorm1d(3072) standardizes features per-channel across the mini-batch,
        preventing any single backbone from overpowering the fused vector while
        strictly preserving inter-image L2 distance.
      - Dropout(dropout)
      - Linear(3072, head_hidden) -> GELU -> BatchNorm1d(head_hidden)
      - Dropout(dropout * 0.5)
      - Linear(head_hidden, num_classes) -> Output Logits (no sigmoid)
    """

    BACKBONE_DIMS: Dict[str, int] = {
        "densenet121": 1024,
        "convnext_small": 768,
        "tf_efficientnetv2_m": 1280,
    }
    TOTAL_DIM: int = sum(BACKBONE_DIMS.values())  # 3072

    def __init__(
        self,
        num_classes: int = len(ALL_LABELS),
        pretrained: bool = True,
        dropout: float = 0.3,
        head_hidden: int = 768,
        drop_path_rate: float = 0.1,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # -------------------------------------------------------------------
        # Pretrained CNN Backbones (num_classes=0 returns pooled feature vectors)
        # -------------------------------------------------------------------
        self.densenet = timm.create_model(
            "densenet121",
            pretrained=pretrained,
            num_classes=0,
        )
        self.convnext = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.efficientnet = timm.create_model(
            "tf_efficientnetv2_m",
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        if grad_checkpointing:
            for m in self._cnns():
                if hasattr(m, "set_grad_checkpointing"):
                    m.set_grad_checkpointing(True)

        # -------------------------------------------------------------------
        # Standardized Normalization & MLP Projection Head
        # -------------------------------------------------------------------
        self.fusion_bn = nn.BatchNorm1d(self.TOTAL_DIM)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.TOTAL_DIM, head_hidden),
            nn.GELU(),
            nn.BatchNorm1d(head_hidden),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, num_classes),
        )

        self._init_head()

    def _cnns(self) -> List[nn.Module]:
        return [self.densenet, self.convnext, self.efficientnet]

    def _init_head(self) -> None:
        """Initializes projection head layers with truncated normal distribution."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape [B, 3, H, W]
        Returns:
            logits: Output classification logits of shape [B, num_classes]
        """
        f1 = self.densenet(x)       # [B, 1024]
        f2 = self.convnext(x)       # [B,  768]
        f3 = self.efficientnet(x)   # [B, 1280]

        fused = torch.cat([f1, f2, f3], dim=1)  # [B, 3072]

        # Robust BatchNorm: if single-sample batch arrives during train mode,
        # fallback to running statistics to prevent PyTorch BatchNorm exception.
        if fused.shape[0] == 1 and self.training:
            fused = nn.functional.batch_norm(
                fused, self.fusion_bn.running_mean, self.fusion_bn.running_var,
                self.fusion_bn.weight, self.fusion_bn.bias, False, 0.0, self.fusion_bn.eps
            )
        else:
            fused = self.fusion_bn(fused)

        return self.head(fused)                  # [B, num_classes]

    def split_params(self) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
        """
        Splits model parameters into (backbone_params, head_params).
        Enables discriminative learning rate assignment in AdamW optimizer.
        """
        cnn_ids = {id(p) for m in self._cnns() for p in m.parameters()}
        backbone_params, head_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in cnn_ids:
                backbone_params.append((n, p))
            else:
                head_params.append((n, p))
        return backbone_params, head_params

    @torch.no_grad()
    def set_output_bias(self, prior: torch.Tensor) -> None:
        """
        Initializes output linear layer bias to empirical log-odds of training labels:
            bias_c = ln(prior_c / (1 - prior_c))

        This critical initialization starts the network at the exact class distribution
        at step 0, directing all subsequent gradient updates toward image feature
        discrimination rather than wasting entire epochs discovering prevalence.
        """
        last_linear: Optional[nn.Linear] = None
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                last_linear = m

        if last_linear is None or last_linear.bias is None:
            return

        prior = prior.to(last_linear.bias.device).float().clamp(min=1e-4, max=1.0 - 1e-4)
        log_odds = torch.log(prior / (1.0 - prior))
        last_linear.bias.copy_(log_odds)


class SingleBackboneClassifier(nn.Module):
    """
    Standard single-backbone classifier (e.g., DenseNet-121) for baseline evaluation.
    """

    def __init__(
        self,
        model_name: str = "densenet121",
        num_classes: int = len(ALL_LABELS),
        pretrained: bool = True,
        dropout: float = 0.2,
        drop_path_rate: float = 0.0,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        kwargs = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
        # Stochastic depth for ConvNeXt / EfficientNet (DenseNet does not support it)
        if drop_path_rate > 0 and not model_name.lower().startswith("densenet"):
            kwargs["drop_path_rate"] = drop_path_rate
        self.backbone = timm.create_model(model_name, **kwargs)
        # Small classifier init so training starts at the class priors (bias set by set_output_bias).
        # timm's EfficientNet init uses uniform(+-1/sqrt(num_classes)) = +-0.38 for 7 classes, which makes
        # the first predictions confidently random (initial BCE 2.08 instead of ~0.6).
        classifier = self.backbone.get_classifier()
        if isinstance(classifier, nn.Linear):
            nn.init.trunc_normal_(classifier.weight, std=0.01)
            if classifier.bias is not None:
                nn.init.zeros_(classifier.bias)
        # Recompute activations in the backward pass: much less VRAM at high resolution, ~25-35% slower
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def split_params(self) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
        head_ids = {id(p) for p in self.backbone.get_classifier().parameters()}
        backbone_params, head_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_ids:
                head_params.append((n, p))
            else:
                backbone_params.append((n, p))
        return backbone_params, head_params

    @torch.no_grad()
    def set_output_bias(self, prior: torch.Tensor) -> None:
        classifier = self.backbone.get_classifier()
        if hasattr(classifier, "bias") and classifier.bias is not None:
            prior = prior.to(classifier.bias.device).float().clamp(min=1e-4, max=1.0 - 1e-4)
            log_odds = torch.log(prior / (1.0 - prior))
            classifier.bias.copy_(log_odds)


class RadDinoClassifier(nn.Module):
    """
    RAD-DINO transformer classifier.

    Encoder: RAD-DINO (Microsoft; Perez-Garcia et al., Nature Machine Intelligence 2025), a ViT-B/14
    trained with DINOv2 self-supervision on 882,775 chest X-rays (MIMIC-CXR, CheXpert, NIH, PadChest,
    BRAX; images only, no labels). Native input 518 px = 37x37 patches. MIT licence, research use.
    Weights: huggingface.co/microsoft/rad-dino (backbone_compatible.safetensors, loads into timm).

    Head: [CLS token ; mean of patch tokens] (1536) -> LayerNorm -> Dropout -> Linear(num_classes).

    freeze_blocks: 0 = full fine-tuning, 12 = frozen encoder (head only), k = first k blocks frozen.
    param_groups() applies layer-wise learning-rate decay: block i gets lr * layer_decay^(13 - i).
    """

    HF_REPO = "microsoft/rad-dino"
    HF_FILE = "backbone_compatible.safetensors"
    MEAN = [0.5307, 0.5307, 0.5307]
    STD = [0.2583, 0.2583, 0.2583]

    def __init__(
        self,
        num_classes: int = len(ALL_LABELS),
        pretrained: bool = True,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        image_size: int = 518,
        freeze_blocks: int = 0,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2", pretrained=False, num_classes=0,
            img_size=image_size, drop_path_rate=drop_path_rate,
        )
        if pretrained:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download(self.HF_REPO, self.HF_FILE))
            sd.pop("mask_token", None)
            if sd["pos_embed"].shape != self.backbone.pos_embed.shape:
                from timm.layers import resample_abs_pos_embed
                sd["pos_embed"] = resample_abs_pos_embed(
                    sd["pos_embed"], new_size=list(self.backbone.patch_embed.grid_size),
                    num_prefix_tokens=self.backbone.num_prefix_tokens,
                )
            self.backbone.load_state_dict(sd, strict=True)

        dim = self.backbone.num_features
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Dropout(dropout), nn.Linear(2 * dim, num_classes))
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)

        self.freeze_blocks = max(0, min(int(freeze_blocks), len(self.backbone.blocks)))
        if self.freeze_blocks:
            frozen = [self.backbone.patch_embed] + list(self.backbone.blocks[: self.freeze_blocks])
            if self.freeze_blocks == len(self.backbone.blocks):
                frozen.append(self.backbone.norm)
            for module in frozen:
                for p in module.parameters():
                    p.requires_grad = False
            self.backbone.cls_token.requires_grad = False
            self.backbone.pos_embed.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        for block in self.backbone.blocks[: self.freeze_blocks]:
            block.eval()  # frozen blocks: no stochastic depth noise
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)               # [B, 1 + 37*37, 768] (final norm applied)
        n_prefix = self.backbone.num_prefix_tokens
        feat = torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(dim=1)], dim=1)
        return self.head(feat)

    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float,
                     layer_decay: float = 0.75) -> List[Dict[str, Any]]:
        """AdamW groups with layer-wise LR decay; biases, norms, LayerScale, cls/pos tokens get no weight decay."""
        top = len(self.backbone.blocks) + 1                      # 0 = embeddings, 1..12 = blocks, 13 = final norm

        def layer_id(name: str) -> int:
            if name.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed")):
                return 0
            if name.startswith("blocks."):
                return int(name.split(".")[1]) + 1
            return top

        groups: Dict[Tuple[int, bool], Dict[str, Any]] = {}
        for name, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            lid = layer_id(name)
            no_wd = p.ndim <= 1 or name in ("cls_token", "pos_embed", "reg_token")
            g = groups.setdefault((lid, no_wd), {
                "params": [], "lr": backbone_lr * layer_decay ** (top - lid),
                "weight_decay": 0.0 if no_wd else weight_decay,
                "name": f"layer{lid:02d}_{'no_decay' if no_wd else 'decay'}",
            })
            g["params"].append(p)
        out = [groups[k] for k in sorted(groups)]
        out.append({"params": [p for p in self.head.parameters() if p.ndim > 1], "lr": head_lr,
                    "weight_decay": weight_decay, "name": "head_decay"})
        out.append({"params": [p for p in self.head.parameters() if p.ndim <= 1], "lr": head_lr,
                    "weight_decay": 0.0, "name": "head_no_decay"})
        return out

    @torch.no_grad()
    def set_output_bias(self, prior: torch.Tensor) -> None:
        last = self.head[-1]
        prior = prior.to(last.bias.device).float().clamp(min=1e-4, max=1.0 - 1e-4)
        last.bias.copy_(torch.log(prior / (1.0 - prior)))


RAD_DINO_ALIASES = {"rad_dino", "rad-dino", "raddino"}


def model_normalization(model_name: str) -> Tuple[List[float], List[float]]:
    """Input normalization (mean, std) expected by the model's pretrained weights."""
    if model_name.strip().lower() in RAD_DINO_ALIASES:
        return RadDinoClassifier.MEAN, RadDinoClassifier.STD
    from config import IMAGENET_MEAN, IMAGENET_STD
    return IMAGENET_MEAN, IMAGENET_STD


# Backwards compatibility alias
TriNNTransformerSOTA = TriBackboneFusion

TRI_MODEL_ALIASES = {
    "tri_nn_transformer",
    "hybrid_3nn_transformer",
    "3nn_transformer",
    "tri_backbone_fusion",
    "tri_fusion",
}


def build_model(
    model_name: str = "tri_nn_transformer",
    num_classes: int = len(ALL_LABELS),
    pretrained: bool = True,
    dropout: float = 0.3,
    head_hidden: int = 768,
    drop_path_rate: float = 0.1,
    grad_checkpointing: bool = False,
    image_size: Optional[int] = None,
    freeze_blocks: int = 0,
    **kwargs,
) -> nn.Module:
    """
    Factory function creating model architecture instances.
    """
    name_clean = model_name.strip().lower()
    if name_clean in RAD_DINO_ALIASES:
        return RadDinoClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            image_size=image_size or 518,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
        )
    if name_clean in TRI_MODEL_ALIASES:
        return TriBackboneFusion(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            head_hidden=head_hidden,
            drop_path_rate=drop_path_rate,
            grad_checkpointing=grad_checkpointing,
        )
    return SingleBackboneClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        grad_checkpointing=grad_checkpointing,
    )
