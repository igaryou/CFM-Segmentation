from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    ImageEncoderG,
    SegFormerSourceGenerator,
    SinusoidalTimeEmbedding,
    group_norm,
)


class SegFormerEndpointHead(nn.Module):
    """
    SegFormer endpoint predictor over fused RRDB image/state features.
    """

    VARIANTS = ("b1", "b2", "b3", "b4", "b5")

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        variant: str = "b5",
        decoder_channels: int = 256,
        time_emb_dim: int = 512,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown endpoint SegFormer variant: {variant}")

        try:
            from transformers import SegformerConfig, SegformerModel
        except ImportError as exc:
            raise ImportError(
                "SegFormerEndpointHead requires the 'transformers' package. "
                "Install it or use --backbone unet."
            ) from exc

        self.num_classes = num_classes
        self.variant = variant

        config = SegformerConfig(
            num_channels=in_channels,
            num_encoder_blocks=4,
            depths=SegFormerSourceGenerator.DEPTHS[variant],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=SegFormerSourceGenerator.HIDDEN_SIZES[variant],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=SegFormerSourceGenerator.NUM_HEADS[variant],
            mlp_ratios=[4, 4, 4, 4],
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            drop_path_rate=drop_path_rate,
            hidden_act="gelu",
            initializer_range=0.02,
            layer_norm_eps=1e-6,
        )
        self.encoder = SegformerModel(config)

        hidden_sizes = list(self.encoder.config.hidden_sizes)
        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(ch, decoder_channels, kernel_size=1) for ch in hidden_sizes]
        )

        decoder_in_channels = decoder_channels * len(hidden_sizes)
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in_channels, decoder_channels, kernel_size=3, padding=1),
            group_norm(decoder_channels),
            nn.SiLU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            group_norm(decoder_channels),
            nn.SiLU(),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

        self.time_embed_s = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_embed_delta = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.input_time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, in_channels),
        )
        self.decoder_time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, decoder_in_channels),
        )

    def forward(
        self,
        fused: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        if target_size is None:
            target_size = fused.shape[-2:]

        time_emb = self.time_embed_s(s) + self.time_embed_delta(t - s)
        fused = fused + self.input_time_proj(time_emb)[:, :, None, None]

        outputs = self.encoder(
            pixel_values=fused,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-len(self.proj_layers):]

        feats = []
        for feat, proj in zip(hidden_states, self.proj_layers):
            feat = proj(feat)
            feat = F.interpolate(
                feat,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            feats.append(feat)

        decoder_in = torch.cat(feats, dim=1)
        decoder_in = decoder_in + self.decoder_time_proj(time_emb)[:, :, None, None]
        logits = self.decoder(decoder_in)
        return logits


class SegDiffSegFormerModel(nn.Module):
    """
    CFM-compatible SegDiff model with a SegFormer endpoint predictor.
    """

    def __init__(
        self,
        image_channels: int = 3,
        mask_channels: int = 20,
        fusion_channels: int = 128,
        rrdb_blocks: int = 15,
        rrdb_growth_channels: int = 32,
        endpoint_segformer_variant: str = "b5",
        endpoint_decoder_channels: int = 256,
        endpoint_time_emb_dim: int = 512,
        endpoint_drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.image_encoder = ImageEncoderG(
            in_channels=image_channels,
            feat_channels=fusion_channels,
            out_channels=fusion_channels,
            num_rrdb_blocks=rrdb_blocks,
            growth_channels=rrdb_growth_channels,
        )
        self.null_image_feat = nn.Parameter(torch.zeros(1, fusion_channels, 1, 1))
        self.state_encoder = nn.Conv2d(
            mask_channels,
            fusion_channels,
            kernel_size=3,
            padding=1,
        )
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=3, padding=1),
            group_norm(fusion_channels),
            nn.SiLU(),
        )
        self.endpoint = SegFormerEndpointHead(
            in_channels=fusion_channels,
            num_classes=mask_channels,
            variant=endpoint_segformer_variant,
            decoder_channels=endpoint_decoder_channels,
            time_emb_dim=endpoint_time_emb_dim,
            drop_path_rate=endpoint_drop_path_rate,
        )

    def encode_image(self, img):
        return self.image_encoder(img)

    def get_null_image_feat(
        self,
        image_feat: torch.Tensor,
        null_condition: str = "learned",
    ) -> torch.Tensor:
        if null_condition == "zero":
            return torch.zeros_like(image_feat)

        if null_condition == "learned":
            null_feat = self.null_image_feat.to(
                device=image_feat.device,
                dtype=image_feat.dtype,
            )
            return null_feat.expand(
                image_feat.size(0),
                -1,
                image_feat.size(2),
                image_feat.size(3),
            )

        raise ValueError(f"Unknown null_condition: {null_condition}")

    def forward_logits_with_image_feat(self, x_s, image_feat, s, t):
        state_feat = self.state_encoder(x_s)

        if image_feat.shape[-2:] != x_s.shape[-2:]:
            image_feat = F.interpolate(
                image_feat,
                size=x_s.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = self.fuse_conv(torch.cat([image_feat, state_feat], dim=1))
        return self.endpoint(fused, s, t, target_size=x_s.shape[-2:])

    def forward_with_image_feat(self, x_s, image_feat, s, t):
        logits = self.forward_logits_with_image_feat(x_s, image_feat, s, t)
        pi = torch.softmax(logits, dim=1)
        return logits, pi

    def forward_logits(self, x_s, img, s, t):
        image_feat = self.encode_image(img)
        return self.forward_logits_with_image_feat(x_s, image_feat, s, t)

    def forward(self, x_s, img, s, t):
        logits = self.forward_logits(x_s, img, s, t)
        pi = torch.softmax(logits, dim=1)
        return logits, pi
