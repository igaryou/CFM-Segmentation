import math
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# time embedding
# =========================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B]  (float or int)
        return: [B, dim]
        """
        device = t.device
        half = self.dim // 2
        emb_scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =========================================================
# RRDB blocks for image encoder G
# =========================================================
class DenseResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        growth_channels: int = 32,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.act = nn.LeakyReLU(0.2, inplace=False)

        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.residual_scale * x5


class RRDB(nn.Module):
    def __init__(
        self,
        channels: int,
        growth_channels: int = 32,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.db1 = DenseResidualBlock(channels, growth_channels, residual_scale)
        self.db2 = DenseResidualBlock(channels, growth_channels, residual_scale)
        self.db3 = DenseResidualBlock(channels, growth_channels, residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + self.residual_scale * out


class ImageEncoderG(nn.Module):
    """
    SegDiff の画像エンコーダ G を意識した RRDB ベース実装
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 64,
        out_channels: int = 64,
        num_rrdb_blocks: int = 6,
        growth_channels: int = 32,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, feat_channels, 3, padding=1)

        self.rrdb_body = nn.Sequential(
            *[RRDB(feat_channels, growth_channels=growth_channels) for _ in range(num_rrdb_blocks)]
        )
        self.body_conv = nn.Conv2d(feat_channels, feat_channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.conv_out = nn.Conv2d(feat_channels, out_channels, 3, padding=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(img)
        trunk = self.rrdb_body(feat)
        trunk = self.body_conv(trunk)
        feat = feat + trunk
        feat = self.act(feat)
        out = self.conv_out(feat)
        return out


# =========================================================
# mask encoder F
# =========================================================
class MaskEncoderF(nn.Module):
    """
    SegDiff の F は浅い 2D Conv として実装
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        return self.conv(x_t)


# =========================================================
# basic UNet blocks
# =========================================================
def group_norm(num_channels: int) -> nn.GroupNorm:
    # 小さいチャネル数でも動きやすいように調整
    groups = min(32, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = group_norm(out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x

        x = self.norm(x).view(b, c, h * w)       # [B, C, HW]
        qkv = self.qkv(x)                        # [B, 3C, HW]
        q, k, v = torch.chunk(qkv, 3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        scale = head_dim ** -0.5
        attn = torch.einsum("bncd,bnce->bnde", q * scale, k)   # [B, heads, HW, HW]
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bnde,bnce->bncd", attn, v)         # [B, heads, dim, HW]
        out = out.reshape(b, c, h * w)
        out = self.proj_out(out).view(b, c, h, w)

        return x_in + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# =========================================================
# UNet trunk (E / D)
# =========================================================
class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        use_attention: bool = False,
        dropout: float = 0.0,
        num_heads: int = 4,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim, dropout)
        self.attn = AttentionBlock(out_channels, num_heads) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class SegDiffUNet(nn.Module):
    """
    E/D 相当の U-Net trunk
    入力: fused features = F(x_t) + G(I)
    出力: predicted endpoint / velocity

    Categorical Flow Maps (Roos et al., 2026) に従い，
    二つの時刻 (s, t) を入力として受け取る．
    s と Δ=t-s をそれぞれ独立に sinusoidal embedding + MLP で埋め込み，
    加算して一つの time embedding とする．
    s=t (Δ=0) のとき，標準的な VFM（瞬時速度場）を復元する．
    s≠t のとき，flow map / self-distillation 学習に対応する．
    """
    def __init__(
        self,
        in_channels: int = 64,          # fusion後のチャネル数
        out_channels: int = 1,          # 予測のチャネル数（maskと同じ）
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attn_levels: Iterable[int] = (2, 3),  # 例: 16x16, 8x8 相当を想定
        dropout: float = 0.0,
        num_heads: int = 4,
    ):
        super().__init__()
        # s の埋め込み
        self.time_embed_s = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Δ = t - s の埋め込み
        self.time_embed_delta = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # down path
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        self.skip_channels = []

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    UNetBlock(
                        ch, out_ch, time_emb_dim,
                        use_attention=(level in attn_levels),
                        dropout=dropout,
                        num_heads=num_heads,
                    )
                )
                ch = out_ch
                self.skip_channels.append(ch)
            self.down_blocks.append(blocks)

            if level != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # middle
        self.mid1 = UNetBlock(ch, ch, time_emb_dim, use_attention=True, dropout=dropout, num_heads=num_heads)
        self.mid2 = UNetBlock(ch, ch, time_emb_dim, use_attention=False, dropout=dropout, num_heads=num_heads)

        # up path
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        reversed_mults = list(reversed(channel_mults))
        skip_channels = list(reversed(self.skip_channels))

        for level, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop(0)
                blocks.append(
                    UNetBlock(
                        ch + skip_ch, out_ch, time_emb_dim,
                        use_attention=((len(channel_mults) - 1 - level) in attn_levels),
                        dropout=dropout,
                        num_heads=num_heads,
                    )
                )
                ch = out_ch
            self.up_blocks.append(blocks)

            if level != len(reversed_mults) - 1:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = group_norm(ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]  fused features
        s: [B]            始点時刻
        t: [B]            終点時刻

        s=t のとき VFM (瞬時速度場) に対応．
        s≠t のとき flow map (partial denoiser π_{s,t}) に対応．
        """
        delta = t - s  # [B]
        t_emb = self.time_embed_s(s) + self.time_embed_delta(delta)
        h = self.input_conv(x)

        skips = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, t_emb)
                skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for blocks, up in zip(self.up_blocks, self.upsamples):
            for block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            h = up(h)

        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


# =========================================================
# full SegDiff model
# =========================================================
class SegDiffModel(nn.Module):
    """
    Categorical Flow Maps 対応モデル:
        x_s --mask_encoder--> mask_feat
        I   --image_encoder--> image_feat
        fused = mask_feat + image_feat
        fused --UNet(E/D, s, t)--> π_{s,t}(x_s)  (partial denoiser / endpoint prediction)

    二つの時刻 (s, t) を入力として受け取る．
    s=t: VFM loss (瞬時速度場の学習) に使用
    s<t: Lagrangian self-distillation / ECLD loss に使用
    """
    def __init__(
        self,
        image_channels: int = 3,
        mask_channels: int = 20,
        fusion_channels: int = 64,
        rrdb_blocks: int = 6,
        rrdb_growth_channels: int = 32,
        rrdb_blocks_mask: int = 3,
        rrdb_growth_channels_mask: int = 16, 
        unet_base_channels: int = 64,
        unet_channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attn_levels: Iterable[int] = (2, 3),
        dropout: float = 0.0,
        num_heads: int = 4,
    ):
        super().__init__()

        self.mask_encoder = MaskEncoderF(in_channels=mask_channels, out_channels=fusion_channels)

        self.image_encoder = ImageEncoderG(
            in_channels=image_channels,
            feat_channels=fusion_channels,
            out_channels=fusion_channels,
            num_rrdb_blocks=rrdb_blocks,
            growth_channels=rrdb_growth_channels,
        )
        self.null_image_feat = nn.Parameter(torch.zeros(1, fusion_channels, 1, 1))

        self.unet = SegDiffUNet(
            in_channels=fusion_channels,
            out_channels=mask_channels,
            base_channels=unet_base_channels,
            channel_mults=unet_channel_mults,
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
            attn_levels=attn_levels,
            dropout=dropout,
            num_heads=num_heads,
        )
    def forward_logits(self, x_s, img, s, t):
        mask_feat = self.mask_encoder(x_s)
        image_feat = self.image_encoder(img)
        fused = mask_feat + image_feat
        logits = self.unet(fused, s, t)
        return logits
    
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
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


    def forward_logits_with_image_feat(
        self,
        x_s: torch.Tensor,
        image_feat: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        ) -> torch.Tensor:
        mask_feat = self.mask_encoder(x_s)
        fused = mask_feat + image_feat
        logits = self.unet(fused, s, t)
        return logits


    def forward_with_image_feat(
        self,
        x_s: torch.Tensor,
        image_feat: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ):
        logits = self.forward_logits_with_image_feat(x_s, image_feat, s, t)
        pi = torch.softmax(logits, dim=1)
        return logits, pi

    def forward(
        self,
        x_s: torch.Tensor,
        img: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_s: [B, mask_channels, H, W]   時刻 s における状態（ノイズ付きマスク）
        img: [B, image_channels, H, W]   条件画像
        s:   [B]                         始点時刻
        t:   [B]                         終点時刻

        Returns:
            π_{s,t}(x_s): [B, mask_channels, H, W]  partial denoiser の出力
        """
        logits = self.forward_logits(x_s, img, s, t)
        pi = torch.softmax(logits, dim=1)
        return logits, pi


class SegFormerSourceGenerator(nn.Module):
    """
    Image-conditioned initial distribution for K-channel CFM states.

    The output lives in the same K-channel space as the segv1 CFM state.  No
    softmax or simplex projection is applied to mu or x0.
    """

    MODEL_NAMES = {
        "b0": "nvidia/mit-b0",
        "b1": "nvidia/mit-b1",
        "b2": "nvidia/mit-b2",
        "b3": "nvidia/mit-b3",
        "b4": "nvidia/mit-b4",
        "b5": "nvidia/mit-b5",
    }
    DEPTHS = {
        "b0": [2, 2, 2, 2],
        "b1": [2, 2, 2, 2],
        "b2": [3, 4, 6, 3],
        "b3": [3, 4, 18, 3],
        "b4": [3, 8, 27, 3],
        "b5": [3, 6, 40, 3],
    }
    HIDDEN_SIZES = {
        "b0": [32, 64, 160, 256],
        "b1": [64, 128, 320, 512],
        "b2": [64, 128, 320, 512],
        "b3": [64, 128, 320, 512],
        "b4": [64, 128, 320, 512],
        "b5": [64, 128, 320, 512],
    }
    NUM_HEADS = {
        "b0": [1, 2, 5, 8],
        "b1": [1, 2, 5, 8],
        "b2": [1, 2, 5, 8],
        "b3": [1, 2, 5, 8],
        "b4": [1, 2, 5, 8],
        "b5": [1, 2, 5, 8],
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "b0",
        pretrained: bool = False,
        decoder_channels: int = 128,
        freeze_encoder: bool = False,
        learned_logvar: bool = False,
        fixed_std: Optional[float] = 1.0,
        mu_tanh_scale: float = 0.0,
    ):
        super().__init__()
        if variant not in self.MODEL_NAMES:
            raise ValueError(f"Unknown SegFormer variant: {variant}")

        try:
            from transformers import SegformerConfig, SegformerModel
        except ImportError as exc:
            raise ImportError(
                "SegFormerSourceGenerator requires the 'transformers' package. "
                "Install it or use a non-image prior."
            ) from exc

        self.num_classes = num_classes
        self.variant = variant
        self.learned_logvar = learned_logvar
        self.fixed_std = None if learned_logvar else fixed_std
        self.mu_tanh_scale = mu_tanh_scale

        if self.fixed_std is not None and self.fixed_std <= 0:
            raise ValueError("fixed_std must be positive when learned_logvar=False")

        model_name = self.MODEL_NAMES[variant]
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(model_name)
        else:
            config = SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=self.DEPTHS[variant],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=self.HIDDEN_SIZES[variant],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=self.NUM_HEADS[variant],
                mlp_ratios=[4, 4, 4, 4],
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                classifier_dropout_prob=0.1,
                drop_path_rate=0.1,
                hidden_act="gelu",
                initializer_range=0.02,
                layer_norm_eps=1e-6,
            )
            self.encoder = SegformerModel(config)

        hidden_sizes = list(self.encoder.config.hidden_sizes)
        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(ch, decoder_channels, kernel_size=1) for ch in hidden_sizes]
        )

        out_channels = num_classes * 2 if self.fixed_std is None else num_classes
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_channels * len(hidden_sizes), decoder_channels, kernel_size=3, padding=1),
            group_norm(decoder_channels),
            nn.SiLU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            group_norm(decoder_channels),
            nn.SiLU(),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, img: torch.Tensor):
        target_size = img.shape[-2:]
        mean = self.mean.to(device=img.device, dtype=img.dtype)
        std = self.std.to(device=img.device, dtype=img.dtype)
        pixel_values = (img - mean) / std

        outputs = self.encoder(
            pixel_values=pixel_values,
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

        out = self.decoder(torch.cat(feats, dim=1))
        if self.fixed_std is not None:
            mu = out
            logvar = torch.full_like(mu, math.log(float(self.fixed_std) ** 2))
        else:
            mu, logvar = out.split(self.num_classes, dim=1)

        if self.mu_tanh_scale > 0:
            mu = torch.tanh(mu) * self.mu_tanh_scale

        eps = torch.randn_like(mu)
        sigma = torch.exp(0.5 * logvar)
        x0 = mu + sigma * eps
        return x0, mu, logvar


# =========================================================
# example
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SegDiffModel(
        image_channels=3,
        mask_channels=20,
        fusion_channels=64,
        rrdb_blocks=6,
        unet_base_channels=64,
        unet_channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        attn_levels=(2, 3),
        dropout=0.1,
        num_heads=4,
    ).to(device)

    B, H, W = 2, 128, 128
    x_s = torch.randn(B, 20, H, W, device=device)   # 時刻 s における状態
    img = torch.randn(B, 3, H, W, device=device)   # conditioning image
    s = torch.rand(B, device=device)                # 始点時刻 ∈ [0, 1)
    t = s + torch.rand(B, device=device) * (1 - s)  # 終点時刻 ∈ [s, 1]

    # Case 1: flow map (s < t)
    out = model(x_s, img, s, t)
    print("flow map output shape:", out.shape)  # [B, 1, H, W]

    # Case 2: VFM / instantaneous velocity (s = t)
    out_vfm = model(x_s, img, s, s)
    print("VFM output shape:", out_vfm.shape)  # [B, 1, H, W]
