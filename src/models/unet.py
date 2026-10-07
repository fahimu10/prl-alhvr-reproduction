"""U-Net backbones: `UNet` (U-Net baseline, CPS) and `UNet_fea_aux` (ALHVR).

Architecture ported from ALHVR's networks/unet.py. `UNet_fea_aux` is the
ALHVR-specific variant with feature-noise/dropout auxiliary decoders.
Original attributes this architecture to https://github.com/HiLab-git/PyMIC.
Device-agnostic: the original hardcodes .cuda(), this doesn't, so the same
code runs on CUDA, MPS or CPU.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform


class DinoBlock(nn.Module):
    """PART 2 ONLY - frozen DINOv2 features projected to (B, out_channels, H, W).

    Not part of the Part 1 reproduction: `Encoder` only builds this when
    params["use_dino"] is true, which no Part 1 config sets.

    DINOv2 is patch-14, so for a 256 input its feature map is 18x18 - a ~14x
    spatial reduction that is then upsampled back. The result carries DINOv2's
    semantics at DINOv2's granularity, NOT pixel detail, which is why the
    encoder concatenates it alongside `in_conv` rather than replacing it.

    `timm` is an optional dependency; the import is deferred so Part 1 runs on
    a machine without it.
    """

    def __init__(self, out_channels, model="vit_small_patch14_dinov2",
                 img_size=256, pretrained=True, weights_file=None):
        super().__init__()
        import timm

        overlay = {"pretrained_cfg_overlay": dict(file=weights_file)} if weights_file else {}
        self.dino = timm.create_model(model, features_only=True, img_size=img_size,
                                      in_chans=1, pretrained=pretrained, **overlay)
        # Freezing needs all three: no grads, eval mode (dropout/droppath off),
        # and the train() override below so a parent .train() cannot undo it.
        for q in self.dino.parameters():
            q.requires_grad = False
        self.dino.eval()

        in_ch = self.dino.feature_info[-1]["num_chs"]
        # 1x1 conv is a per-pixel linear layer; GroupNorm(1, C) is LayerNorm
        # over (C, H, W). Only these are trainable.
        self.proj = nn.Conv2d(in_ch, out_channels, kernel_size=1)
        self.act = nn.GELU()
        self.norm = nn.GroupNorm(1, out_channels)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, x):
        size = x.shape[-2:]
        with torch.no_grad():            # no autograd graph through the frozen trunk
            feat = self.dino(x)[-1]
        feat = self.norm(self.act(self.proj(feat)))
        return F.interpolate(feat, size=size, mode="bilinear", align_corners=False)


class ConvBlock(nn.Module):
    """Two convolution layers with batch norm and leaky relu."""

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock."""

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling followed by ConvBlock."""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super().__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1_high = self.up(x1)
        x = torch.cat([x2, x1_high], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_chns = params["in_chns"]
        ft_chns = params["feature_chns"]
        dropout = params["dropout"]
        assert len(ft_chns) == 5
        # PART 2 ONLY, default off: Part 1 must stay bit-identical to the port.
        self.use_dino = params.get("use_dino", False)
        if self.use_dino:
            self.dino = DinoBlock(out_channels=ft_chns[0],
                                  pretrained=params.get("dino_pretrained", True),
                                  weights_file=params.get("dino_weights"))
            self.fuse = nn.Conv2d(ft_chns[0] * 2, ft_chns[0], kernel_size=1)
        self.in_conv = ConvBlock(in_chns, ft_chns[0], dropout[0])
        self.down1 = DownBlock(ft_chns[0], ft_chns[1], dropout[1])
        self.down2 = DownBlock(ft_chns[1], ft_chns[2], dropout[2])
        self.down3 = DownBlock(ft_chns[2], ft_chns[3], dropout[3])
        self.down4 = DownBlock(ft_chns[3], ft_chns[4], dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        if self.use_dino:
            # Concatenate, not replace: x0 is the full-resolution skip the
            # decoder needs for boundary detail, and DINOv2's map is 14x
            # coarser. Fusing keeps both.
            x0 = self.fuse(torch.cat([x0, self.dino(x)], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        ft_chns = params["feature_chns"]
        n_class = params["class_num"]
        up_type = params["up_type"]
        assert len(ft_chns) == 5
        self.up1 = UpBlock(ft_chns[4], ft_chns[3], ft_chns[3], dropout_p=0.0, mode_upsampling=up_type)
        self.up2 = UpBlock(ft_chns[3], ft_chns[2], ft_chns[2], dropout_p=0.0, mode_upsampling=up_type)
        self.up3 = UpBlock(ft_chns[2], ft_chns[1], ft_chns[1], dropout_p=0.0, mode_upsampling=up_type)
        self.up4 = UpBlock(ft_chns[1], ft_chns[0], ft_chns[0], dropout_p=0.0, mode_upsampling=up_type)
        self.out_conv = nn.Conv2d(ft_chns[0], n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.out_conv(x)


class UpBlock_fea(nn.Module):
    """UpBlock that also returns the pre-upsampling 1x1 projection.

    That second return value is `F^d` in the paper - the penultimate
    decoder feature CG-CPCL builds class prototypes from.
    """

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super().__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1_high = self.up(x1)
        x = torch.cat([x2, x1_high], dim=1)
        x = self.conv(x)
        return x, x1


class Decoder_fea(nn.Module):
    """Decoder that also emits the penultimate feature map (via up2)."""

    def __init__(self, params):
        super().__init__()
        ft_chns = params["feature_chns"]
        n_class = params["class_num"]
        up_type = params["up_type"]
        assert len(ft_chns) == 5
        self.up1 = UpBlock(ft_chns[4], ft_chns[3], ft_chns[3], dropout_p=0.0, mode_upsampling=up_type)
        self.up2 = UpBlock_fea(ft_chns[3], ft_chns[2], ft_chns[2], dropout_p=0.0, mode_upsampling=up_type)
        self.up3 = UpBlock(ft_chns[2], ft_chns[1], ft_chns[1], dropout_p=0.0, mode_upsampling=up_type)
        self.up4 = UpBlock(ft_chns[1], ft_chns[0], ft_chns[0], dropout_p=0.0, mode_upsampling=up_type)
        self.out_conv = nn.Conv2d(ft_chns[0], n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x, x6 = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output, x6


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super().__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


def feature_chns(width):
    """Channel counts per level from a base width: [w, 2w, 4w, 8w, 16w].

    width=16 is ALHVR's own setting and the only one valid for Part 1.
    Larger values are a Part 2 capacity experiment - depth, resolutions and
    every other hyperparameter are unchanged, so only channel count varies.
    """
    return [width * m for m in (1, 2, 4, 8, 16)]


class UNet(nn.Module):
    def __init__(self, in_chns, class_num, width=16, use_dino=False,
                 dino_pretrained=True, dino_weights=None):
        super().__init__()
        params = {
            "in_chns": in_chns,
            "feature_chns": feature_chns(width),
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": class_num,
            "up_type": 1,
            "acti_func": "relu",
            "use_dino": use_dino,
            "dino_pretrained": dino_pretrained,
            "dino_weights": dino_weights,
        }
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        return self.decoder(feature)


class UNet_fea_aux(nn.Module):
    """ALHVR's backbone: one decoder run on clean and on perturbed features.

    Returns (out_seg, out_seg_aux, fea, fea_aux). The perturbation is
    chosen per forward call - feature dropout or uniform feature noise,
    50/50 - which is what makes the "aux" branch a perturbed view of the
    same sample. Note the decoder is shared (called twice), not duplicated.
    """

    def __init__(self, in_chns, class_num, width=16, use_dino=False,
                 dino_pretrained=True, dino_weights=None):
        super().__init__()
        params1 = {
            "in_chns": in_chns,
            "feature_chns": feature_chns(width),
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": class_num,
            "up_type": 1,
            "acti_func": "relu",
            "use_dino": use_dino,
            "dino_pretrained": dino_pretrained,
            "dino_weights": dino_weights,
        }
        self.encoder = Encoder(params1)
        self.decoder = Decoder_fea(params1)

    def forward(self, x):
        feature = self.encoder(x)
        random_number = random.random()
        if random_number > 0.5:
            aux_feature = [FeatureDropout(i) for i in feature]
        else:
            aux_feature = [FeatureNoise()(i) for i in feature]
        out_seg, fea = self.decoder(feature)
        out_seg_aux, fea_aux = self.decoder(aux_feature)
        return out_seg, out_seg_aux, fea, fea_aux
