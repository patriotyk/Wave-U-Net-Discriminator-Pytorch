import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Union


class GlobalNorm(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(
            torch.mean(x ** 2, dim=(1, 2), keepdim=True) + self.eps
        )
        return x / norm


class ResBlockDown(nn.Module):
    """
    Residual downsampling block.

    Args:
        in_ch:  number of input channels
        out_ch: number of output channels per path;
                total output = 2 × out_ch after Concat
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.skip_pool = nn.AvgPool1d(kernel_size=3, stride=3)
        self.skip_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=6, stride=3, padding=2)

        self.res_pool = nn.AvgPool1d(kernel_size=3, stride=3)

        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)

        self.norm = GlobalNorm()

    def _dup_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat channels along dim=1 until we reach exactly out_ch."""
        B, C, T = x.shape
        if C == self.out_ch:
            return x
        reps = -(-self.out_ch // C)
        return x.repeat(1, reps, 1)[:, :self.out_ch, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_pool(x)
        skip = self.skip_conv(skip)

        r = self.lrelu1(x)
        r = self.conv1(r)

        sc = self.res_pool(x)
        sc = self._dup_channels(sc)

        r = r + sc
        r = self.lrelu2(r)
        r = self.conv2(r)

        out = torch.cat([skip, r * 0.4], dim=1)
        return self.norm(out)


class ResBlockUp(nn.Module):
    """
    Residual upsampling block.

    Args:
        in_ch:  number of input channels
        out_ch: number of output channels per path;
                total output = 2 × out_ch after Concat
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.skip_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)


        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.convt = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=6, stride=3, padding=2, output_padding=1
        )

        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)

        self.norm = GlobalNorm()

    @staticmethod
    def _upsample(x: torch.Tensor, factor: int = 3) -> torch.Tensor:
        return F.interpolate(x, scale_factor=factor, mode="linear", align_corners=False)

    def _drop_channels(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :self.out_ch, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self._upsample(x)
        skip = self.skip_conv(skip)

        r = self.lrelu1(x)
        r = self.convt(r)

        sc = self._upsample(x)
        sc = self._drop_channels(sc)

        r = r + sc
        r = self.lrelu2(r)
        r = self.conv2(r)

        out = torch.cat([skip, r * 0.4], dim=1)
        return self.norm(out)


class WaveUNetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ResBlockDown(in_ch=1,    out_ch=32)
        self.enc2 = ResBlockDown(in_ch=64,   out_ch=64)
        self.enc3 = ResBlockDown(in_ch=128,  out_ch=128)
        self.enc4 = ResBlockDown(in_ch=256,  out_ch=256)
        self.enc5 = ResBlockDown(in_ch=512,  out_ch=512)

        self.dec1 = ResBlockUp(in_ch=1024, out_ch=256)
        self.dec2 = ResBlockUp(in_ch=1024, out_ch=128)
        self.dec3 = ResBlockUp(in_ch=512,  out_ch=64)
        self.dec4 = ResBlockUp(in_ch=256,  out_ch=32)
        self.dec5 = ResBlockUp(in_ch=128,  out_ch=32)

        self.out_conv = nn.Conv1d(64, 1, kernel_size=5, stride=1, padding=2)

    def pad_to_multiple(self, x: torch.Tensor, multiple: int = 243) -> tuple[torch.Tensor, int]:
        T = x.shape[-1]
        pad = (multiple - T % multiple) % multiple
        return F.pad(x, (0, pad)), pad
    
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        
        x, pad = self.pad_to_multiple(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d1 = self.dec1(e5)
        d2 = self.dec2(torch.cat([d1, e4], dim=1))
        d3 = self.dec3(torch.cat([d2, e3], dim=1))
        d4 = self.dec4(torch.cat([d3, e2], dim=1))
        d5 = self.dec5(torch.cat([d4, e1], dim=1))

        logits = self.out_conv(d5)
        if pad > 0:
            logits = logits[..., :-pad]

        if not return_features:
            return logits

        features = [e1, e2, e3, e4, e5, d1, d2, d3, d4, d5]
        return logits, features


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Least-squares GAN loss for the discriminator

    """
    loss_real = torch.mean((real_logits - 1.0) ** 2)
    loss_fake = torch.mean(fake_logits ** 2)
    return loss_real + loss_fake


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Least-squares GAN loss for the generator

    """
    return torch.mean((fake_logits - 1.0) ** 2)


def feature_matching_loss(
    real_features: list[torch.Tensor],
    fake_features: list[torch.Tensor],
) -> torch.Tensor:
    """
    Feature-matching loss for the generator

    """
    assert len(real_features) == len(fake_features), \
        "real and fake feature lists must have the same length"

    loss = torch.tensor(0.0, device=real_features[0].device)
    for real_feat, fake_feat in zip(real_features, fake_features):
        N_i = real_feat.numel() / real_feat.shape[0]
        loss = loss + torch.sum(torch.abs(real_feat - fake_feat)) / (real_feat.shape[0] * N_i)

    return loss