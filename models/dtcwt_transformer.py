"""
DTCWT Transformer Module
========================
Implements the Dual-Tree Complex Wavelet Transform (DTCWT) based
multi-scale spectral layers described in Section 3.2 and 3.3 of the paper.

Key components:
  - DTCWTForward / DTCWTInverse wrappers
  - Multi-Scale Spectral Layer (MSSL)
  - INF Mixing Module (Illumination-guided Neural Frequency mixing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Standard Conv -> BN -> ReLU block."""
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block used inside encoder/decoder."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ---------------------------------------------------------------------------
# INF Mixing Module  (Section 3.4.3)
# ---------------------------------------------------------------------------

class INFMixingModule(nn.Module):
    """
    Illumination-guided Neural Frequency (INF) Mixing Module.

    Decomposes input features with DTCWT, applies learnable complex-valued
    spectral modulation (separate weights for real & imaginary parts),
    injects global illumination as an additive bias, then reconstructs
    via inverse DTCWT.

        Ffused = W_real * Re(X) + W_imag * Im(X)   (Eq. 17)
    """

    def __init__(self, channels, J=1):
        """
        Args:
            channels: number of feature channels
            J       : wavelet decomposition levels
        """
        super().__init__()
        self.J = J
        self.xfm = DTCWTForward(J=J, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        # Learnable weights for real and imaginary spectral modulation
        # 6 subbands per level → 6 * 2 (real/imag) channel groups
        self.w_real = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.w_imag = nn.Parameter(torch.ones(1, channels, 1, 1))

        # Illumination bias projection
        self.illum_proj = nn.Conv2d(channels, channels, 1)

        # Low-frequency learnable weight (WLL in Section 3.3.2)
        self.wll = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x, fillu=None):
        """
        Args:
            x     : feature tensor  [B, C, H, W]
            fillu : illumination feature [B, C, H, W] (optional)
        Returns:
            fused feature [B, C, H, W]
        """
        # DTCWT decomposition
        x_ll, x_lh = self.xfm(x)   # x_ll: [B,C,H/2,W/2], x_lh: list of complex tensors

        # ---------- Low-frequency calibration (Eq. 15 analog) ----------
        x_ll = x_ll * self.wll

        # ---------- High-frequency spectral modulation (Eq. 17) ----------
        processed_lh = []
        for level_coeff in x_lh:
            # level_coeff shape: [B, C, 6, H', W', 2]  (6 subbands, 2=real/imag)
            real_part = level_coeff[..., 0]  # [B, C, 6, H', W']
            imag_part = level_coeff[..., 1]

            # Channel-wise modulation (broadcast over subband dim)
            fused_real = self.w_real.unsqueeze(2) * real_part
            fused_imag = self.w_imag.unsqueeze(2) * imag_part

            # Inject illumination bias if provided
            if fillu is not None:
                bias = self.illum_proj(fillu)  # [B, C, H, W]
                # Resize bias to match subband spatial dims
                bias_r = F.adaptive_avg_pool2d(bias, real_part.shape[-2:])
                fused_real = fused_real + bias_r.unsqueeze(2)
                fused_imag = fused_imag + bias_r.unsqueeze(2)

            # Recombine real / imaginary
            new_coeff = torch.stack([fused_real, fused_imag], dim=-1)
            processed_lh.append(new_coeff)

        # Inverse DTCWT
        out = self.ifm((x_ll, processed_lh))
        return out


# ---------------------------------------------------------------------------
# Multi-Scale Spectral Layer  (Section 3.3)
# ---------------------------------------------------------------------------

class MultiScaleSpectralLayer(nn.Module):
    """
    MSSL: Multi-Scale Spectral Layer (Stage-2 block, Fig. 3)

    Pipeline:
        Input -> PatchEmbed -> [Conv Stage] -> DTCWT decompose
             -> [Sp Transformer x2] (LF branch)
             -> [Sp Transformer x2] (HF branch)
             -> fuse -> Output

    The 'Sp Transformer' (Spectral Transformer) here is realised as the
    INFMixingModule combined with a feed-forward projection.
    """

    def __init__(self, in_channels=3, embed_dim=64, patch_size=16, J=2):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.J          = J

        # --- Token embedding (FT) + position encoding (FP), Eq. 10 ---
        self.token_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed   = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # --- Convolution Stage (Fig. 3 centre block) ---
        self.conv_stage = nn.Sequential(
            ConvBNReLU(embed_dim, embed_dim, 3, 1, 1),
            ResBlock(embed_dim),
            ConvBNReLU(embed_dim, embed_dim, 3, 1, 1),
        )

        # --- DTCWT ---
        self.xfm = DTCWTForward(J=J, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.wll = nn.Parameter(torch.ones(1, embed_dim, 1, 1))

        # --- 4 Spectral Transformers (2 LF path, 2 HF path) ---
        self.sp_lf1 = INFMixingModule(embed_dim, J=1)
        self.sp_lf2 = INFMixingModule(embed_dim, J=1)
        self.sp_hf1 = INFMixingModule(embed_dim, J=1)
        self.sp_hf2 = INFMixingModule(embed_dim, J=1)

        # --- Fusion + upsample back to original resolution ---
        self.fuse   = nn.Conv2d(embed_dim * 2, embed_dim, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, in_channels * patch_size * patch_size, 1),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x, fillu=None):
        """
        Args:
            x    : low-light image [B, 3, H, W]
            fillu: illumination feature from Stage-1 [B, C, H, W] (optional)
        Returns:
            enhanced feature [B, 3, H, W]
        """
        B, C, H, W = x.shape

        # Patch embedding (Eq. 10): X = FT(I) + FP(I)
        ft = self.token_embed(x)   # [B, D, H/p, W/p]
        fp = self.pos_embed(x)
        feat = ft + fp             # X ∈ R^{C x H' x W'}

        # Convolution stage
        feat = self.conv_stage(feat)   # still [B, D, H', W']

        # DTCWT decompose
        ll, lh = self.xfm(feat)
        ll = ll * self.wll          # calibrate LF subband

        # --- LF branch: two Sp Transformers ---
        # Upsample LL to feat size for processing
        ll_up = F.interpolate(ll, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        lf_out = self.sp_lf1(ll_up, fillu)
        lf_out = self.sp_lf2(lf_out, fillu)

        # --- HF branch: reconstruct HF-only signal, then two Sp Transformers ---
        zeros_ll = torch.zeros_like(ll)
        hf_signal = self.ifm((zeros_ll, lh))
        hf_out = self.sp_hf1(hf_signal, fillu)
        hf_out = self.sp_hf2(hf_out, fillu)

        # Fuse LF + HF features (concatenate on channel dim)
        fused = self.fuse(torch.cat([lf_out, hf_out], dim=1))

        # Upsample back to input resolution
        out = self.upsample(fused)    # [B, 3, H, W]
        return out
