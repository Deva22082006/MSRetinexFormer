"""
Spectrum Optimization Network (SON)  —  Stage 3
================================================
Implements Section 3.4 of the paper:
  - Low-Frequency Information Refinement  (LFIR)  via Tensor Fusion (Eq. 15)
  - High-Frequency Information Refinement (HFIR)  via Einstein Fusion (Eq. 16)
  - Spectral Channel Mixing (SCM)
  - Token Mixing (TM)
  - Final inverse DTCWT reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import opt_einsum as oe          # installed via pytorch_wavelets deps
import math


# ---------------------------------------------------------------------------
# Helper: safe einsum wrapper (falls back to torch.einsum if opt_einsum absent)
# ---------------------------------------------------------------------------

def _einsum(pattern, *tensors):
    try:
        return oe.contract(pattern, *tensors)
    except Exception:
        return torch.einsum(pattern, *tensors)


# ---------------------------------------------------------------------------
# Low-Frequency Information Refinement  (Section 3.4.1)
# ---------------------------------------------------------------------------

class LFIR(nn.Module):
    """
    Tensor Fusion Method (TFM) for low-frequency refinement.

        M_φ = X_φ ⊙ W_φ       (Hadamard / element-wise product, Eq. 15)

    The learnable weight W_φ has the same shape as the LF subband, allowing
    the network to dynamically adjust global illumination and large-scale
    contrast.
    """

    def __init__(self, channels):
        super().__init__()
        # Learnable per-channel spatial weight
        self.wl = nn.Parameter(torch.ones(1, channels, 1, 1))
        # Optional 1×1 conv refinement after Hadamard
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x_phi):
        """
        Args:
            x_phi : low-frequency tensor  [B, C, H, W]
        Returns:
            m_phi : refined LF tensor     [B, C, H, W]
        """
        m_phi = x_phi * self.wl          # Hadamard product (Eq. 15)
        m_phi = m_phi + self.refine(m_phi)
        return m_phi


# ---------------------------------------------------------------------------
# High-Frequency Information Refinement  (Section 3.4.2)
# ---------------------------------------------------------------------------

class HFIR(nn.Module):
    """
    Einstein Fusion Method (EFM) for high-frequency refinement.

        S_ψ = X_ψ ⊗ W_ψ       (Einstein product, Eq. 16)

    To avoid parameter explosion from full tensor fusion, the HF tensor is
    reshaped to R^{H×W×Cb×Cd} and the weight is R^{Cb×Cd×Cd}, reducing
    quadratic channel growth to linear.
    """

    def __init__(self, channels, b_factor=4):
        """
        Args:
            channels : total feature channels C
            b_factor : b in C = Cb × Cd; Cb = b_factor, Cd = C // b_factor
        """
        super().__init__()
        assert channels % b_factor == 0, "channels must be divisible by b_factor"
        self.cb = b_factor
        self.cd = channels // b_factor

        # W_ψ ∈ R^{Cb × Cd × Cd}
        self.wh = nn.Parameter(
            torch.randn(self.cb, self.cd, self.cd) * 0.02
        )
        self.norm = nn.LayerNorm(channels)
        self.act  = nn.GELU()

    def forward(self, x_psi):
        """
        Args:
            x_psi : high-frequency tensor  [B, C, H, W]
        Returns:
            s_psi : refined HF tensor      [B, C, H, W]
        """
        B, C, H, W = x_psi.shape
        Cb, Cd = self.cb, self.cd

        # Reshape: [B, C, H, W] -> [B, H, W, Cb, Cd]
        x = x_psi.permute(0, 2, 3, 1).reshape(B, H, W, Cb, Cd)

        # Einstein product: contract over Cd → [B, H, W, Cb, Cd]
        # Using einsum: "bhwij, ijk -> bhwik"
        s = _einsum('bhwij,ijk->bhwik', x, self.wh)

        # Reshape back: [B, H, W, Cb, Cd] -> [B, C, H, W]
        s = s.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Residual + norm
        s = x_psi + self.act(s)
        s = self.norm(s.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return s


# ---------------------------------------------------------------------------
# Spectral Channel Mixing  (Section 3.4.4)
# ---------------------------------------------------------------------------

class SpectralChannelMixing(nn.Module):
    """
    SCM reshapes HF coefficients to R^{2×k×H×W×Cb×Cd} then applies an
    Einstein product with W_ψc ∈ R^{Cb×Cd×Cd}.
    """

    def __init__(self, channels, b_factor=4):
        super().__init__()
        assert channels % b_factor == 0
        self.cb = b_factor
        self.cd = channels // b_factor
        self.wc = nn.Parameter(
            torch.randn(self.cb, self.cd, self.cd) * 0.02
        )
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x : [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        Cb, Cd = self.cb, self.cd
        xr = x.permute(0, 2, 3, 1).reshape(B, H, W, Cb, Cd)
        out = _einsum('bhwij,ijk->bhwik', xr, self.wc)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.act(x + out)


# ---------------------------------------------------------------------------
# Token Mixing  (Section 3.4.4)
# ---------------------------------------------------------------------------

class TokenMixing(nn.Module):
    """
    TM reshapes HF components to R^{2×k×C×W×H} then applies
    Einstein product with W_ψt ∈ R^{W×H×HW}.
    """

    def __init__(self, spatial_size):
        """
        Args:
            spatial_size: H (= W assumed) of the feature map
        """
        super().__init__()
        hw = spatial_size * spatial_size
        # W_ψt ∈ R^{H×W×HW} — project spatial tokens
        self.wt = nn.Parameter(
            torch.randn(spatial_size, spatial_size, hw) * (1.0 / math.sqrt(hw))
        )
        self.spatial_size = spatial_size
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x : [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        if H != self.spatial_size or W != self.spatial_size:
            # Resize weight at test time if needed
            wt = F.interpolate(
                self.wt.unsqueeze(0).unsqueeze(0),
                size=(H, W * H * W), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0).view(H, W, H * W)
        else:
            wt = self.wt

        # x: [B, C, H, W] -> [B, C, H, W]
        # token mix: for each channel, mix spatial positions
        xr = x.reshape(B * C, H * W)          # [B*C, HW]
        # wt: [H, W, HW] -> [HW, HW]
        wt_flat = wt.reshape(H * W, H * W)    # square mixing matrix
        out = xr @ wt_flat.T                  # [B*C, HW]
        out = out.reshape(B, C, H, W)
        return self.act(x + out)


# ---------------------------------------------------------------------------
# Full Spectrum Optimization Network  (Section 3.4, Fig. 4)
# ---------------------------------------------------------------------------

class SpectrumOptimizationNetwork(nn.Module):
    """
    SON: Spectrum Optimization Network

    Architecture (Fig. 4):
        Input → MSSL → LF branch → TBM → LFR → SON output
                      → HF branch → EBM → HFR → SON output
        Combined output → Inverse DTCWT → Final enhanced image
    """

    def __init__(self, channels=64, spatial_size=16, b_factor=4, J=1):
        """
        Args:
            channels     : feature channels
            spatial_size : H=W of feature map fed into SON
            b_factor     : Cb in channel factorisation for EFM
            J            : DTCWT levels
        """
        super().__init__()
        self.J = J
        self.xfm = DTCWTForward(J=J, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        # LF path: Token-Based Mixing → Low-Frequency Refinement
        self.tbm  = TokenMixing(spatial_size)    # TBM on LF
        self.lfir = LFIR(channels)

        # HF path: Embedding-Based Mixing → High-Frequency Refinement
        self.ebm  = SpectralChannelMixing(channels, b_factor)  # EBM on HF
        self.hfir = HFIR(channels, b_factor)

        # Final projection
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, feat):
        """
        Args:
            feat : feature tensor [B, C, H, W]
        Returns:
            optimised feature [B, C, H, W]
        """
        B, C, H, W = feat.shape

        # DTCWT decompose
        ll, lh = self.xfm(feat)

        # ---------- LF path ----------
        # Upsample LL to feat spatial size for TBM / LFIR
        ll_up = F.interpolate(ll, size=(H, W), mode='bilinear', align_corners=False)
        lf = self.tbm(ll_up)    # Token-Based Mixing
        lf = self.lfir(lf)      # Tensor Fusion (Eq. 15)

        # ---------- HF path ----------
        # Reconstruct pure HF signal
        zeros = torch.zeros_like(ll)
        hf = self.ifm((zeros, lh))
        hf = self.ebm(hf)       # Embedding-Based Mixing (SCM)
        hf = self.hfir(hf)      # Einstein Fusion (Eq. 16)

        # ---------- Combine ----------
        out = lf + hf
        out = self.proj(out)
        return feat + out       # residual
