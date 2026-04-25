"""
MSRetinexFormer  —  Full 3-Stage Pipeline
==========================================
Combines all three stages described in the paper:

    Stage 1  (Fig. 2):  PN → DN → Gamma Correct → I_hat_1
    Stage 2  (Fig. 3):  Input + Prior → Conv Stage → [Sp Transformers] → MSSL output
    Stage 3  (Fig. 4):  MSSL → SON (LFIR + HFIR) → Final enhanced image

The fillu feature produced by Stage-1 PN is passed to Stage-2 spectral
transformers as an illumination bias (Section 3.4.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_retinex     import Stage1Net
from .dtcwt_transformer  import MultiScaleSpectralLayer
from .spectrum_optimizer import SpectrumOptimizationNetwork


class MSRetinexFormer(nn.Module):
    """
    Full MSRetinexFormer network.

    Args:
        in_channels     : input image channels (3 for RGB)
        base_channels   : base feature channels for Stage-1
        embed_dim       : embedding dimension for Stage-2 MSSL
        patch_size      : patch size for token embedding in Stage-2
        J               : DTCWT decomposition levels
        num_heads       : number of attention heads in IGAB blocks
        num_igab        : number of IGAB blocks in Stage-1 DN
        spatial_size    : spatial size of feature maps fed into Stage-3 SON
        b_factor        : channel factorisation factor for EFM in HFIR
        lambda_default  : initial gamma value for illumination correction
    """

    def __init__(
        self,
        in_channels    = 3,
        base_channels  = 64,
        embed_dim      = 64,
        patch_size     = 4,          # smaller patch for better spatial resolution
        J              = 2,
        num_heads      = 8,
        num_igab       = 4,
        spatial_size   = 32,         # H=W of SON feature map
        b_factor       = 4,
        lambda_default = 0.2,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        #  Stage 1: Illumination Estimation & Retinex Decomposition           #
        # ------------------------------------------------------------------ #
        self.stage1 = Stage1Net(
            in_channels    = in_channels,
            base_channels  = base_channels,
            num_heads      = num_heads,
            num_igab       = num_igab,
            lambda_default = lambda_default,
        )

        # ------------------------------------------------------------------ #
        #  Stage 2: Multi-Scale Frequency Decomposition (DTCWT)               #
        # ------------------------------------------------------------------ #
        self.stage2 = MultiScaleSpectralLayer(
            in_channels = in_channels,
            embed_dim   = embed_dim,
            patch_size  = patch_size,
            J           = J,
        )

        # ------------------------------------------------------------------ #
        #  Stage 3: Spectrum Optimization & Reconstruction                    #
        # ------------------------------------------------------------------ #
        # We first project the stage-2 output into embed_dim channels
        self.stage3_pre = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        self.stage3 = SpectrumOptimizationNetwork(
            channels     = embed_dim,
            spatial_size = spatial_size,
            b_factor     = b_factor,
            J            = 1,
        )
        # Final projection back to image space
        self.out_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, in_channels, 1),
            nn.Sigmoid(),
        )

        # Fusion: blend Stage-1 enhanced image with Stage-3 output
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x : low-light input image  [B, 3, H, W],  values in [0, 1]

        Returns (dict):
            'enhanced'    : final enhanced image        [B, 3, H, W]
            'stage1_out'  : Stage-1 enhanced image      [B, 3, H, W]
            'l_hat'       : illumination map             [B, 1, H, W]
            'r_hat'       : reflectance map              [B, 3, H, W]
            'l0'          : initial illumination         [B, 1, H, W]
            'fillu'       : illumination features        [B, C, H, W]
        """
        # ------ Stage 1 ------
        s1 = self.stage1(x)
        i_s1   = s1['enhanced']    # [B, 3, H, W]
        fillu  = s1['fillu']       # illumination features for Stage-2

        # Resize fillu to match embed_dim if needed (it's base_channels wide)
        fillu_resized = F.interpolate(fillu, size=x.shape[-2:],
                                      mode='bilinear', align_corners=False)

        # ------ Stage 2: MSSL ------
        # Feed the Stage-1 enhanced image into MSSL with illumination injection
        s2_out = self.stage2(i_s1, fillu_resized)    # [B, 3, H, W]

        # ------ Stage 3: SON ------
        feat   = self.stage3_pre(s2_out)             # [B, embed_dim, H, W]
        # Resize to spatial_size for SON
        h_son  = self.stage3.tbm.spatial_size
        feat_s = F.interpolate(feat, size=(h_son, h_son),
                               mode='bilinear', align_corners=False)
        feat_s = self.stage3(feat_s)                 # [B, embed_dim, h_son, h_son]
        # Resize back to original resolution
        feat_s = F.interpolate(feat_s, size=x.shape[-2:],
                               mode='bilinear', align_corners=False)
        s3_out = self.out_proj(feat_s)               # [B, 3, H, W]

        # ------ Final blend: alpha * Stage1 + (1-alpha) * Stage3 ------
        alpha    = self.alpha.clamp(0.0, 1.0)
        enhanced = alpha * i_s1 + (1 - alpha) * s3_out
        enhanced = enhanced.clamp(0, 1)

        return {
            'enhanced'   : enhanced,
            'stage1_out' : i_s1,
            'stage3_out' : s3_out,
            'l_hat'      : s1['l_hat'],
            'r_hat'      : s1['r_hat'],
            'l0'         : s1['l0'],
            'fillu'      : fillu,
        }

    def forward_pair(self, x1, x2):
        """
        Forward pass for paired training (Eq. 3 in paper).
        Returns outputs for both images — used by MSRetinexLoss.

        Args:
            x1, x2 : paired low-light images of the same scene [B, 3, H, W]
        """
        return self.forward(x1), self.forward(x2)
