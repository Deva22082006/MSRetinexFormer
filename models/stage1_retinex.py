"""
Stage 1: Illumination Estimation & Initial Retinex Decomposition
================================================================
Implements the first stage of the pipeline (Fig. 2 of the paper).

Components:
  PN  – Processing Network   : extracts illumination features F_illu and L0
  DN  – Decomposition Network: Transformer encoder-decoder with IGAB blocks
                               that reconstructs R_hat
  GammaCorrect               : applies gamma correction to illumination L_hat
  Stage1Net                  : full Stage-1 wrapper

Block diagram (Fig. 2):
    Input ──► PN ──► DN ──► [i branch → Gamma Correcting] ──► ⊗ ──► Enhanced
                    │                                              ▲
                    └──► [r branch] ─────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-last LayerNorm for 2-D feature maps [B, C, H, W]."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        # Normalize over C dim only, keeping [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DepthwiseSepConv(nn.Module):
    """Depthwise separable convolution block."""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size, padding=padding, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# ---------------------------------------------------------------------------
# IGAB – Illumination-Guided Attention Block  (based on IG-MSA in paper)
# ---------------------------------------------------------------------------

class IGAB(nn.Module):
    """
    Illumination-Guided Attention Block (IG-MSA).

    Uses local window-based attention so memory scales as O(ws^2) not O(H*W).
    Illumination map L0 gates the Query to guide brightness-aware attention.

    Attention shape audit for ws=8, C=64, heads=8:
        q/k/v after reshape: [Bw, heads, ws*ws, head_dim]
                                               64    8
        attn = q @ k^T       : [Bw, heads, 64, 64]   ✓ safe
    """

    def __init__(self, channels, num_heads=8, ffn_ratio=4, window_size=8):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.norm1       = LayerNorm2d(channels)
        self.norm2       = LayerNorm2d(channels)
        self.num_heads   = num_heads
        self.window_size = window_size
        self.head_dim    = channels // num_heads
        # scale by head_dim, NOT by ws*ws (critical fix)
        self.scale       = self.head_dim ** -0.5

        self.qkv        = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.illum_gate = nn.Conv2d(1, channels, 1)
        self.proj       = nn.Conv2d(channels, channels, 1)

        hidden = int(channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    # ------------------------------------------------------------------
    # Window helpers  (standard Swin-style, channel-last inside windows)
    # ------------------------------------------------------------------

    @staticmethod
    def _partition(x, ws):
        """
        [B, C, H, W]  ->  [B*nW, ws*ws, C]
        where nW = (H/ws) * (W/ws)
        """
        B, C, H, W = x.shape
        # -> [B, H/ws, ws, W/ws, ws, C]
        x = x.permute(0, 2, 3, 1)                       # [B, H, W, C]
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # [B, nH, nW, ws, ws, C]
        return x.view(-1, ws * ws, C)                    # [B*nH*nW, ws², C]

    @staticmethod
    def _reverse(x, ws, H, W, B):
        """
        [B*nW, ws*ws, C]  ->  [B, C, H, W]
        """
        nH, nW = H // ws, W // ws
        C = x.shape[-1]
        x = x.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # [B, nH, ws, nW, ws, C]
        x = x.view(B, H, W, C)
        return x.permute(0, 3, 1, 2)                    # [B, C, H, W]

    # ------------------------------------------------------------------

    def forward(self, x, l0=None):
        """
        Args:
            x  : [B, C, H, W]
            l0 : [B, 1, H, W]  illumination estimate (optional)
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad spatial dims to be divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]

        # ---- Norm + QKV ----
        shortcut = x
        xn = self.norm1(x)                              # [B, C, Hp, Wp]

        qkv = self.qkv(xn)                             # [B, 3C, Hp, Wp]
        q, k, v = qkv.chunk(3, dim=1)                  # each [B, C, Hp, Wp]

        # ---- Illumination gate on Q ----
        if l0 is not None:
            l0r  = F.interpolate(l0, size=(Hp, Wp), mode='bilinear', align_corners=False)
            gate = torch.sigmoid(self.illum_gate(l0r))  # [B, C, Hp, Wp]
            q    = q * gate

        # ---- Window partition  -> [Bw, ws², C] ----
        q = self._partition(q, ws)   # [Bw, ws², C]
        k = self._partition(k, ws)
        v = self._partition(v, ws)

        Bw = q.shape[0]   # B * nH * nW

        # ---- Multi-head reshape  -> [Bw, heads, ws², head_dim] ----
        q = q.view(Bw, ws * ws, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(Bw, ws * ws, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(Bw, ws * ws, self.num_heads, self.head_dim).transpose(1, 2)
        # shapes: [Bw, heads, ws², head_dim]

        # ---- Attention  [Bw, heads, ws², ws²] ----
        # Maximum allocation: Bw * heads * ws² * ws²  (e.g. 256*8*64*64 = 8M floats ≈ 32 MB)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # ---- Weighted sum  [Bw, heads, ws², head_dim] ----
        out = attn @ v

        # ---- Merge heads  [Bw, ws², C] ----
        out = out.transpose(1, 2).contiguous().view(Bw, ws * ws, C)

        # ---- Reverse windows  [B, C, Hp, Wp] ----
        out = self._reverse(out, ws, Hp, Wp, B)
        out = self.proj(out)
        out = out + shortcut

        # Remove padding
        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        # ---- FFN ----
        out = out + self.ffn(self.norm2(out))
        return out


# ---------------------------------------------------------------------------
# Processing Network (PN) — Fig. 2 left block
# ---------------------------------------------------------------------------

class ProcessingNetwork(nn.Module):
    """
    PN extracts:
        F_illu : illumination feature map  [B, C, H, W]
        L0     : initial illumination map   [B, 1, H, W]

    Architecture:
        Conv stem → residual blocks → two heads (F_illu, L0)
    """

    def __init__(self, in_channels=3, base_channels=64, num_res=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            *[self._res_block(base_channels) for _ in range(num_res)]
        )
        # F_illu head
        self.illum_head = nn.Conv2d(base_channels, base_channels, 1)
        # L0 head: single-channel initial illumination estimate
        self.l0_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def _res_block(c):
        return nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = feat + self.body(feat)           # residual
        fillu = self.illum_head(feat)           # F_illu [B, C, H, W]
        l0    = self.l0_head(feat)              # L0     [B, 1, H, W]
        return fillu, l0


# ---------------------------------------------------------------------------
# Decomposition Network (DN) — Fig. 2 right block
# ---------------------------------------------------------------------------

class DecompositionNetwork(nn.Module):
    """
    DN: encoder-decoder with IGAB blocks.

    Takes as input the fused (F_illu + image features) and produces:
        L_hat : enhanced illumination map  [B, 1, H, W]
        R_hat : reflectance map            [B, 3, H, W]
    """

    def __init__(self, in_channels=3, base_channels=64, fillu_channels=64,
                 num_heads=8, num_igab=4):
        super().__init__()
        fused_c = in_channels + fillu_channels   # concat image + F_illu

        # Encoder
        self.enc_conv = nn.Conv2d(fused_c, base_channels, 3, 1, 1)
        self.enc_igab = nn.ModuleList([
            IGAB(base_channels, num_heads) for _ in range(num_igab // 2)
        ])
        self.down = nn.Conv2d(base_channels, base_channels * 2, 2, 2)

        # Bottleneck
        self.bottleneck = IGAB(base_channels * 2, min(num_heads * 2, 16))

        # Decoder
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec_fuse = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.dec_igab = nn.ModuleList([
            IGAB(base_channels, num_heads) for _ in range(num_igab // 2)
        ])

        # Output heads
        self.l_head = nn.Sequential(nn.Conv2d(base_channels, 1, 1), nn.Sigmoid())
        self.r_head = nn.Sequential(nn.Conv2d(base_channels, 3, 1), nn.Sigmoid())

    def forward(self, x, fillu, l0):
        """
        Args:
            x     : input image       [B, 3, H, W]
            fillu : illum features    [B, C, H, W]
            l0    : initial illum     [B, 1, H, W]
        Returns:
            l_hat : illumination map  [B, 1, H, W]
            r_hat : reflectance map   [B, 3, H, W]
        """
        # Fuse image and illumination features
        fused = torch.cat([x, fillu], dim=1)    # [B, 3+C, H, W]
        enc   = self.enc_conv(fused)            # [B, C, H, W]

        # Encoder IGAB blocks
        enc_skip = enc
        for blk in self.enc_igab:
            enc = blk(enc, l0)

        # Down-sample
        down = self.down(enc)                   # [B, 2C, H/2, W/2]
        bot  = self.bottleneck(down)

        # Up-sample + skip connection
        up = self.up(bot)                       # [B, C, H, W]
        up = torch.cat([up, enc_skip], dim=1)  # [B, 2C, H, W]
        up = self.dec_fuse(up)                  # [B, C, H, W]

        for blk in self.dec_igab:
            up = blk(up, l0)

        l_hat = self.l_hat_with_l0(up, l0)
        r_hat = self.r_head(up)
        return l_hat, r_hat

    def l_hat_with_l0(self, feat, l0):
        """Ensure L_hat is informed by L0 (third regularisation term in Eq. 19)."""
        raw = self.l_head(feat)
        # Blend with L0 so initialisation guides final prediction
        return 0.5 * raw + 0.5 * l0


# ---------------------------------------------------------------------------
# Gamma Correction module
# ---------------------------------------------------------------------------

class GammaCorrection(nn.Module):
    """
    Learnable gamma correction applied to the illumination map.
    I_enhanced = R_hat ⊙ GammaCorrect(L_hat)
    """

    def __init__(self, lambda_default=0.2):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(lambda_default))

    def forward(self, l_hat):
        # Clamp gamma to keep correction physically meaningful
        gamma = self.gamma.clamp(0.05, 1.0)
        return l_hat ** gamma


# ---------------------------------------------------------------------------
# Full Stage-1 Network
# ---------------------------------------------------------------------------

class Stage1Net(nn.Module):
    """
    Complete Stage-1: Illumination Estimation & Retinex Decomposition.

    I = L ⊙ R  (Eq. 1)
    I_hat = L_hat ⊙ R_hat

    Forward returns:
        i_hat  : enhanced image          [B, 3, H, W]
        l_hat  : illumination map        [B, 1, H, W]
        r_hat  : reflectance map         [B, 3, H, W]
        l0     : initial illum estimate  [B, 1, H, W]
        fillu  : illum features (for Stage-2 injection)
    """

    def __init__(self, in_channels=3, base_channels=64,
                 num_heads=8, num_igab=4, lambda_default=0.2):
        super().__init__()
        self.pn    = ProcessingNetwork(in_channels, base_channels)
        self.dn    = DecompositionNetwork(in_channels, base_channels,
                                          base_channels, num_heads, num_igab)
        self.gamma = GammaCorrection(lambda_default)

    def forward(self, x):
        # PN: get illumination features and initial map
        fillu, l0 = self.pn(x)

        # DN: decompose into L_hat and R_hat
        l_hat, r_hat = self.dn(x, fillu, l0)

        # Gamma-correct illumination
        l_corrected = self.gamma(l_hat)

        # Reconstruct enhanced image: I_hat = L_hat ⊙ R_hat
        i_hat = l_corrected * r_hat
        i_hat = i_hat.clamp(0, 1)

        return {
            'enhanced' : i_hat,
            'l_hat'    : l_hat,
            'r_hat'    : r_hat,
            'l0'       : l0,
            'fillu'    : fillu,
        }
