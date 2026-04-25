"""
Loss Functions  (Section 3.5)
==============================
Two losses based on Retinex theory:

    L_C : Reflectance Consistency Loss   (Eq. 18)
    L_R : Retinex Decomposition Loss     (Eq. 19)

Total loss = w0 * L_C + w1 * L_R  (w0=500, w1=1 per Table 1)

Additionally a perceptual/SSIM loss can be included for completeness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient helper for illumination smoothness (∇L term in Eq. 19)
# ---------------------------------------------------------------------------

def _total_variation(x):
    """Anisotropic total variation: ‖∇L‖₁"""
    diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    return diff_h.mean() + diff_w.mean()


# ---------------------------------------------------------------------------
# L_C : Reflectance Consistency Loss  (Eq. 18)
# ---------------------------------------------------------------------------

class ReflectanceConsistencyLoss(nn.Module):
    """
    L_C = ‖R1 - R2‖²₂

    Enforces that reflectance maps predicted from two images of the same
    scene under different illumination should be identical (the reflectance
    is an intrinsic property, independent of lighting).

    In unsupervised training with paired low-light images:
        R1 = reflectance from image pair I1
        R2 = reflectance from image pair I2
    """

    def __init__(self):
        super().__init__()

    def forward(self, r1, r2):
        """
        Args:
            r1, r2 : reflectance maps  [B, 3, H, W]
        Returns:
            scalar loss
        """
        return F.mse_loss(r1, r2)


# ---------------------------------------------------------------------------
# L_R : Retinex Decomposition Loss  (Eq. 19)
# ---------------------------------------------------------------------------

class RetinexDecompositionLoss(nn.Module):
    """
    L_R = ‖R ⊙ L - i‖²₂                   (reconstruction fidelity)
        + ‖R - i / stopgrad(L)‖²₂          (reflectance guidance)
        + ‖L - L0‖²₂                       (illumination prior)
        + ‖∇L‖₁                            (illumination smoothness)

    Args:
        lambda_smooth : weight for TV term (default 0.1)
    """

    def __init__(self, lambda_smooth=0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, r, l, l0, i):
        """
        Args:
            r  : reflectance map     [B, 3, H, W]
            l  : illumination map    [B, 1, H, W]
            l0 : initial illum map   [B, 1, H, W]
            i  : input image         [B, 3, H, W]
        Returns:
            scalar loss
        """
        # --- Term 1: reconstruction fidelity  ‖R ⊙ L - i‖² ---
        recon = r * l                           # [B, 3, H, W]
        t1 = F.mse_loss(recon, i)

        # --- Term 2: reflectance guidance  ‖R - i/stop_grad(L)‖² ---
        l_sg = l.detach().clamp(min=1e-6)       # stop gradient through L
        r_target = i / l_sg                     # [B, 3, H, W]
        t2 = F.mse_loss(r, r_target.clamp(0, 1))

        # --- Term 3: illumination prior  ‖L - L0‖² ---
        t3 = F.mse_loss(l, l0)

        # --- Term 4: illumination smoothness  ‖∇L‖₁ ---
        t4 = _total_variation(l)

        return t1 + t2 + t3 + self.lambda_smooth * t4


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class MSRetinexLoss(nn.Module):
    """
    Full training loss combining L_C and L_R.

        L_total = w0 * L_C + w1 * L_R

    Default weights from Table 1: w0 = 500, w1 = 1.
    w2 can be used to weight an additional perceptual term.
    """

    def __init__(self, w0=500.0, w1=1.0, w2=0.0, lambda_smooth=0.1):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.lc = ReflectanceConsistencyLoss()
        self.lr = RetinexDecompositionLoss(lambda_smooth)

    def forward(self, out1, out2, i1, i2):
        """
        Args:
            out1, out2 : dicts from Stage1Net.forward() for paired images I1/I2
            i1, i2     : original input images [B, 3, H, W]
        Returns:
            total_loss  : scalar
            loss_dict   : dict of individual loss values for logging
        """
        # L_C: reflectance should be identical for the scene pair
        lc_val = self.lc(out1['r_hat'], out2['r_hat'])

        # L_R for each image in the pair
        lr_val1 = self.lr(out1['r_hat'], out1['l_hat'], out1['l0'], i1)
        lr_val2 = self.lr(out2['r_hat'], out2['l_hat'], out2['l0'], i2)
        lr_val  = (lr_val1 + lr_val2) * 0.5

        total = self.w0 * lc_val + self.w1 * lr_val

        return total, {
            'L_C'   : lc_val.item(),
            'L_R'   : lr_val.item(),
            'total' : total.item(),
        }
