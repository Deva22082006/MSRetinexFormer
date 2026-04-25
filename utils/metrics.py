"""
Evaluation Metrics  (Section 4.2)
===================================
PSNR  : Peak Signal-to-Noise Ratio      (higher is better ↑)
SSIM  : Structural Similarity Index     (higher is better ↑)
LPIPS : Learned Perceptual Image Patch  (lower is better ↓)
MAE   : Mean Absolute Error             (lower is better ↓)
"""

import torch
import torch.nn.functional as F
import numpy as np
from math import log10


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(pred, target, max_val=1.0):
    """
    Compute PSNR between predicted and target images.

    Args:
        pred, target : tensors [B, C, H, W] or [C, H, W], values in [0, max_val]
    Returns:
        mean PSNR (dB) over the batch
    """
    if pred.dim() == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])                       # per-image MSE
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-10))
    return psnr.mean().item()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size=11, sigma=1.5):
    """1-D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _create_window(window_size, channel):
    _1d = _gaussian_kernel(window_size).unsqueeze(1)
    _2d = _1d @ _1d.T
    window = _2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, -1, -1)
    return window.contiguous()


def compute_ssim(pred, target, window_size=11, max_val=1.0):
    """
    Differentiable SSIM computation.

    Args:
        pred, target : [B, C, H, W] or [C, H, W]
    Returns:
        mean SSIM over batch
    """
    if pred.dim() == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    B, C, H, W = pred.shape
    window     = _create_window(window_size, C).to(pred.device).to(pred.dtype)
    padding    = window_size // 2

    mu1    = F.conv2d(pred,   window, padding=padding, groups=C)
    mu2    = F.conv2d(target, window, padding=padding, groups=C)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2= mu1 * mu2

    s1  = F.conv2d(pred   * pred,   window, padding=padding, groups=C) - mu1_sq
    s2  = F.conv2d(target * target, window, padding=padding, groups=C) - mu2_sq
    s12 = F.conv2d(pred   * target, window, padding=padding, groups=C) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

def compute_mae(pred, target, max_val=1.0):
    """
    Mean Absolute Error (scaled to [0, 255] to match paper reporting).

    Args:
        pred, target : [B, C, H, W] or [C, H, W], values in [0, max_val]
    Returns:
        MAE in [0, 255] scale
    """
    if pred.dim() == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    return (F.l1_loss(pred, target) * 255.0).item()


# ---------------------------------------------------------------------------
# LPIPS  (lightweight VGG-based approximation if lpips package absent)
# ---------------------------------------------------------------------------

try:
    import lpips as _lpips_lib
    _lpips_net = None

    def compute_lpips(pred, target):
        """
        LPIPS using official lpips library (alex backbone).
        """
        global _lpips_net
        if _lpips_net is None:
            _lpips_net = _lpips_lib.LPIPS(net='alex').to(pred.device)
            _lpips_net.eval()
        if pred.dim() == 3:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        # lpips expects [-1, 1]
        pred_n   = pred   * 2 - 1
        target_n = target * 2 - 1
        with torch.no_grad():
            val = _lpips_net(pred_n, target_n)
        return val.mean().item()

except ImportError:
    def compute_lpips(pred, target):
        """
        Lightweight LPIPS proxy using VGG features from torchvision.
        (Used when the official 'lpips' package is not installed.)
        """
        import torchvision.models as models

        if not hasattr(compute_lpips, '_feat'):
            vgg = models.vgg16(weights=None).features[:16]
            for p in vgg.parameters():
                p.requires_grad_(False)
            compute_lpips._feat = vgg.eval()

        feat = compute_lpips._feat.to(pred.device)
        if pred.dim() == 3:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        with torch.no_grad():
            f1 = feat(pred   * 2 - 1)
            f2 = feat(target * 2 - 1)
        diff = (f1 - f2).pow(2).mean()
        return diff.item()


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------

class MetricEvaluator:
    """
    Accumulates metrics over a validation set and returns averages.

    Usage:
        evaluator = MetricEvaluator()
        for pred, ref in val_loader:
            evaluator.update(pred, ref)
        results = evaluator.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._psnr  = []
        self._ssim  = []
        self._lpips = []
        self._mae   = []

    def update(self, pred, target):
        """
        Args:
            pred, target : [B, C, H, W] tensors in [0, 1]
        """
        for i in range(pred.shape[0]):
            p = pred[i:i+1]
            t = target[i:i+1]
            self._psnr .append(compute_psnr (p, t))
            self._ssim .append(compute_ssim (p, t))
            self._lpips.append(compute_lpips(p, t))
            self._mae  .append(compute_mae  (p, t))

    def compute(self):
        import numpy as np
        return {
            'PSNR' : float(np.mean(self._psnr)),
            'SSIM' : float(np.mean(self._ssim)),
            'LPIPS': float(np.mean(self._lpips)),
            'MAE'  : float(np.mean(self._mae)),
        }
