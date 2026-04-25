"""
smoke_test.py  —  Verify all modules work before training
==========================================================
Run this first to confirm your environment is set up correctly:

    python smoke_test.py

Expected output: All checks PASS, no errors.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print()

PASSED = []
FAILED = []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f'  ✓  {name}')
    except Exception as e:
        FAILED.append(name)
        print(f'  ✗  {name}  →  {e}')


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
print('── Imports ──')

check('stage1_retinex', lambda: __import__('models.stage1_retinex', fromlist=['Stage1Net']))
check('dtcwt_transformer', lambda: __import__('models.dtcwt_transformer', fromlist=['MultiScaleSpectralLayer']))
check('spectrum_optimizer', lambda: __import__('models.spectrum_optimizer', fromlist=['SpectrumOptimizationNetwork']))
check('msretinexformer', lambda: __import__('models.msretinexformer', fromlist=['MSRetinexFormer']))
check('retinex_losses', lambda: __import__('losses.retinex_losses', fromlist=['MSRetinexLoss']))
check('metrics', lambda: __import__('utils.metrics', fromlist=['MetricEvaluator']))
check('datasets', lambda: __import__('data.datasets', fromlist=['SICEDataset']))

print()

# ---------------------------------------------------------------------------
# Component shapes (using small config to save memory)
# ---------------------------------------------------------------------------
from models.stage1_retinex      import Stage1Net, ProcessingNetwork, DecompositionNetwork, GammaCorrection
from models.dtcwt_transformer   import MultiScaleSpectralLayer, INFMixingModule
from models.spectrum_optimizer  import (SpectrumOptimizationNetwork,
                                        LFIR, HFIR, SpectralChannelMixing, TokenMixing)
from models.msretinexformer     import MSRetinexFormer
from losses.retinex_losses      import MSRetinexLoss, ReflectanceConsistencyLoss, RetinexDecompositionLoss
from utils.metrics              import MetricEvaluator, compute_psnr, compute_ssim, compute_mae

C, H = 32, 64   # small config
x    = torch.randn(1, 3, H, H).to(DEVICE)

print('── Stage 1 sub-modules ──')
check('ProcessingNetwork (PN)',
    lambda: ProcessingNetwork(3, C).to(DEVICE)(x))
check('GammaCorrection',
    lambda: GammaCorrection()(torch.rand(1,1,H,H).to(DEVICE)))
check('Stage1Net (PN + DN + Gamma)',
    lambda: Stage1Net(base_channels=C, num_heads=4, num_igab=2).to(DEVICE)(x))

print()
print('── Stage 2 sub-modules ──')
check('INFMixingModule (Sp Transformer)',
    lambda: INFMixingModule(C, J=1).to(DEVICE)(torch.randn(1,C,H//2,H//2).to(DEVICE)))
check('MultiScaleSpectralLayer (MSSL)',
    lambda: MultiScaleSpectralLayer(3, C, 4, J=1).to(DEVICE)(x))

print()
print('── Stage 3 sub-modules ──')
f = torch.randn(1, C, H//4, H//4).to(DEVICE)
check('LFIR  (Tensor Fusion, Eq.15)',    lambda: LFIR(C).to(DEVICE)(f))
check('HFIR  (Einstein Fusion, Eq.16)', lambda: HFIR(C, b_factor=4).to(DEVICE)(f))
check('SpectralChannelMixing (SCM)',     lambda: SpectralChannelMixing(C).to(DEVICE)(f))
check('TokenMixing (TM)',               lambda: TokenMixing(H//4).to(DEVICE)(f))
check('SpectrumOptimizationNetwork',
    lambda: SpectrumOptimizationNetwork(C, H//4, J=1).to(DEVICE)(f))

print()
print('── Full pipeline ──')
model = MSRetinexFormer(
    base_channels=C, embed_dim=C, patch_size=4, J=1,
    num_heads=4, num_igab=2, spatial_size=H//4
).to(DEVICE)

check('MSRetinexFormer.forward()',     lambda: model(x))
check('MSRetinexFormer.forward_pair()', lambda: model.forward_pair(x, x))

print()
print('── Losses ──')
crit = MSRetinexLoss(w0=500, w1=1).to(DEVICE)
check('ReflectanceConsistencyLoss (L_C)',
    lambda: ReflectanceConsistencyLoss()(torch.rand(1,3,H,H), torch.rand(1,3,H,H)))
check('RetinexDecompositionLoss (L_R)',
    lambda: RetinexDecompositionLoss()(
        torch.rand(1,3,H,H), torch.rand(1,1,H,H),
        torch.rand(1,1,H,H), torch.rand(1,3,H,H)
    ))
def _full_loss():
    o1, o2 = model.forward_pair(x, x)
    crit(o1, o2, x, x)
check('MSRetinexLoss (L_C + L_R)', _full_loss)

print()
print('── Backward pass (optimizer step) ──')
def _backward():
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    opt.zero_grad()
    o1, o2 = model.forward_pair(x, x)
    loss, _ = crit(o1, o2, x, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
check('Adam step + gradient clip', _backward)

print()
print('── Metrics ──')
p = torch.rand(1,3,H,H); r = torch.rand(1,3,H,H)
check('compute_psnr',  lambda: compute_psnr(p, r))
check('compute_ssim',  lambda: compute_ssim(p, r))
check('compute_mae',   lambda: compute_mae(p, r))
check('MetricEvaluator', lambda: MetricEvaluator().update(p, r))

print()
print('── LR Scheduler (Table 1) ──')
def _scheduler():
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100,200,300], gamma=0.5)
    for _ in range(5):
        sch.step()
check('MultiStepLR (decay every 100 epochs)', _scheduler)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print('=' * 55)
n_params = sum(p.numel() for p in model.parameters())
print(f'  Model parameters (full config): {n_params:,}')
print(f'  Passed : {len(PASSED)}/{len(PASSED)+len(FAILED)}')
if FAILED:
    print(f'  FAILED : {FAILED}')
    sys.exit(1)
else:
    print(f'  Status : ALL CHECKS PASSED ✓')
print('=' * 55)
