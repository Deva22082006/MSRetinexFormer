# MSRetinexFormer — Base Paper Implementation

**Paper**: *Low-light image enhancement method based on Retinex theory and dual-tree complex wavelet transform*  
Journal of King Saud University – Computer and Information Sciences (2025) 37:83  
https://doi.org/10.1007/s44443-025-00102-6

---

## Architecture Overview

The model is a **3-stage pipeline**. Every block in the paper's figures has a
corresponding Python module:

```
Input (low-light image)
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 1 — Illumination Estimation & Retinex Decomposition    │
│  (Fig. 2)                                                     │
│                                                               │
│  PN (Processing Network)                                      │
│    models/stage1_retinex.py :: ProcessingNetwork              │
│    → outputs F_illu (illumination features) + L0 (init map)  │
│                                                               │
│  DN (Decomposition Network)                                   │
│    models/stage1_retinex.py :: DecompositionNetwork           │
│    → Transformer encoder-decoder with IGAB blocks             │
│    → outputs L_hat (illumination) + R_hat (reflectance)       │
│                                                               │
│  Gamma Correcting                                             │
│    models/stage1_retinex.py :: GammaCorrection                │
│    → I_hat = GammaCorrect(L_hat) ⊙ R_hat                     │
└───────────────────────────────────────────────────────────────┘
    │  F_illu (passed to Stage 2 as illumination injection)
    │  I_hat_1 (Stage-1 enhanced image)
    ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 2 — Multi-Scale Frequency Decomposition (DTCWT)        │
│  (Fig. 3)                                                     │
│                                                               │
│  Convolution Stage                                            │
│    models/dtcwt_transformer.py :: MultiScaleSpectralLayer     │
│    → PatchEmbed + PositionEncode + ResBlocks                  │
│                                                               │
│  Sp Transformer × 4  (2 on LF path, 2 on HF path)            │
│    models/dtcwt_transformer.py :: INFMixingModule             │
│    → DTCWT decompose → spectral modulation → IDTCWT           │
│    → F_fused = W_real·Re(X) + W_imag·Im(X)  (Eq. 17)        │
│    → Illumination F_illu injected as additive bias            │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 3 — Spectrum Optimization & Reconstruction             │
│  (Fig. 4)                                                     │
│                                                               │
│  MSSL → LF branch                                             │
│    TBM (Token-Based Mixing)                                   │
│      models/spectrum_optimizer.py :: TokenMixing              │
│    LFR (Low-Frequency Refinement) via Tensor Fusion           │
│      models/spectrum_optimizer.py :: LFIR                     │
│      M_φ = X_φ ⊙ W_φ  (Eq. 15, Hadamard product)            │
│                                                               │
│  MSSL → HF branch                                             │
│    EBM (Embedding-Based Mixing / Spectral Channel Mixing)     │
│      models/spectrum_optimizer.py :: SpectralChannelMixing    │
│    HFR (High-Frequency Refinement) via Einstein Fusion        │
│      models/spectrum_optimizer.py :: HFIR                     │
│      S_ψ = X_ψ ⊗ W_ψ  (Eq. 16, Einstein product)            │
│                                                               │
│  SON (Spectrum Optimization Network)                          │
│    models/spectrum_optimizer.py :: SpectrumOptimizationNetwork│
│    → Combines LF + HF → IDTCWT → Final reconstruction        │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
Final Enhanced Image  (α·Stage1 + (1-α)·Stage3)
```

---

## Project Structure

```
MSRetinexFormer/
├── models/
│   ├── msretinexformer.py       ← Full 3-stage model (main entry point)
│   ├── stage1_retinex.py        ← Stage 1: PN + DN + Gamma Correct (Fig. 2)
│   ├── dtcwt_transformer.py     ← Stage 2: MSSL + INF Mixing (Fig. 3)
│   └── spectrum_optimizer.py   ← Stage 3: SON/LFIR/HFIR/SCM/TM (Fig. 4)
├── losses/
│   └── retinex_losses.py        ← L_C (Eq. 18) + L_R (Eq. 19)
├── data/
│   └── datasets.py              ← SICE + MIT5K dataset loaders
├── utils/
│   └── metrics.py               ← PSNR, SSIM, LPIPS, MAE
├── train.py                     ← Training script
├── test.py                      ← Evaluation script (Table 2 metrics)
└── infer.py                     ← Single-image inference demo
```

---

## Installation

```bash
pip install torch torchvision pytorch_wavelets einops timm opt_einsum lpips
```

### Dataset Preparation

**SICE:**
```
data/SICE/
    train/
        scene_001/
            1.jpg   ← underexposed image 1
            2.jpg   ← underexposed image 2
            ref.jpg ← ground truth
        scene_002/ ...
    val/
        scene_229/ ...
```

**MIT Adobe FiveK:**
```
data/MIT5K/
    train/
        input/  *.jpg
        ref/    *.jpg   (Expert-C retouched)
    val/
        input/  *.jpg
        ref/    *.jpg
```

---

## Training

```bash
# Train on SICE (matches Table 1 exactly)
python train.py \
    --dataset sice \
    --data_root ./data/SICE/train \
    --val_root  ./data/SICE/val \
    --epochs 400 \
    --batch_size 1 \
    --lr 1e-4 \
    --decay_step 100 \
    --decay_gamma 0.5 \
    --w0 500 --w1 1 \
    --crop_size 128 \
    --save_dir ./checkpoints/sice

# For very dark scenes (LOL dataset), set lambda=0.14
python train.py --lambda_correction 0.14 ...

# Train on MIT5K
python train.py \
    --dataset mit5k \
    --data_root ./data/MIT5K/train \
    --val_root  ./data/MIT5K/val \
    --save_dir  ./checkpoints/mit5k
```

---

## Evaluation

```bash
# Evaluate on MIT5K (expects results matching Table 2 in paper)
python test.py \
    --checkpoint ./checkpoints/mit5k/best_psnr.pth \
    --dataset    mit5k \
    --data_root  ./data/MIT5K/val \
    --save_images ./results/mit5k

# Expected output (paper Table 2, MIT column):
# PSNR=20.26  SSIM=0.655  LPIPS=0.221  MAE=21.29
```

---

## Key Equations from Paper

| Equation | Description | Code Location |
|----------|-------------|---------------|
| Eq. 1  | `I = L ⊙ R` (Retinex decomposition) | `stage1_retinex.py` |
| Eq. 10 | `X = FT(I) + FP(I)` (token+pos embed) | `dtcwt_transformer.py` |
| Eq. 12 | `XF = Xφ + Xψ` (LF + HF decompose) | `dtcwt_transformer.py` |
| Eq. 15 | `Mφ = Xφ ⊙ Wφ` (Tensor Fusion, LFIR) | `spectrum_optimizer.py :: LFIR` |
| Eq. 16 | `Sψ = Xψ ⊗ Wψ` (Einstein Fusion, HFIR) | `spectrum_optimizer.py :: HFIR` |
| Eq. 17 | `Ffused = Wreal·Re(X) + Wimag·Im(X)` | `dtcwt_transformer.py :: INFMixingModule` |
| Eq. 18 | `LC = ‖R1 - R2‖²` (Reflectance Consistency) | `losses/retinex_losses.py` |
| Eq. 19 | `LR = ‖R⊙L - i‖² + ...` (Retinex Decomp.) | `losses/retinex_losses.py` |

---

## Hyperparameters (Table 1)

| Parameter | Value |
|-----------|-------|
| Batch Size | 1 |
| Learning Rate | 1×10⁻⁴ |
| LR Decay Strategy | Multi-step |
| LR Decay Period | 100 epochs |
| LR Decay Factor | 0.5 |
| Training Epochs | 400 |
| Correction Factor λ (default) | 0.2 |
| Correction Factor λ (dark) | 0.14 |
| w0 (L_C weight) | 500 |
| w1, w2 | 1 |

---

## Results (Table 2)

| Dataset | PSNR↑ | SSIM↑ | LPIPS↓ | MAE↓ |
|---------|-------|-------|--------|------|
| SICE    | 19.21 | 0.714 | 0.193  | 22.94 |
| MIT5K   | 20.26 | 0.655 | 0.221  | 21.29 |
