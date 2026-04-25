"""
Training Script  —  MSRetinexFormer Base Paper Implementation
=============================================================
Matches the experimental setup from Table 1:
  - Adam optimizer, lr = 1e-4, β1=0.9, β2=0.99
  - MultiStep LR decay, period=100 epochs, factor=0.5
  - 400 epochs total
  - Batch size = 1
  - Random crop 128×128
  - Best model saved per PSNR / SSIM / LPIPS

Usage:
    python train.py --dataset sice --data_root ./data/SICE \
                    --epochs 400 --save_dir ./checkpoints
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from models.msretinexformer import MSRetinexFormer
from losses.retinex_losses  import MSRetinexLoss
from data.datasets          import get_dataloader
from utils.metrics          import MetricEvaluator


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_file):
    logger = logging.getLogger('MSRetinex')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train MSRetinexFormer')

    # Data
    p.add_argument('--dataset',    type=str, default='sice',
                   choices=['sice', 'mit5k'], help='Dataset to train on')
    p.add_argument('--data_root',  type=str, default='./data/SICE',
                   help='Path to dataset root')
    p.add_argument('--val_root',   type=str, default=None,
                   help='Path to validation set (default: data_root/val)')
    p.add_argument('--crop_size',  type=int, default=128)
    p.add_argument('--num_workers',type=int, default=4)

    # Model
    p.add_argument('--base_channels', type=int, default=64)
    p.add_argument('--embed_dim',     type=int, default=64)
    p.add_argument('--patch_size',    type=int, default=4)
    p.add_argument('--J',             type=int, default=2,
                   help='DTCWT decomposition levels')
    p.add_argument('--num_heads',     type=int, default=8)
    p.add_argument('--num_igab',      type=int, default=4)
    p.add_argument('--spatial_size',  type=int, default=32,
                   help='Spatial size of SON feature maps')

    # Training
    p.add_argument('--epochs',    type=int,   default=400)
    p.add_argument('--batch_size',type=int,   default=1)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--beta1',     type=float, default=0.9)
    p.add_argument('--beta2',     type=float, default=0.99)
    p.add_argument('--decay_step',type=int,   default=100,
                   help='LR decay period in epochs')
    p.add_argument('--decay_gamma',type=float,default=0.5)
    p.add_argument('--lambda_correction', type=float, default=0.2,
                   help='Gamma correction factor λ (0.14 for very dark scenes)')

    # Loss weights (Table 1)
    p.add_argument('--w0', type=float, default=500.0, help='Weight for L_C')
    p.add_argument('--w1', type=float, default=1.0,   help='Weight for L_R')

    # Checkpointing
    p.add_argument('--save_dir',  type=str, default='./checkpoints')
    p.add_argument('--resume',    type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--val_every', type=int, default=10,
                   help='Validate every N epochs')

    # Device
    p.add_argument('--device', type=str, default='',
                   help='cuda / cpu (auto-detect if empty)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, logger, epoch):
    model.train()
    total_loss = 0.0
    lc_acc = lr_acc = 0.0
    t0 = time.time()

    for step, (x1, x2, _ref) in enumerate(loader):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad()

        out1, out2 = model.forward_pair(x1, x2)
        loss, ld   = criterion(out1, out2, x1, x2)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += ld['total']
        lc_acc     += ld['L_C']
        lr_acc     += ld['L_R']

        if (step + 1) % 100 == 0:
            logger.info(
                f'Epoch [{epoch}] Step [{step+1}/{len(loader)}] '
                f'Loss={ld["total"]:.4f} L_C={ld["L_C"]:.6f} '
                f'L_R={ld["L_R"]:.4f}'
            )

    n = len(loader)
    elapsed = time.time() - t0
    logger.info(
        f'Epoch [{epoch}] TRAIN | '
        f'AvgLoss={total_loss/n:.4f} | '
        f'L_C={lc_acc/n:.6f} | L_R={lr_acc/n:.4f} | '
        f'Time={elapsed:.1f}s'
    )
    return total_loss / n


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, logger, epoch):
    model.eval()
    evaluator = MetricEvaluator()

    for x1, x2, ref in loader:
        x1  = x1.to(device)
        ref = ref.to(device)
        out = model(x1)
        evaluator.update(out['enhanced'].cpu(), ref.cpu())

    metrics = evaluator.compute()
    logger.info(
        f'Epoch [{epoch}] VAL | '
        f'PSNR={metrics["PSNR"]:.2f} | SSIM={metrics["SSIM"]:.4f} | '
        f'LPIPS={metrics["LPIPS"]:.4f} | MAE={metrics["MAE"]:.2f}'
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(save_dir / 'train.log')
    logger.info(f'Args: {vars(args)}')

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Data
    train_loader, _ = get_dataloader(
        args.dataset, args.data_root,
        batch_size  = args.batch_size,
        crop_size   = args.crop_size,
        is_train    = True,
        num_workers = args.num_workers,
    )
    val_root = args.val_root or str(Path(args.data_root) / 'val')
    val_loader, _ = get_dataloader(
        args.dataset, val_root,
        batch_size  = 1,
        crop_size   = args.crop_size,
        is_train    = False,
        num_workers = args.num_workers,
    )
    logger.info(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    # Model
    model = MSRetinexFormer(
        base_channels  = args.base_channels,
        embed_dim      = args.embed_dim,
        patch_size     = args.patch_size,
        J              = args.J,
        num_heads      = args.num_heads,
        num_igab       = args.num_igab,
        spatial_size   = args.spatial_size,
        lambda_default = args.lambda_correction,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {n_params:,}')

    # Loss
    criterion = MSRetinexLoss(w0=args.w0, w1=args.w1).to(device)

    # Optimizer  (Adam, lr=1e-4, β1=0.9, β2=0.99 per Table 1)
    optimizer = Adam(model.parameters(), lr=args.lr,
                     betas=(args.beta1, args.beta2))

    # LR scheduler: decay by 0.5 every 100 epochs (MultiStep)
    milestones = list(range(args.decay_step, args.epochs, args.decay_step))
    scheduler  = MultiStepLR(optimizer, milestones=milestones,
                              gamma=args.decay_gamma)

    # Resume
    start_epoch   = 1
    best_psnr     = 0.0
    best_ssim     = 0.0
    best_lpips    = float('inf')

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', 0.0)
        logger.info(f'Resumed from epoch {ckpt["epoch"]}')

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger, epoch
        )
        scheduler.step()

        # Save periodic checkpoint
        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics = validate(model, val_loader, device, logger, epoch)

            # Save best models  (paper saves best per PSNR, SSIM, LPIPS)
            def _save(tag):
                path = save_dir / f'best_{tag}.pth'
                torch.save({
                    'epoch'    : epoch,
                    'model'    : model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'metrics'  : metrics,
                    'best_psnr': best_psnr,
                }, path)
                logger.info(f'Saved best_{tag} checkpoint (epoch {epoch})')

            if metrics['PSNR']  > best_psnr:
                best_psnr  = metrics['PSNR'];   _save('psnr')
            if metrics['SSIM']  > best_ssim:
                best_ssim  = metrics['SSIM'];   _save('ssim')
            if metrics['LPIPS'] < best_lpips:
                best_lpips = metrics['LPIPS'];  _save('lpips')

        # Always save latest checkpoint
        torch.save({
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
        }, save_dir / 'latest.pth')

    logger.info('Training complete.')
    logger.info(f'Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
