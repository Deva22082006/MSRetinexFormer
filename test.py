"""
Inference / Test Script
========================
Evaluate a trained MSRetinexFormer checkpoint on SICE or MIT5K test sets
and produce a results table matching Table 2 of the paper.

Usage:
    # Evaluate on MIT5K test set
    python test.py --checkpoint ./checkpoints/best_psnr.pth \
                   --dataset mit5k --data_root ./data/MIT5K/val \
                   --save_images ./results/mit5k

    # Evaluate on SICE
    python test.py --checkpoint ./checkpoints/best_psnr.pth \
                   --dataset sice --data_root ./data/SICE/val \
                   --save_images ./results/sice
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.msretinexformer import MSRetinexFormer
from data.datasets          import get_dataloader
from utils.metrics          import MetricEvaluator


def parse_args():
    p = argparse.ArgumentParser(description='Test MSRetinexFormer')
    p.add_argument('--checkpoint',   type=str, required=True)
    p.add_argument('--dataset',      type=str, default='sice',
                   choices=['sice', 'mit5k'])
    p.add_argument('--data_root',    type=str, required=True)
    p.add_argument('--save_images',  type=str, default=None,
                   help='Directory to save enhanced images (skip if None)')
    p.add_argument('--device',       type=str, default='')
    # Model config — must match the trained checkpoint
    p.add_argument('--base_channels', type=int, default=64)
    p.add_argument('--embed_dim',     type=int, default=64)
    p.add_argument('--patch_size',    type=int, default=4)
    p.add_argument('--J',             type=int, default=2)
    p.add_argument('--num_heads',     type=int, default=8)
    p.add_argument('--num_igab',      type=int, default=4)
    p.add_argument('--spatial_size',  type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'Device: {device}')

    # Load model
    model = MSRetinexFormer(
        base_channels = args.base_channels,
        embed_dim     = args.embed_dim,
        patch_size    = args.patch_size,
        J             = args.J,
        num_heads     = args.num_heads,
        num_igab      = args.num_igab,
        spatial_size  = args.spatial_size,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint (epoch {ckpt.get("epoch", "?")})')

    # Data
    loader, dataset = get_dataloader(
        args.dataset, args.data_root,
        batch_size=1, crop_size=256,    # larger crop for evaluation
        is_train=False, num_workers=2,
    )
    print(f'Test images: {len(dataset)}')

    # Save dir
    if args.save_images:
        Path(args.save_images).mkdir(parents=True, exist_ok=True)

    evaluator = MetricEvaluator()

    for idx, (x1, _x2, ref) in enumerate(tqdm(loader, desc='Evaluating')):
        x1  = x1.to(device)
        ref = ref.to(device)
        out = model(x1)
        enhanced = out['enhanced']

        evaluator.update(enhanced.cpu(), ref.cpu())

        if args.save_images:
            # Save side-by-side: input | enhanced | reference
            grid = torch.cat([x1, enhanced, ref], dim=3)   # concat width
            path = Path(args.save_images) / f'{idx:04d}.png'
            vutils.save_image(grid, path)

    metrics = evaluator.compute()
    print('\n' + '='*55)
    print(f'  Dataset : {args.dataset.upper()}')
    print(f'  {"PSNR":>8} {"SSIM":>8} {"LPIPS":>8} {"MAE":>8}')
    print(f'  {"↑":>8} {"↑":>8} {"↓":>8} {"↓":>8}')
    print('-'*55)
    print(
        f'  {metrics["PSNR"]:>8.2f} '
        f'{metrics["SSIM"]:>8.4f} '
        f'{metrics["LPIPS"]:>8.4f} '
        f'{metrics["MAE"]:>8.2f}'
    )
    print('='*55)

    return metrics


if __name__ == '__main__':
    main()
