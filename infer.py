"""
Single-Image Inference Demo
============================
Run MSRetinexFormer on one image and save the result.

Usage:
    python infer.py --image ./dark_photo.jpg \
                    --checkpoint ./checkpoints/best_psnr.pth \
                    --output ./enhanced.png
"""

import sys
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from models.msretinexformer import MSRetinexFormer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image',      type=str, required=True)
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Optional checkpoint; runs random weights if not given')
    p.add_argument('--output',     type=str, default='enhanced.png')
    p.add_argument('--lambda_correction', type=float, default=0.2)
    p.add_argument('--device',     type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    model = MSRetinexFormer(lambda_default=args.lambda_correction).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'Loaded checkpoint from {args.checkpoint}')
    else:
        print('No checkpoint provided — running with random weights (demo only)')

    model.eval()

    # Load image
    img  = Image.open(args.image).convert('RGB')
    orig_size = img.size        # (W, H)

    x = T.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    enhanced = out['enhanced'][0].cpu().clamp(0, 1)
    result   = TF.to_pil_image(enhanced)
    result   = result.resize(orig_size, Image.LANCZOS)
    result.save(args.output)
    print(f'Enhanced image saved to: {args.output}')

    # Also print intermediate stats
    l_mean = out['l_hat'].mean().item()
    r_mean = out['r_hat'].mean().item()
    print(f'  L_hat mean (illumination): {l_mean:.4f}')
    print(f'  R_hat mean (reflectance) : {r_mean:.4f}')


if __name__ == '__main__':
    main()
