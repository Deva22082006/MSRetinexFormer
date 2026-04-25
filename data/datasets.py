"""
Dataset Loaders
===============
Supports:
  - SICE Dataset   (paired underexposed images)
  - MIT Adobe FiveK Dataset (input + expert-C reference)

Both return paired images (x1, x2, ref) where x1/x2 are two underexposed
variants of the same scene (for reflectance consistency training) and ref
is the ground truth reference.

Directory structure expected:
    SICE/
        train/
            pair1/
                1.jpg   <- underexposed image 1
                2.jpg   <- underexposed image 2 (different exposure)
                ref.jpg <- reference (ground truth)
            pair2/ ...
        val/
            ...

    MIT5K/
        train/
            input/   *.jpg
            ref/     *.jpg   (expert-C retouched)
        val/ ...
"""

import os
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Paired augmentation (same crop / flip for both images)
# ---------------------------------------------------------------------------

def paired_random_crop(img1, img2, ref, crop_size=128):
    """Apply the same random crop to all three images."""
    i, j, h, w = T.RandomCrop.get_params(img1, output_size=(crop_size, crop_size))
    return (TF.crop(img1, i, j, h, w),
            TF.crop(img2, i, j, h, w),
            TF.crop(ref,  i, j, h, w))


def paired_augment(img1, img2, ref):
    """Random horizontal/vertical flip (same for all)."""
    if random.random() > 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)
        ref  = TF.hflip(ref)
    if random.random() > 0.5:
        img1 = TF.vflip(img1)
        img2 = TF.vflip(img2)
        ref  = TF.vflip(ref)
    return img1, img2, ref


to_tensor = T.ToTensor()


# ---------------------------------------------------------------------------
# SICE Dataset
# ---------------------------------------------------------------------------

class SICEDataset(Dataset):
    """
    SICE multi-exposure dataset.

    Each scene folder contains multiple exposures (sorted by filename).
    We pick two underexposed images as the training pair (x1, x2) and
    the reference (brightest or provided ref) as ground truth.

    Args:
        root      : path to SICE split (e.g. 'SICE/train')
        crop_size : random crop size during training
        is_train  : whether to apply data augmentation
    """

    def __init__(self, root, crop_size=128, is_train=True):
        super().__init__()
        self.root      = Path(root)
        self.crop_size = crop_size
        self.is_train  = is_train
        self.samples   = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            imgs = sorted([
                p for p in scene_dir.iterdir()
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
                   and 'ref' not in p.stem.lower()
            ])
            ref_candidates = [
                p for p in scene_dir.iterdir()
                if 'ref' in p.stem.lower()
            ]
            if len(imgs) < 2:
                continue
            ref = ref_candidates[0] if ref_candidates else imgs[-1]
            # Use first two underexposed images as the pair
            samples.append((imgs[0], imgs[1], ref))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, pr = self.samples[idx]
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')
        ref  = Image.open(pr).convert('RGB')

        if self.is_train:
            img1, img2, ref = paired_random_crop(img1, img2, ref, self.crop_size)
            img1, img2, ref = paired_augment(img1, img2, ref)

        return to_tensor(img1), to_tensor(img2), to_tensor(ref)


# ---------------------------------------------------------------------------
# MIT Adobe FiveK Dataset
# ---------------------------------------------------------------------------

class MIT5KDataset(Dataset):
    """
    MIT Adobe FiveK dataset.

    Single-image enhancement: input → expert-C retouched reference.
    For reflectance consistency, we create synthetic pairs by applying
    random gamma darkening to the same input (self-supervised augmentation).

    Args:
        root      : path to split (e.g. 'MIT5K/train')
        crop_size : random crop size
        is_train  : apply augmentation
    """

    def __init__(self, root, crop_size=128, is_train=True):
        super().__init__()
        self.root      = Path(root)
        self.crop_size = crop_size
        self.is_train  = is_train
        self.samples   = self._collect_samples()

    def _collect_samples(self):
        input_dir = self.root / 'input'
        ref_dir   = self.root / 'ref'
        if not input_dir.exists():
            # Flat structure: all files in root, no sub-split
            exts = ('.jpg', '.jpeg', '.png')
            files = sorted([p for p in self.root.iterdir()
                            if p.suffix.lower() in exts])
            return [(f, f) for f in files]  # ref = input (unsupervised)

        input_files = sorted(input_dir.glob('*'))
        samples = []
        for inp in input_files:
            ref = ref_dir / inp.name
            if not ref.exists():
                # Try common extension swaps
                for ext in ('.jpg', '.jpeg', '.png'):
                    alt = ref_dir / (inp.stem + ext)
                    if alt.exists():
                        ref = alt
                        break
            if ref.exists():
                samples.append((inp, ref))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_inp, p_ref = self.samples[idx]
        inp = Image.open(p_inp).convert('RGB')
        ref = Image.open(p_ref).convert('RGB')

        if self.is_train:
            # Paired crop on (inp, ref)
            i, j, h, w = T.RandomCrop.get_params(inp, (self.crop_size, self.crop_size))
            inp = TF.crop(inp, i, j, h, w)
            ref = TF.crop(ref, i, j, h, w)
            # Flip
            if random.random() > 0.5:
                inp = TF.hflip(inp);  ref = TF.hflip(ref)
            if random.random() > 0.5:
                inp = TF.vflip(inp);  ref = TF.vflip(ref)

        inp_t = to_tensor(inp)
        ref_t = to_tensor(ref)

        # Create synthetic pair: apply random gamma darkening to inp_t
        gamma = random.uniform(0.4, 0.9)
        inp2_t = inp_t ** (1.0 / gamma)          # slightly different exposure

        return inp_t, inp2_t, ref_t


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(dataset_name, root, batch_size=1, crop_size=128,
                   is_train=True, num_workers=4):
    """
    Convenience factory.

    Args:
        dataset_name : 'sice' or 'mit5k'
        root         : dataset root path
        batch_size   : training batch size (paper uses 1)
        crop_size    : random crop size (paper: 128×128)
        is_train     : train vs validation mode
        num_workers  : DataLoader workers
    """
    ds_map = {'sice': SICEDataset, 'mit5k': MIT5KDataset}
    assert dataset_name.lower() in ds_map, \
        f"dataset_name must be one of {list(ds_map.keys())}"

    dataset = ds_map[dataset_name.lower()](root, crop_size, is_train)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = is_train,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = is_train,
    )
    return loader, dataset
