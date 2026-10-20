#!/usr/bin/env python3
"""Score saved UDIS++ stitches.

Sparse datasets use histogram-matched scores on the sparse GT mask.
WideDrive3 has dense GT, so it uses direct full-image PSNR, SSIM, MAE, RMSE,
and AlexNet LPIPS. No histogram matching.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

ROOT = Path(__file__).resolve().parent
_LEGACY = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub/datasets")
WORK = _LEGACY if _LEGACY.exists() else ROOT.parent / "datasets"
SPECS = {
    "nuscenes": (WORK / "nuscenes/sparseWideFOVImages3_1554x294", 5, False),
    "lyft1920": (WORK / "lyft/1920_sparseWideFOVImages3_1554x294", 5, False),
    "lyft1224": (WORK / "lyft/1224_sparseWideFOVImages3_1554x434", 5, False),
    "widedrive3": (WORK / "WideDrive_processed/sparseWideFOVImages3_1554x294", 2, True),
}


def find_gt(folder, frame, cam):
    suffix = f"_{cam}_sparse_wide.png"
    rgb = folder / "rgb" / f"{frame}{suffix}"
    mask = folder / "mask" / f"{frame}{suffix}"
    return (rgb, mask) if rgb.is_file() and mask.is_file() else (None, None)


def resize_to(image, size, nearest=False):
    mode = Image.NEAREST if nearest else Image.BILINEAR
    return np.array(Image.fromarray(image).resize(size, mode))


def score(render, gt, mask, dense):
    if dense:
        mask = np.ones(render.shape[:2], dtype=bool)
        matched = render[mask]
    else:
        matched = match_histograms(render[mask][np.newaxis], gt[mask][np.newaxis], channel_axis=-1)[0]
    target = gt[mask]
    diff = matched - target
    mse = float(np.mean(diff ** 2))
    c1 = 0.01 ** 2
    return {
        "psnr": float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
        "ssim": float(np.mean((2 * matched * target + c1) / (matched ** 2 + target ** 2 + c1))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
    }


def lpips_score(render, gt):
    import torch
    import lpips
    model = lpips.LPIPS(net="alex").cuda()
    tensor = lambda image: torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).cuda() * 2 - 1
    with torch.no_grad():
        return float(model(tensor(render), tensor(gt)))


def main():
    parser = argparse.ArgumentParser(description="Score UDIS++ driving stitches")
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--pred", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    gt_root, cam, dense = SPECS[args.dataset]
    pred_root = Path(args.pred or ROOT / "outputs" / args.dataset)
    rows = []
    for scene_dir in sorted(path for path in pred_root.iterdir() if path.is_dir()):
        crop = scene_dir / "pairwise"
        if not crop.is_dir():
            continue
        names = sorted(crop.glob("*.png"))
        if args.limit:
            names = names[:args.limit]
        for path in names:
            gt_path, mask_path = find_gt(gt_root / scene_dir.name, path.stem, cam)
            if gt_path is None:
                continue
            render = np.array(Image.open(path).convert("RGB")).astype(np.float64) / 255
            gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float64) / 255
            mask = np.array(Image.open(mask_path).convert("L")) > 127
            if gt.shape[:2] != render.shape[:2]:
                size = (render.shape[1], render.shape[0])
                gt = resize_to((gt * 255).astype(np.uint8), size).astype(np.float64) / 255
                mask = resize_to((mask.astype(np.uint8) * 255), size, nearest=True) > 127
            row = score(render, gt, mask, dense)
            if dense:
                row["lpips"] = lpips_score(render, gt)
            rows.append(row)
            print(path.name, row, flush=True)
    if not rows:
        raise SystemExit("no scored frames")
    keys = rows[0]
    print("mean", {key: float(np.mean([row[key] for row in rows])) for key in keys}, flush=True)


if __name__ == "__main__":
    main()
