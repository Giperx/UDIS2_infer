#!/usr/bin/env python3
"""Finetune UDIS++ warp, then composition, on driving camera pairs.

Warp uses batch 4. Composition uses batch 1 because the warp canvas size
changes per pair. Warp stays frozen while composition trains. Oversized
canvases are skipped.

    python drive_train.py --dataset widedrive3 --gpu 4 --epochs 3
"""
import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prelaunch():
    if os.environ.get("UDIS_GPU_READY") == "1":
        return
    gpu = "0"
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "--gpu" and index + 1 < len(args):
            gpu = args[index + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["UDIS_GPU_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    prelaunch()

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_LEGACY = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub/datasets")
WORK = _LEGACY if _LEGACY.exists() else ROOT.parent / "datasets"
def camera_size(name):
    return (518, 434) if "1224" in name else (518, 294)


EPOCHS = {"nuscenes": 1, "lyft": 3, "lyft1920": 3, "lyft1224": 3, "widedrive3": 3}
PARTS = {
    "nuscenes": [(WORK / "nuscenes/processed_10Hz_v2", "nuScenes_Train.txt")],
    "lyft1920": [(WORK / "lyft/lyft_train1920_3cams", "train1920.txt")],
    "lyft1224": [(WORK / "lyft/lyft_train1224_3cams", "train1224.txt")],
    "widedrive3": [(WORK / "WideDrive3_processed/WideDriveTrain", "train.txt")],
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec.loader.exec_module(module)
    return module


def state_of(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def read_list(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


class Pairs(Dataset):
    def __init__(self, parts, limit=0, size=(518, 294)):
        self.size = size
        self.samples = []
        for root, name in parts:
            for scene in read_list(root / name):
                image_dir = root / scene / "images"
                if not image_dir.is_dir():
                    continue
                frames = {}
                for path in image_dir.glob("*.jpg"):
                    frame, cam = path.stem.rsplit("_", 1)
                    if cam in {"3", "4", "5"}:
                        frames.setdefault(frame, set()).add(cam)
                for frame, cams in frames.items():
                    if {"3", "4", "5"} <= cams:
                        self.samples.append((image_dir / f"{frame}_5.jpg", image_dir / f"{frame}_4.jpg"))
                        self.samples.append((image_dir / f"{frame}_5.jpg", image_dir / f"{frame}_3.jpg"))
        if limit:
            random.Random(0).shuffle(self.samples)
            self.samples = self.samples[:limit]
        if not self.samples:
            raise SystemExit("no pairs")
        print(f"pairs {len(self.samples)}", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        left, right = self.samples[index]
        if random.randint(0, 1):
            left, right = right, left
        return image(left, self.size), image(right, self.size)


def image(path, size):
    array = cv2.imread(str(path), cv2.IMREAD_COLOR)
    array = cv2.resize(array, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    array = (array / 127.5) - 1.0
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


def train_warp(args, epochs):
    warp = load_module("udis_warp_net", ROOT / "Warp" / "Codes" / "network.py")
    loss_fn = load_module("udis_warp_loss", ROOT / "Warp" / "Codes" / "loss.py")
    loader = DataLoader(Pairs(PARTS[args.dataset], args.limit, camera_size(args.dataset)), batch_size=args.warp_batch, shuffle=True, num_workers=args.workers, drop_last=True)
    net = warp.Network().cuda().train()
    weight = args.warp_weight or str(ROOT / "pretrained" / "Warp" / "epoch100_model.pth")
    net.load_state_dict(state_of(weight))
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    save_dir = ROOT / "outputs" / "finetune" / args.dataset / "warp"
    save_dir.mkdir(parents=True, exist_ok=True)
    steps = 0
    for epoch in range(epochs):
        for index, (left, right) in enumerate(loader):
            left = left.float().cuda(non_blocking=True)
            right = right.float().cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = warp.build_model(net, left, right)
            overlap = loss_fn.cal_lp_loss(left, right, out["output_H"], out["output_H_inv"], out["warp_mesh"], out["warp_mesh_mask"])
            regular = 10 * loss_fn.inter_grid_loss(out["overlap"], out["mesh2"]) + 10 * loss_fn.intra_grid_loss(out["mesh2"])
            loss = overlap + regular
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
            optimizer.step()
            steps += 1
            if index % 20 == 0:
                print(f"warp epoch {epoch + 1}/{epochs} step {index + 1} loss {float(loss):.4f}", flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        path = save_dir / f"epoch_{epoch + 1}.pth"
        torch.save({"model": net.state_dict(), "epoch": epoch + 1}, path)
        print(f"saved {path}", flush=True)
        if args.max_steps and steps >= args.max_steps:
            return path
    return save_dir / f"epoch_{epochs}.pth"


def train_composition(args, epochs, warp_path):
    warp = load_module("udis_warp_net", ROOT / "Warp" / "Codes" / "network.py")
    composition = load_module("udis_comp_net", ROOT / "Composition" / "Codes" / "network.py")
    loss_fn = load_module("udis_comp_loss", ROOT / "Composition" / "Codes" / "loss.py")
    loader = DataLoader(Pairs(PARTS[args.dataset], args.limit, camera_size(args.dataset)), batch_size=1, shuffle=True, num_workers=args.workers)
    warp_net = warp.Network().cuda().eval()
    warp_net.load_state_dict(state_of(warp_path))
    for parameter in warp_net.parameters():
        parameter.requires_grad_(False)
    net = composition.Network().cuda().train()
    weight = args.composition_weight or str(ROOT / "pretrained" / "Composition" / "epoch050_model.pth")
    net.load_state_dict(state_of(weight))
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    save_dir = ROOT / "outputs" / "finetune" / args.dataset / "composition"
    save_dir.mkdir(parents=True, exist_ok=True)
    steps = skipped = 0
    for epoch in range(epochs):
        for index, (left, right) in enumerate(loader):
            left = left.float().cuda(non_blocking=True)
            right = right.float().cuda(non_blocking=True)
            try:
                with torch.no_grad():
                    warped = warp.build_output_model(warp_net, left, right)
                canvas = max(int(warped["final_warp1"].shape[-1]), int(warped["final_warp1"].shape[-2]))
                if canvas > args.max_canvas:
                    raise RuntimeError(f"canvas {canvas}")
                optimizer.zero_grad(set_to_none=True)
                mask1 = warped["final_warp1_mask"][:, :1]
                mask2 = warped["final_warp2_mask"][:, :1]
                out = composition.build_model(
                    net,
                    warped["final_warp1"],
                    warped["final_warp2"],
                    mask1,
                    mask2,
                )
                boundary, _mask = loss_fn.cal_boundary_term(
                    warped["final_warp1"], warped["final_warp2"], mask1, mask2, out["stitched_image"]
                )
                smooth = 1000 * loss_fn.cal_smooth_term_stitch(out["stitched_image"], out["learned_mask1"])
                diff = 1000 * loss_fn.cal_smooth_term_diff(
                    warped["final_warp1"], warped["final_warp2"], out["learned_mask1"], mask1 * mask2
                )
                loss = 10000 * boundary + smooth + diff
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3)
                optimizer.step()
            except RuntimeError as exc:
                skipped += 1
                print(f"skip composition {index}: {exc}", flush=True)
                if args.max_steps and index + 1 >= args.max_steps * 4 and steps == 0:
                    break
                continue
            steps += 1
            print(f"composition epoch {epoch + 1}/{epochs} step {steps} loss {float(loss):.4f} canvas {canvas}", flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        path = save_dir / f"epoch_{epoch + 1}.pth"
        torch.save({"model": net.state_dict(), "epoch": epoch + 1}, path)
        print(f"saved {path} skipped {skipped}", flush=True)
        if args.max_steps and steps >= args.max_steps:
            return


def main():
    parser = argparse.ArgumentParser(description="UDIS++ warp then composition")
    parser.add_argument("--dataset", required=True, choices=sorted(EPOCHS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--stage", choices=("both", "warp", "composition"), default="both")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warp-batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warp-weight", default="")
    parser.add_argument("--composition-weight", default="")
    parser.add_argument("--max-canvas", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    names = ("lyft1920", "lyft1224") if args.dataset == "lyft" else (args.dataset,)
    for name in names:
        args.dataset = name
        epochs = args.epochs or EPOCHS[name]
        warp_path = ROOT / "outputs" / "finetune" / name / "warp" / f"epoch_{epochs}.pth"
        if args.stage in ("both", "warp"):
            warp_path = train_warp(args, epochs)
        if args.stage in ("both", "composition"):
            if not Path(warp_path).is_file():
                warp_path = Path(args.warp_weight or ROOT / "pretrained" / "Warp" / "epoch100_model.pth")
            train_composition(args, epochs, warp_path)


if __name__ == "__main__":
    main()
