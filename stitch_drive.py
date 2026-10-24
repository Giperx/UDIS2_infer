#!/usr/bin/env python3
"""Stitch one driving camera pair with the UDIS++ warp and composition nets.

    python stitch_drive.py --dataset nuscenes --scene 000 --frame 0 --gpu 4
"""
import argparse
import importlib.util
import os
import re
import subprocess
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

_LEGACY = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub/datasets")
WORK = _LEGACY if _LEGACY.exists() else ROOT.parent / "datasets"
ROOTS = {
    "nuscenes": WORK / "nuscenes/processed_10Hz_v2",
    "lyft1920": WORK / "lyft/lyft_val1920_3cams",
    "lyft1224": WORK / "lyft/lyft_val1224_3cams",
    "widedrive3": WORK / "WideDrive3_processed/WideDriveVal",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def state_of(path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint


def load_image(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"missing {path}")
    original = image
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    tensor = torch.from_numpy(np.transpose((resized / 127.5) - 1.0, (2, 0, 1))).unsqueeze(0).cuda()
    return tensor, original


class MethodMemoryError(RuntimeError):
    """The stitch itself needs more memory than the GPU, with nobody else using it."""


def is_oom(exc):
    return "out of memory" in str(exc).lower()


def other_process_holds_gpu(exc=None, minimum_mib=1024):
    me = os.getpid()
    if exc is not None:
        for pid, amount, unit in re.findall(r"Process (\d+) has ([0-9.]+) ([MG]iB)", str(exc)):
            if int(pid) == me:
                continue
            mib = float(amount) * (1024 if unit.startswith("G") else 1)
            if mib >= minimum_mib:
                return True
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    try:
        text = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, used = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid != me and used >= minimum_mib:
            return True
    return False


def oom_kind(exc):
    if not is_oom(exc):
        return None
    return "contention" if other_process_holds_gpu(exc) else "method"


def camera_on_white(image):
    """Camera 5 in the middle, one camera width of white on each side."""
    height, width = image.shape[:2]
    canvas = np.full((height, width * 3, 3), 255, np.uint8)
    canvas[:, width:width * 2] = image
    return canvas


def to_image(tensor):
    array = tensor[0].detach().float().cpu().numpy().transpose(1, 2, 0)
    return np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="UDIS++ driving stitch")
    parser.add_argument("--dataset", required=True, choices=sorted(ROOTS))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame", default="000")
    parser.add_argument("--side", type=int, default=4, choices=(3, 4))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warp", default=str(ROOT / "pretrained/Warp/epoch100_model.pth"))
    parser.add_argument("--composition", default=str(ROOT / "pretrained/Composition/epoch050_model.pth"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    frame = f"{int(args.frame):03d}"
    image_dir = ROOTS[args.dataset] / args.scene / "images"
    warp = load_module("udis_warp_net", ROOT / "Warp/Codes/network.py")
    composition = load_module("udis_comp_net", ROOT / "Composition/Codes/network.py")
    warp_net = warp.Network().cuda().eval()
    comp_net = composition.Network().cuda().eval()
    warp_net.load_state_dict(state_of(args.warp))
    comp_net.load_state_dict(state_of(args.composition))
    size = (518, 434) if "1224" in args.dataset else (518, 294)
    left, camera5 = load_image(image_dir / f"{frame}_5.jpg", size)
    right, _original = load_image(image_dir / f"{frame}_{args.side}.jpg", size)
    try:
        with torch.no_grad():
            warped = warp.build_output_model(warp_net, left, right)
            stitched = composition.build_model(
                comp_net,
                warped["final_warp1"],
                warped["final_warp2"],
                warped["final_warp1_mask"][:, :1],
                warped["final_warp2_mask"][:, :1],
            )["stitched_image"]
        image = to_image(stitched)
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        kind = oom_kind(exc)
        if kind == "contention":
            print(f"keep camera 5 on white: another process holds the GPU ({exc})", flush=True)
            image = camera_on_white(camera5)
        else:
            print(f"fail {frame}: method cannot stitch this frame, nothing saved ({exc})", flush=True)
            return
    output = Path(args.output or ROOT / "outputs" / args.dataset / args.scene / "pairwise" / f"{frame}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print(f"saved {output} canvas {image.shape[1]}x{image.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
