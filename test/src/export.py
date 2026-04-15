"""Validate that mIoU computed from exported features matches the canvit_eval reference pipeline."""

from pathlib import Path

import torch
from canvit_pytorch import SegmentationProbe
from canvit_pytorch.teacher import DINOv3Teacher
from canvit_specialize.datasets.ade20k import IGNORE_LABEL, NUM_CLASSES
from canvit_specialize.metrics import mIoUAccumulator

from canvit_eval.config import ade20k_root
from canvit_eval.tasks.ade20k_seg import _update_miou

RESOLUTIONS = [128, 144, 160, 192, 256, 384]
PROBE_REPO_TEMPLATE = "canvit/probe-ade20k-40k-dv3b-{}px"
EXPORTS_DIR = Path("outputs")
ATOL = 1e-3  # mIoU tolerance (~0.1 pp)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def miou_from_export(
    resolution: int, probe: SegmentationProbe, device: torch.device
) -> float:
    """Compute mIoU at t=0, s=1 by running the probe over exported features.

    Loads outputs/{resolution}px/features.pt, streams in batches of 32, calls
    _update_miou, and returns acc.compute(). Probe is passed in to avoid
    reloading it on every call.
    """
    data = torch.load(EXPORTS_DIR / f"{resolution}px/features.pt", map_location="cpu", weights_only=False)
    feats_all: torch.Tensor = data["feats"]   # [N, grid*grid, D]  float32
    masks_all: torch.Tensor = data["masks"]   # [N, scene_size, scene_size]  uint8
    grid: int = data["grid"]
    N = feats_all.shape[0]

    acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)

    with torch.inference_mode():
        for start in range(0, N, 32):
            feats = feats_all[start : start + 32].to(device, non_blocking=True)
            masks = masks_all[start : start + 32].long().to(device, non_blocking=True)
            feats = feats.view(feats.shape[0], grid, grid, -1)  # [B, grid, grid, D]
            _update_miou(acc, probe, feats, masks)

    return acc.compute()


def miou_reference(
    resolution: int, probe: SegmentationProbe, teacher: DINOv3Teacher, device: torch.device
) -> float:
    """Run the canvit_eval DINOv3 reference pipeline and return t0 mIoU.

    Equivalent to:
        uv run python -m canvit_eval ade20k-seg-dinov3 \\
            --probe-repo canvit/probe-ade20k-40k-dv3b-{resolution}px \\
            --eval-resolution {resolution}

    Teacher and probe are passed in to avoid reloading per resolution.
    """
    from canvit_specialize.datasets.ade20k import make_val_transforms, ADE20kDataset
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    img_tf, mask_tf = make_val_transforms(512, "squish")
    dataset = ADE20kDataset(root=ade20k_root(), split="validation",
                            img_transform=img_tf, mask_transform=mask_tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    grid = resolution // 16

    acc = mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            B = images.shape[0]
            resized = F.interpolate(images, size=(resolution, resolution),
                                    mode="bilinear", align_corners=False)
            feats = teacher.forward_norm_features(resized).patches.view(B, grid, grid, -1)
            _update_miou(acc, probe, feats, masks)

    return acc.compute()


def test_miou_from_export(resolution: int = 128) -> float:
    """Smoke test: compute and return mIoU from exported features at one resolution."""
    device = _device()
    probe = SegmentationProbe.from_pretrained(PROBE_REPO_TEMPLATE.format(resolution)).to(device).eval()
    miou = miou_from_export(resolution, probe, device)
    print(f"  res={resolution}px  export_miou={miou:.4f}  ({100 * miou:.2f}%)")
    return miou


def test_all_resolutions() -> None:
    """For every exported resolution, assert export mIoU matches the canvit_eval reference.

    Loads the teacher once and each probe once, reusing them across both pipelines.
    """
    from canvit_pytorch.teacher import load_teacher
    from canvit_eval.config import TEACHER_REPO

    device = _device()
    teacher = load_teacher(TEACHER_REPO, device)

    failures = []
    for resolution in RESOLUTIONS:
        probe = SegmentationProbe.from_pretrained(
            PROBE_REPO_TEMPLATE.format(resolution)
        ).to(device).eval()
        export_miou = miou_from_export(resolution, probe, device)
        ref_miou = miou_reference(resolution, probe, teacher, device)
        diff = abs(export_miou - ref_miou)
        status = "OK" if diff < ATOL else "FAIL"
        print(f"  [{status}] res={resolution}px  export={export_miou:.4f}  ref={ref_miou:.4f}  diff={diff:.6f}")
        if diff >= ATOL:
            failures.append(f"{resolution}px: export={export_miou:.4f} ref={ref_miou:.4f} diff={diff:.6f}")

    if failures:
        raise AssertionError("mIoU mismatch(es):\n" + "\n".join(failures))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    smoke_test_res = 512
    print(f"=== smoke test ({smoke_test_res}px) ===") # Should be 47.2
    test_miou_from_export(smoke_test_res)

    torch.cuda.empty_cache()
    print("\n=== all resolutions vs reference ===")
    test_all_resolutions()
