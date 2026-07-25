from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


CITYSCAPES_20_PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    invalid = (mask_np < 0) | (mask_np >= len(CITYSCAPES_20_PALETTE))
    mask_np = np.where(invalid, len(CITYSCAPES_20_PALETTE) - 1, mask_np)
    return CITYSCAPES_20_PALETTE[mask_np]


def denormalize_imagenet(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0.0, 1.0)


def image_to_numpy(img: torch.Tensor, imagenet_normalize: bool = False) -> np.ndarray:
    if imagenet_normalize:
        img = denormalize_imagenet(img)
    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img_np, 0.0, 1.0)


def save_prediction_triplet(
    img: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    save_path: Path,
    title: Optional[str] = None,
    imagenet_normalize: bool = False,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_to_numpy(img, imagenet_normalize=imagenet_normalize))
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(colorize_mask(gt))
    axes[1].set_title("gt_color")
    axes[1].axis("off")

    axes[2].imshow(colorize_mask(pred))
    axes[2].set_title("pred_color")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def snap_indices(num_frames: int, num_snap_points: int) -> List[int]:
    if num_frames <= 1:
        return [0]
    if num_snap_points <= 1:
        return [num_frames - 1]
    idxs = torch.linspace(0, num_frames - 1, steps=num_snap_points)
    idxs = torch.round(idxs).to(torch.int64).tolist()
    out = []
    for idx in idxs:
        if idx not in out:
            out.append(idx)
    return out


def save_trajectory_grid(
    img: torch.Tensor,
    gt: torch.Tensor,
    traj: torch.Tensor,
    save_path: Path,
    num_snap_points: int = 5,
    imagenet_normalize: bool = False,
) -> None:
    if traj.dim() != 3:
        raise ValueError("traj must be [T,H,W]")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    snap_ids = snap_indices(traj.shape[0], num_snap_points)
    ncols = len(snap_ids) + 2
    fig, axes = plt.subplots(1, ncols, figsize=(2.7 * ncols, 3.5))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(image_to_numpy(img, imagenet_normalize=imagenet_normalize))
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(colorize_mask(gt))
    axes[1].set_title("gt_color")
    axes[1].axis("off")

    for j, tidx in enumerate(snap_ids, start=2):
        axes[j].imshow(colorize_mask(traj[tidx]))
        axes[j].set_title(f"pred_t{tidx}")
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
