import argparse
import json
import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Cityscapes20ClassDataset
from eval import init_wandb, load_train_config, resolve_eval_amp_args
from main import DEVICE, autocast_context, build_source_net
from visualization import CITYSCAPES_20_PALETTE, colorize_mask, image_to_numpy


CITYSCAPES_VOID_IGNORE_INDEX = 19
CITYSCAPES_NUM_CLASSES = 20


class SegmentationMetricsIgnoreVoid:
    """Twenty-class confusion matrix with Cityscapes void GT ignored."""

    def __init__(self, model_num_classes: int, ignore_index: int = 19) -> None:
        self.model_num_classes = int(model_num_classes)
        self.ignore_index = int(ignore_index)
        if self.model_num_classes <= 0:
            raise ValueError("model_num_classes must be positive")
        if not (0 <= self.ignore_index < self.model_num_classes):
            raise ValueError("ignore_index must be within model_num_classes")

        self.evaluated_classes = [
            cls for cls in range(self.model_num_classes) if cls != self.ignore_index
        ]
        self.evaluated_num_classes = len(self.evaluated_classes)
        self.confmat = torch.zeros(
            self.model_num_classes,
            self.model_num_classes,
            dtype=torch.int64,
        )

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.reshape(-1).cpu().long()
        target = target.reshape(-1).cpu().long()
        valid = (
            (target >= 0)
            & (target < self.model_num_classes)
            & (target != self.ignore_index)
            & (pred >= 0)
            & (pred < self.model_num_classes)
        )
        pred = pred[valid]
        target = target[valid]

        # A valid GT pixel predicted as void remains in column 19 and therefore
        # contributes a false negative to its GT class.
        idx = target * self.model_num_classes + pred
        bins = torch.bincount(idx, minlength=self.model_num_classes**2)
        self.confmat += bins.reshape(self.model_num_classes, self.model_num_classes)

    def compute(self) -> Dict[str, object]:
        conf = self.confmat.to(torch.float64)
        eval_idx = torch.tensor(self.evaluated_classes, dtype=torch.long)
        tp = conf.diag()
        gt_count = conf.sum(dim=1)
        pred_count = conf.sum(dim=0)
        union = gt_count + pred_count - tp

        iou = tp[eval_idx] / union[eval_idx].clamp_min(1.0)
        acc_cls = tp[eval_idx] / gt_count[eval_idx].clamp_min(1.0)
        pixel_acc = tp[eval_idx].sum() / conf[eval_idx, :].sum().clamp_min(1.0)
        return {
            "pixel_acc": float(pixel_acc.item()),
            "mIoU": float(iou.mean().item()),
            "mAcc": float(acc_cls.mean().item()),
            "IoU_per_class": [float(value.item()) for value in iou],
            "Acc_per_class": [float(value.item()) for value in acc_cls],
            "confusion_matrix": self.confmat.tolist(),
            "evaluated_classes": self.evaluated_classes,
            "evaluated_num_classes": self.evaluated_num_classes,
            "ignore_index": self.ignore_index,
        }


class SourceStatistics:
    """Streaming population statistics for mu, confidence, and predictions."""

    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)

        self.mu_count = 0
        self.mu_mean = 0.0
        self.mu_m2 = 0.0
        self.mu_abs_sum = 0.0
        self.mu_min = math.inf
        self.mu_max = -math.inf

        self.confidence_count = 0
        self.confidence_mean = 0.0
        self.confidence_m2 = 0.0

        self.valid_gt_count = 0
        self.void_prediction_count = 0
        self.prediction_count = torch.zeros(num_classes, dtype=torch.int64)

    @staticmethod
    def _merge_moments(
        old_count: int,
        old_mean: float,
        old_m2: float,
        values: torch.Tensor,
    ) -> tuple[int, float, float]:
        batch_count = values.numel()
        if batch_count == 0:
            return old_count, old_mean, old_m2

        batch_mean = float(values.mean().item())
        batch_m2 = float(values.var(unbiased=False).item()) * batch_count
        if old_count == 0:
            return batch_count, batch_mean, batch_m2

        total_count = old_count + batch_count
        delta = batch_mean - old_mean
        total_mean = old_mean + delta * batch_count / total_count
        total_m2 = (
            old_m2
            + batch_m2
            + delta * delta * old_count * batch_count / total_count
        )
        return total_count, total_mean, total_m2

    @torch.no_grad()
    def update(
        self,
        mu: torch.Tensor,
        confidence: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        mu_float = mu.detach().float()
        confidence_float = confidence.detach().float()
        self.mu_count, self.mu_mean, self.mu_m2 = self._merge_moments(
            self.mu_count,
            self.mu_mean,
            self.mu_m2,
            mu_float,
        )
        self.mu_abs_sum += float(mu_float.abs().sum().item())
        self.mu_min = min(self.mu_min, float(mu_float.min().item()))
        self.mu_max = max(self.mu_max, float(mu_float.max().item()))

        (
            self.confidence_count,
            self.confidence_mean,
            self.confidence_m2,
        ) = self._merge_moments(
            self.confidence_count,
            self.confidence_mean,
            self.confidence_m2,
            confidence_float,
        )

        pred_cpu = pred.detach().reshape(-1).cpu().long()
        target_cpu = target.detach().reshape(-1).cpu().long()
        self.prediction_count += torch.bincount(
            pred_cpu,
            minlength=self.num_classes,
        )[: self.num_classes]

        valid_gt = (
            (target_cpu >= 0)
            & (target_cpu < self.num_classes)
            & (target_cpu != self.ignore_index)
        )
        self.valid_gt_count += int(valid_gt.sum().item())
        self.void_prediction_count += int(
            ((pred_cpu == self.ignore_index) & valid_gt).sum().item()
        )

    def compute(self) -> Dict[str, object]:
        if self.mu_count == 0 or self.confidence_count == 0:
            raise RuntimeError("No source-generator outputs were evaluated.")

        total_predictions = int(self.prediction_count.sum().item())
        prediction_ratios = (
            self.prediction_count.to(torch.float64)
            / max(total_predictions, 1)
        )
        return {
            "mu_mean": self.mu_mean,
            "mu_std": math.sqrt(max(self.mu_m2 / self.mu_count, 0.0)),
            "mu_abs_mean": self.mu_abs_sum / self.mu_count,
            "mu_min": self.mu_min,
            "mu_max": self.mu_max,
            "confidence_mean": self.confidence_mean,
            "confidence_std": math.sqrt(
                max(self.confidence_m2 / self.confidence_count, 0.0)
            ),
            "void_prediction_ratio": (
                self.void_prediction_count / max(self.valid_gt_count, 1)
            ),
            "prediction_ratio_per_class": [
                float(value.item()) for value in prediction_ratios
            ],
            "valid_gt_pixel_count": self.valid_gt_count,
            "void_prediction_count": self.void_prediction_count,
            "prediction_pixel_count": total_predictions,
        }


def save_source_visualization(
    img: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    confidence: torch.Tensor,
    save_path: Path,
    imagenet_normalize: bool,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(
        image_to_numpy(img, imagenet_normalize=imagenet_normalize)
    )
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(colorize_mask(gt))
    axes[1].set_title("GT")
    axes[1].axis("off")

    axes[2].imshow(colorize_mask(pred))
    axes[2].set_title("source prediction")
    axes[2].axis("off")

    confidence_image = axes[3].imshow(
        confidence.detach().cpu().numpy(),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[3].set_title("confidence")
    axes[3].axis("off")
    fig.colorbar(confidence_image, ax=axes[3], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_individual_outputs(
    pred: torch.Tensor,
    confidence: torch.Tensor,
    stem: str,
    pred_color_dir: Path,
    pred_label_dir: Path,
    confidence_dir: Path,
) -> None:
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    confidence_np = np.rint(
        confidence.detach().float().cpu().numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)

    Image.fromarray(CITYSCAPES_20_PALETTE[pred_np], mode="RGB").save(
        pred_color_dir / f"{stem}.png"
    )
    Image.fromarray(pred_np, mode="L").save(pred_label_dir / f"{stem}.png")
    Image.fromarray(confidence_np, mode="L").save(
        confidence_dir / f"{stem}.png"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.num_visualize < 0:
        raise ValueError("--num_visualize must be non-negative")
    if args.eval_image_size is not None and any(
        size <= 0 for size in args.eval_image_size
    ):
        raise ValueError("--eval_image_size values must be positive")
    if args.wandb_num_images < 0:
        raise ValueError("--wandb_num_images must be non-negative")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    validate_args(args)
    result_dir = Path(args.result_dir)
    ckpt_path = result_dir / args.ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    train_args = load_train_config(result_dir, ckpt)
    args = resolve_eval_amp_args(args, train_args)

    if int(train_args.num_classes) != CITYSCAPES_NUM_CLASSES:
        raise RuntimeError(
            "Source-generator evaluation expects 20 model classes "
            f"(0-18 plus void=19), but got {train_args.num_classes}."
        )

    if args.eval_image_size is None:
        eval_image_size = tuple(train_args.image_size)
    else:
        eval_image_size = tuple(args.eval_image_size)

    source_variant_tag = str(
        getattr(train_args, "source_segformer_variant", "unknown")
    )
    save_dir = (
        result_dir
        / (
            f"eval_source_{args.split}_"
            f"{source_variant_tag}_"
            f"size{eval_image_size[0]}x{eval_image_size[1]}"
        )
    )
    vis_dir = save_dir / "visualizations"
    pred_color_dir = save_dir / "pred_color"
    pred_label_dir = save_dir / "pred_label"
    confidence_dir = save_dir / "confidence"
    for output_dir in (
        save_dir,
        vis_dir,
        pred_color_dir,
        pred_label_dir,
        confidence_dir,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

    source_net = build_source_net(train_args)
    if source_net is None:
        raise RuntimeError(
            "Source-generator evaluation requires prior_type='image_gaussian'."
        )

    if "source_net" not in ckpt:
        raise RuntimeError(
            "Checkpoint does not contain source_net state."
        )

    source_net.load_state_dict(ckpt["source_net"], strict=True)
    source_net.eval()
    # The endpoint/optimizer states may be large and are not part of this
    # evaluation.  Drop the checkpoint container after the source state loads.
    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = Cityscapes20ClassDataset(
        root=train_args.root,
        split=args.split,
        mode="fine",
        image_size=eval_image_size,
        augment=False,
        color_jitter=False,
        imagenet_normalize=getattr(train_args, "imagenet_normalize", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEVICE == "cuda",
    )

    wandb_config = vars(train_args).copy()
    wandb_config.update(vars(args))
    wandb_config.update(
        {
            "evaluation_target": "source_generator_mu",
            "prediction_rule": "argmax(mu, dim=1)",
            "eval_image_size": list(eval_image_size),
        }
    )
    wandb = init_wandb(args, wandb_config)

    metrics = SegmentationMetricsIgnoreVoid(
        model_num_classes=train_args.num_classes,
        ignore_index=CITYSCAPES_VOID_IGNORE_INDEX,
    )
    source_statistics = SourceStatistics(
        num_classes=train_args.num_classes,
        ignore_index=CITYSCAPES_VOID_IGNORE_INDEX,
    )
    visualized = 0
    wandb_images = []

    try:
        for batch_idx, (img, _, gt_mask) in enumerate(
            tqdm(loader, desc=f"eval_source:{args.split}")
        ):
            img = img.to(DEVICE, non_blocking=True)
            gt_mask = gt_mask.to(DEVICE, non_blocking=True)

            with autocast_context(args):
                source_out = source_net(img)

                if isinstance(source_out, (tuple, list)) and len(source_out) == 3:
                    source_x0, mu, logvar = source_out
                    del source_x0
                elif isinstance(source_out, (tuple, list)) and len(source_out) == 2:
                    mu, logvar = source_out
                else:
                    raise RuntimeError(
                        "source_net(img) must return (x0, mu, logvar) or "
                        "(mu, logvar)."
                    )

            # Neither the sampled x0 nor logvar participates in prediction.
            del source_out, logvar
            if not isinstance(mu, torch.Tensor) or mu.ndim != 4:
                shape = getattr(mu, "shape", None)
                raise RuntimeError(
                    "source_net mu must be a 4D [B,K,H,W] tensor, "
                    f"but got shape {shape}."
                )

            if mu.shape[-2:] != gt_mask.shape[-2:]:
                mu = F.interpolate(
                    mu.float(),
                    size=gt_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if mu.shape[1] != train_args.num_classes:
                raise RuntimeError(
                    f"source_net mu has {mu.shape[1]} class channels; "
                    f"expected {train_args.num_classes}."
                )
            expected_mu_shape = (
                img.shape[0],
                int(train_args.num_classes),
                gt_mask.shape[-2],
                gt_mask.shape[-1],
            )
            if tuple(mu.shape) != expected_mu_shape:
                raise RuntimeError(
                    f"source_net mu shape {tuple(mu.shape)} is incompatible with "
                    f"expected [B,K,H,W]={expected_mu_shape}."
                )

            # The segmentation class is determined directly in CFM state space.
            pred = mu.argmax(dim=1)
            expected_pred_shape = (
                img.shape[0],
                gt_mask.shape[-2],
                gt_mask.shape[-1],
            )
            if tuple(pred.shape) != expected_pred_shape:
                raise RuntimeError(
                    f"prediction shape {tuple(pred.shape)} is incompatible with "
                    f"expected [B,H,W]={expected_pred_shape}."
                )

            # Softmax is used only for confidence, never to choose pred.
            probs = torch.softmax(mu.float(), dim=1)
            confidence = probs.max(dim=1).values

            metrics.update(pred, gt_mask)
            source_statistics.update(mu, confidence, pred, gt_mask)

            remaining = max(0, args.num_visualize - visualized)
            take_visualizations = min(remaining, img.shape[0])
            for sample_idx in range(img.shape[0]):
                stem = (
                    f"{args.split}_batch{batch_idx:04d}_idx{sample_idx:02d}"
                )
                if args.save_individual_masks:
                    save_individual_outputs(
                        pred=pred[sample_idx],
                        confidence=confidence[sample_idx],
                        stem=stem,
                        pred_color_dir=pred_color_dir,
                        pred_label_dir=pred_label_dir,
                        confidence_dir=confidence_dir,
                    )

                if sample_idx < take_visualizations:
                    save_path = vis_dir / f"{stem}.png"
                    save_source_visualization(
                        img=img[sample_idx].detach().cpu(),
                        gt=gt_mask[sample_idx].detach().cpu(),
                        pred=pred[sample_idx].detach().cpu(),
                        confidence=confidence[sample_idx].detach().cpu(),
                        save_path=save_path,
                        imagenet_normalize=getattr(
                            train_args, "imagenet_normalize", False
                        ),
                    )
                    if (
                        wandb is not None
                        and args.wandb_log_images
                        and len(wandb_images) < args.wandb_num_images
                    ):
                        wandb_images.append(
                            wandb.Image(
                                str(save_path),
                                caption=(
                                    f"{args.split}/batch_{batch_idx}/"
                                    f"idx_{sample_idx}"
                                ),
                            )
                        )
            visualized += take_visualizations

        result = metrics.compute()
        stats_result = source_statistics.compute()
        result.update(
            {
                "evaluation_target": "source_generator_mu",
                "prediction_rule": "argmax(mu, dim=1)",
                "uses_source_noise": False,
                "uses_cfm_model": False,
                "uses_flow_sampling": False,
                "split": args.split,
                "checkpoint": str(ckpt_path),
                "eval_image_size": list(eval_image_size),
                "num_classes": int(train_args.num_classes),
                "source_backbone": getattr(
                    train_args, "source_backbone", None
                ),
                "source_segformer_variant": getattr(
                    train_args, "source_segformer_variant", None
                ),
                "source_fixed_std": getattr(
                    train_args, "source_fixed_std", None
                ),
                "source_learned_logvar": bool(
                    getattr(train_args, "source_learned_logvar", False)
                ),
                "source_statistics": stats_result,
            }
        )

        with open(save_dir / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

        source_name = (
            f"SegFormer-{result['source_segformer_variant']}"
            if result["source_backbone"] == "segformer"
            else str(result["source_backbone"])
        )
        with open(save_dir / "metrics.txt", "w", encoding="utf-8") as file:
            file.write("evaluation target : source_generator mu\n")
            file.write("prediction rule   : argmax(mu)\n")
            file.write("uses source noise : false\n")
            file.write("uses CFM sampling : false\n")
            file.write(f"split             : {args.split}\n")
            file.write(f"checkpoint        : {ckpt_path}\n")
            file.write(
                f"image size        : {eval_image_size[0]}x{eval_image_size[1]}\n"
            )
            file.write(f"source            : {source_name}\n")
            file.write(f"pixel_acc         : {result['pixel_acc']:.6f}\n")
            file.write(f"mIoU              : {result['mIoU']:.6f}\n")
            file.write(f"mAcc              : {result['mAcc']:.6f}\n")
            file.write(
                "void pred ratio   : "
                f"{stats_result['void_prediction_ratio']:.6f}\n"
            )

        if wandb is not None:
            log_payload = {
                "source_eval/pixel_acc": result["pixel_acc"],
                "source_eval/mIoU": result["mIoU"],
                "source_eval/mAcc": result["mAcc"],
                "source_eval/confidence_mean": stats_result[
                    "confidence_mean"
                ],
                "source_eval/void_prediction_ratio": stats_result[
                    "void_prediction_ratio"
                ],
            }
            if args.wandb_log_images and wandb_images:
                log_payload["source_eval/images"] = wandb_images
            wandb.log(log_payload)

        print("Source-generator evaluation finished.")
        print("prediction : argmax(mu)")
        print(f"pixel_acc : {result['pixel_acc']:.6f}")
        print(f"mIoU      : {result['mIoU']:.6f}")
        print(f"mAcc      : {result['mAcc']:.6f}")
        print(f"void ratio: {stats_result['void_prediction_ratio']:.6f}")
        print(f"saved to  : {save_dir}")
    finally:
        if wandb is not None:
            wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="segdiff_final.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_visualize", type=int, default=16)
    parser.add_argument(
        "--eval_image_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
    )
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument(
        "--amp_dtype",
        choices=["bf16", "fp16"],
        default="bf16",
    )
    parser.add_argument(
        "--save_individual_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="CFM-segv1")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_log_images", action="store_true")
    parser.add_argument("--wandb_num_images", type=int, default=4)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
