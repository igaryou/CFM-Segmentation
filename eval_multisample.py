import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from CFM import CategoricalFlowMaps
from dataset import Cityscapes20ClassDataset
from eval import (
    init_wandb,
    load_train_config,
    resolve_eval_amp_args,
)
from main import (
    DEVICE,
    autocast_context,
    build_cfm,
    build_model,
    build_source_net,
    load_model_state_dict_compat,
)
from visualization import colorize_mask, image_to_numpy


PROB_SUM_ATOL = 5e-3
CITYSCAPES_VOID_IGNORE_INDEX = 19
CITYSCAPES_VOID_CLASS_NAME = "void"


class SegmentationMetricsIgnoreVoid:
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
        pred = pred.view(-1).cpu().long()
        target = target.view(-1).cpu().long()
        valid = (
            (target >= 0)
            & (target < self.model_num_classes)
            & (target != self.ignore_index)
            & (pred >= 0)
            & (pred < self.model_num_classes)
        )
        pred = pred[valid]
        target = target[valid]
        idx = target * self.model_num_classes + pred
        bins = torch.bincount(idx, minlength=self.model_num_classes ** 2)
        self.confmat += bins.reshape(self.model_num_classes, self.model_num_classes)

    def compute(self) -> Dict[str, object]:
        conf = self.confmat.float()
        eval_idx = torch.tensor(self.evaluated_classes, dtype=torch.long)
        tp = conf.diag()
        gt = conf.sum(dim=1)
        pred = conf.sum(dim=0)
        union = gt + pred - tp

        iou = tp[eval_idx] / union[eval_idx].clamp_min(1.0)
        acc_cls = tp[eval_idx] / gt[eval_idx].clamp_min(1.0)
        pixel_acc = tp[eval_idx].sum() / conf[eval_idx, :].sum().clamp_min(1.0)
        miou = iou.mean()
        macc = acc_cls.mean()
        return {
            "pixel_acc": float(pixel_acc.item()),
            "mIoU": float(miou.item()),
            "mAcc": float(macc.item()),
            "IoU_per_class": [float(x.item()) for x in iou],
            "Acc_per_class": [float(x.item()) for x in acc_cls],
            "confusion_matrix": conf.to(torch.int64).tolist(),
        }


def positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return out


def normalize_samples(values) -> list[int]:
    samples = sorted({int(v) for v in values})
    if not samples or any(v <= 0 for v in samples):
        raise ValueError("--samples must contain positive integers")
    return samples


def get_aggregation_name(prediction_mode: str) -> str:
    if prediction_mode == "argmax":
        return "probability_mean_argmax"
    if prediction_mode == "sample":
        return "sample_then_pixelwise_majority_vote"
    raise ValueError(f"unsupported prediction mode: {prediction_mode}")


def get_aggregation_tag(prediction_mode: str) -> str:
    if prediction_mode == "argmax":
        return "probmean"
    if prediction_mode == "sample":
        return "hardvote"
    raise ValueError(f"unsupported prediction mode: {prediction_mode}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def build_tags(args: argparse.Namespace, train_args: argparse.Namespace, eval_image_size: tuple[int, int]):
    source_std = getattr(train_args, "source_fixed_std", None)
    std_tag = "std_learned" if source_std is None else f"std{str(source_std)}"

    backbone_tag = getattr(train_args, "backbone", "unet")
    if backbone_tag == "segformer":
        endpoint_tag = f"endpoint_{getattr(train_args, 'endpoint_segformer_variant', 'unknown')}"
    else:
        endpoint_tag = "endpoint_unet"

    if args.use_cfg:
        scale_tag = str(args.cfg_scale).replace(".", "p")
        cfg_tag = f"cfg_{args.cfg_null_condition}_{scale_tag}"
    else:
        cfg_tag = "cfg_off"

    size_tag = f"size{eval_image_size[0]}x{eval_image_size[1]}"
    return backbone_tag, endpoint_tag, std_tag, cfg_tag, size_tag


def build_save_dir(
    result_dir: Path,
    args: argparse.Namespace,
    train_args: argparse.Namespace,
    eval_image_size: tuple[int, int],
    sample_counts: list[int],
) -> Path:
    backbone_tag, endpoint_tag, std_tag, cfg_tag, size_tag = build_tags(
        args,
        train_args,
        eval_image_size,
    )
    aggregation_tag = get_aggregation_tag(args.prediction_mode)
    samples_tag = (
        "samples" + "-".join(str(n) for n in sample_counts) + f"_{aggregation_tag}"
    )
    prediction_tag = f"pred-{args.prediction_mode}"
    return (
        result_dir
        / (
            f"eval_multisample_{args.split}_{args.num_steps}steps_"
            f"{std_tag}_"
            f"{samples_tag}_{prediction_tag}"
        )
    )


@torch.no_grad()
def compute_source_stats(
    cfm: CategoricalFlowMaps,
    img: torch.Tensor,
    source_net: Optional[torch.nn.Module],
    dtype: torch.dtype,
) -> Optional[Dict[str, torch.Tensor]]:
    if cfm.prior_type != "image_gaussian":
        return None
    if source_net is None:
        raise RuntimeError("prior_type='image_gaussian' requires source_net.")

    B, _, H, W = img.shape
    source_out = source_net(img)
    if isinstance(source_out, (tuple, list)) and len(source_out) == 3:
        _, mu, logvar = source_out
    elif isinstance(source_out, (tuple, list)) and len(source_out) == 2:
        mu, logvar = source_out
    else:
        raise RuntimeError("source_net(img) must return (x0, mu, logvar) or (mu, logvar).")

    mu = mu.to(device=img.device, dtype=dtype)
    logvar = logvar.to(device=img.device, dtype=dtype)
    if mu.shape[-2:] != (H, W):
        mu = F.interpolate(mu, size=(H, W), mode="bilinear", align_corners=False)
    if logvar.shape[-2:] != (H, W):
        logvar = F.interpolate(logvar, size=(H, W), mode="bilinear", align_corners=False)
    if mu.shape[0] != B or mu.shape[1] != cfm.num_classes:
        raise RuntimeError(
            f"source_net mu shape {tuple(mu.shape)} is incompatible with "
            f"(B={B}, K={cfm.num_classes}, H={H}, W={W})."
        )

    fixed_std = getattr(source_net, "fixed_std", None)
    if fixed_std is not None:
        std = float(fixed_std)
        sigma = torch.full_like(mu, std)
        logvar = torch.full_like(mu, math.log(std ** 2))
    else:
        sigma = torch.exp(0.5 * logvar)

    return {"mu": mu, "sigma": sigma, "logvar": logvar}


@torch.no_grad()
def sample_initial_state(
    cfm: CategoricalFlowMaps,
    B: int,
    H: int,
    W: int,
    device,
    dtype: torch.dtype,
    img: torch.Tensor,
    source_net: Optional[torch.nn.Module],
    source_stats: Optional[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    if cfm.prior_type == "dirichlet":
        dist = torch.distributions.Dirichlet(
            torch.ones(cfm.num_classes, device=device, dtype=dtype)
        )
        return dist.sample((B, H, W)).permute(0, 3, 1, 2).contiguous()

    if cfm.prior_type == "gaussian":
        return torch.randn(B, cfm.num_classes, H, W, device=device, dtype=dtype)

    if cfm.prior_type == "image_gaussian" and source_stats is not None:
        return source_stats["mu"] + source_stats["sigma"] * torch.randn_like(source_stats["mu"])

    return cfm.sample_prior(
        B=B,
        H=H,
        W=W,
        device=device,
        dtype=dtype,
        img=img,
        source_net=source_net,
    )


@torch.no_grad()
def sample_final_probs(
    cfm: CategoricalFlowMaps,
    model: torch.nn.Module,
    img: torch.Tensor,
    source_net: Optional[torch.nn.Module],
    num_steps: int,
    image_feat: Optional[torch.Tensor],
    source_stats: Optional[Dict[str, torch.Tensor]],
    use_cfg: bool,
    cfg_scale: float,
    cfg_null_condition: str,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("--num_steps must be positive for probability extraction")

    device = img.device
    B, _, H, W = img.shape
    x = sample_initial_state(
        cfm=cfm,
        B=B,
        H=H,
        W=W,
        device=device,
        dtype=img.dtype,
        img=img,
        source_net=source_net,
        source_stats=source_stats,
    )

    if image_feat is None:
        image_feat = model.encode_image(img)

    ts = torch.linspace(0.03, 1.0, num_steps + 1, device=img.device)
    final_probs = None

    for i in range(num_steps):
        s_now = ts[i]
        t_next = ts[i + 1]

        s_batch = torch.full((B,), float(s_now), device=device)
        t_batch = torch.full((B,), float(t_next), device=device)

        if use_cfg and cfg_scale != 1.0:
            logits_cond, _ = model.forward_with_image_feat(
                x,
                image_feat,
                s_batch,
                t_batch,
            )

            null_feat = model.get_null_image_feat(
                image_feat,
                null_condition=cfg_null_condition,
            )

            logits_uncond, _ = model.forward_with_image_feat(
                x,
                null_feat,
                s_batch,
                t_batch,
            )

            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            pi = torch.softmax(logits, dim=1)
        else:
            _, pi = model.forward_with_image_feat(
                x,
                image_feat,
                s_batch,
                t_batch,
            )

        final_probs = pi
        x = cfm.flow_map(x, pi, s_batch, t_batch)

        if cfm.prior_type == "dirichlet" and cfm.project_simplex:
            x = x.clamp_min(1e-8)
            x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-8)

    if final_probs is None:
        raise RuntimeError("no probability was produced by the sampler")
    return final_probs.float()


def probs_to_mask(
    probs: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "argmax":
        return probs.argmax(dim=1)
    if mode != "sample":
        raise ValueError(f"unsupported prediction mode: {mode}")
    if probs.ndim != 4:
        raise ValueError(
            f"probs must have shape [B, K, H, W], got {tuple(probs.shape)}"
        )

    B, K, H, W = probs.shape
    if K <= 0:
        raise ValueError("probs must contain at least one class")

    safe_probs = probs.float()
    safe_probs = torch.nan_to_num(
        safe_probs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe_probs = safe_probs.clamp_min(0.0)

    prob_sum = safe_probs.sum(dim=1, keepdim=True)
    zero_sum = prob_sum <= 1e-12
    safe_probs = safe_probs / prob_sum.clamp_min(1e-12)

    if zero_sum.any():
        uniform = torch.full_like(safe_probs, 1.0 / K)
        safe_probs = torch.where(
            zero_sum.expand_as(safe_probs),
            uniform,
            safe_probs,
        )

    pixel_probs = safe_probs.permute(0, 2, 3, 1).reshape(-1, K)
    sampled = torch.multinomial(pixel_probs, num_samples=1)
    return sampled.reshape(B, H, W).long()


def validate_probs(
    probs: torch.Tensor,
    expected_shape: tuple[int, int, int, int],
    stats: Dict[str, object],
    batch_idx: int,
    sample_idx: int,
) -> None:
    if tuple(probs.shape) != expected_shape:
        raise RuntimeError(
            f"probability shape mismatch at batch={batch_idx}, sample={sample_idx}: "
            f"got {tuple(probs.shape)}, expected {expected_shape}"
        )

    if not torch.isfinite(probs).all():
        raise RuntimeError(f"non-finite probability at batch={batch_idx}, sample={sample_idx}")

    min_prob = float(probs.min().item())
    max_prob = float(probs.max().item())
    max_sum_err = float((probs.sum(dim=1) - 1.0).abs().max().item())
    if min_prob < -1e-6:
        raise RuntimeError(
            f"negative probability at batch={batch_idx}, sample={sample_idx}: {min_prob}"
        )
    if max_sum_err > PROB_SUM_ATOL:
        raise RuntimeError(
            f"class probabilities do not sum to 1 at batch={batch_idx}, "
            f"sample={sample_idx}: max error {max_sum_err}"
        )

    stats["checked_tensors"] = int(stats.get("checked_tensors", 0)) + 1
    stats["shape"] = list(expected_shape)
    stats["min_probability"] = min(float(stats.get("min_probability", min_prob)), min_prob)
    stats["max_probability"] = max(float(stats.get("max_probability", max_prob)), max_prob)
    stats["max_class_sum_abs_error"] = max(
        float(stats.get("max_class_sum_abs_error", 0.0)),
        max_sum_err,
    )
    stats["finite"] = True
    stats["nonnegative"] = True
    stats["class_sum_atol"] = PROB_SUM_ATOL


def save_colored_mask(mask: torch.Tensor, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize_mask(mask)).save(save_path)


def save_entropy_map(entropy: torch.Tensor, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ent = entropy.detach().cpu().numpy().astype(np.float32)
    finite = np.isfinite(ent)
    if finite.any():
        min_val = float(ent[finite].min())
        max_val = float(ent[finite].max())
    else:
        min_val = 0.0
        max_val = 1.0
    denom = max(max_val - min_val, 1e-8)
    norm = np.clip((ent - min_val) / denom, 0.0, 1.0)
    Image.fromarray((norm * 255.0).astype(np.uint8)).save(save_path)


def save_batch_masks(mask: torch.Tensor, save_dir: Path, global_start: int) -> None:
    for i in range(mask.size(0)):
        save_colored_mask(mask[i], save_dir / f"idx{global_start + i:06d}.png")


def save_batch_entropy(entropy: torch.Tensor, save_dir: Path, global_start: int) -> None:
    for i in range(entropy.size(0)):
        save_entropy_map(entropy[i], save_dir / f"idx{global_start + i:06d}.png")


def predictive_entropy(mean_probs: torch.Tensor) -> torch.Tensor:
    return -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(dim=1)


def save_multisample_grid(
    img: torch.Tensor,
    gt: torch.Tensor,
    sample_preds: list[torch.Tensor],
    ensemble_preds: Dict[int, torch.Tensor],
    entropy_maps: Dict[int, torch.Tensor],
    sample_counts: list[int],
    aggregation_tag: str,
    save_path: Path,
    imagenet_normalize: bool,
    save_entropy: bool,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    row1_items = [("image", "image"), ("gt", gt)]
    row1_items.extend((f"sample_{i:02d}", pred) for i, pred in enumerate(sample_preds))

    row2_items = [
        (f"{aggregation_tag}_{n}", ensemble_preds[n]) for n in sample_counts
    ]
    if save_entropy:
        row2_items.extend((f"entropy_{n}", entropy_maps[n]) for n in sample_counts)

    ncols = max(len(row1_items), len(row2_items), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(2.4 * ncols, 5.2))
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for ax in axes.reshape(-1):
        ax.axis("off")

    for col, (title, payload) in enumerate(row1_items):
        ax = axes[0, col]
        if isinstance(payload, str) and payload == "image":
            ax.imshow(image_to_numpy(img, imagenet_normalize=imagenet_normalize))
        else:
            ax.imshow(colorize_mask(payload))
        ax.set_title(title)
        ax.axis("off")

    for col, (title, payload) in enumerate(row2_items):
        ax = axes[1, col]
        if title.startswith("entropy"):
            ax.imshow(payload.detach().cpu().numpy(), cmap="magma")
        else:
            ax.imshow(colorize_mask(payload))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def metric_metadata(
    args: argparse.Namespace,
    train_args: argparse.Namespace,
    ckpt_path: Path,
    eval_image_size: tuple[int, int],
    samples: int,
) -> Dict[str, object]:
    aggregation_name = get_aggregation_name(args.prediction_mode)
    vote_tie_break = (
        "lowest_class_index" if args.prediction_mode == "sample" else None
    )
    return {
        "aggregation": aggregation_name,
        "prediction_mode": args.prediction_mode,
        "vote_tie_break": vote_tie_break,
        "entropy_source": "mean_class_probabilities",
        "samples": samples,
        "split": args.split,
        "num_steps": args.num_steps,
        "checkpoint": str(ckpt_path),
        "image_size": list(eval_image_size),
        "model_num_classes": int(train_args.num_classes),
        "evaluated_num_classes": int(train_args.num_classes - 1),
        "ignore_index": CITYSCAPES_VOID_IGNORE_INDEX,
        "ignore_class_name": CITYSCAPES_VOID_CLASS_NAME,
        "batch_size": args.batch_size,
        "max_val_batches": args.max_val_batches,
        "prior_type": train_args.prior_type,
        "backbone": getattr(train_args, "backbone", "unet"),
        "source_backbone": getattr(train_args, "source_backbone", None),
        "source_segformer_variant": getattr(train_args, "source_segformer_variant", None),
        "endpoint_segformer_variant": getattr(train_args, "endpoint_segformer_variant", None),
        "endpoint_decoder_channels": getattr(train_args, "endpoint_decoder_channels", None),
        "endpoint_time_emb_dim": getattr(train_args, "endpoint_time_emb_dim", None),
        "endpoint_drop_path_rate": getattr(train_args, "endpoint_drop_path_rate", None),
        "fusion_channels": getattr(train_args, "fusion_channels", None),
        "rrdb_blocks": getattr(train_args, "rrdb_blocks", None),
        "rrdb_growth_channels": getattr(train_args, "rrdb_growth_channels", None),
        "source_fixed_std": getattr(train_args, "source_fixed_std", None),
        "use_cfg": bool(args.use_cfg),
        "cfg_scale": float(args.cfg_scale),
        "cfg_drop_prob": float(getattr(train_args, "cfg_drop_prob", 0.0)),
        "cfg_null_condition": getattr(args, "cfg_null_condition", "learned"),
        "seed": int(args.seed),
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    args.samples = normalize_samples(args.samples)
    sample_counts = normalize_samples(args.samples + [1])
    max_samples = max(sample_counts)
    aggregation_name = get_aggregation_name(args.prediction_mode)
    aggregation_tag = get_aggregation_tag(args.prediction_mode)
    vote_tie_break = (
        "lowest_class_index" if args.prediction_mode == "sample" else None
    )
    entropy_source = "mean_class_probabilities"
    if args.num_steps <= 0:
        raise ValueError("--num_steps must be positive")
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        raise ValueError("--max_val_batches must be positive when set")

    set_seed(args.seed)

    result_dir = Path(args.result_dir)
    ckpt_path = result_dir / args.ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    train_args = load_train_config(result_dir, ckpt)
    if args.source_fixed_std is not None:
        train_args.source_fixed_std = args.source_fixed_std
        train_args.source_learned_logvar = False

    args = resolve_eval_amp_args(args, train_args)
    args.use_cfg = bool(getattr(args, "use_cfg", False))
    if args.cfg_scale is None:
        args.cfg_scale = float(getattr(train_args, "cfg_scale", 1.0))
    else:
        args.cfg_scale = float(args.cfg_scale)

    if args.cfg_null_condition is None:
        args.cfg_null_condition = getattr(train_args, "cfg_null_condition", "learned")
    if args.cfg_null_condition not in {"zero", "learned"}:
        raise ValueError("--cfg_null_condition must be one of: zero, learned")
    if args.cfg_scale < 0.0:
        raise ValueError("--cfg_scale must be non-negative")

    if args.eval_image_size is None:
        eval_image_size = tuple(train_args.image_size)
    else:
        eval_image_size = tuple(args.eval_image_size)

    save_dir = build_save_dir(
        result_dir=result_dir,
        args=args,
        train_args=train_args,
        eval_image_size=eval_image_size,
        sample_counts=sample_counts,
    )
    vis_dir = save_dir / "visualizations"
    pred_dir = save_dir / "predictions"
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_all_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(train_args)
    load_model_state_dict_compat(model, ckpt["model"])
    model.eval()

    source_net = build_source_net(train_args)
    if source_net is not None:
        if "source_net" not in ckpt:
            raise RuntimeError("prior_type='image_gaussian' requires source_net state in checkpoint.")
        source_net.load_state_dict(ckpt["source_net"])
        source_net.eval()

    cfm = build_cfm(train_args)
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
        pin_memory=True,
    )

    wandb_config = vars(train_args).copy()
    wandb_config.update(vars(args))
    wandb_config["aggregation"] = aggregation_name
    wandb_config["aggregation_tag"] = aggregation_tag
    wandb_config["prediction_mode"] = args.prediction_mode
    wandb_config["vote_tie_break"] = vote_tie_break
    wandb_config["entropy_source"] = entropy_source
    wandb_config["sample_counts"] = sample_counts
    wandb = init_wandb(args, wandb_config)

    metrics = {
        n: SegmentationMetricsIgnoreVoid(
            model_num_classes=train_args.num_classes,
            ignore_index=CITYSCAPES_VOID_IGNORE_INDEX,
        )
        for n in sample_counts
    }
    probability_checks: Dict[str, object] = {
        "checked_tensors": 0,
        "finite": False,
        "nonnegative": False,
        "class_sum_atol": PROB_SUM_ATOL,
    }
    visualized = 0
    wandb_images = []
    sample_count_set = set(sample_counts)
    max_visual_samples = min(max_samples, 6)

    try:
        for batch_idx, (img, _, gt_mask) in enumerate(tqdm(loader, desc=f"eval_multisample:{args.split}")):
            if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                break

            img = img.to(DEVICE, non_blocking=True)
            gt_mask = gt_mask.to(DEVICE, non_blocking=True)
            if batch_idx == 0:
                label_counts = torch.bincount(
                    gt_mask.detach().view(-1).cpu(),
                    minlength=train_args.num_classes,
                )
                label_counts_text = ", ".join(
                    f"{label}:{int(label_counts[label].item())}"
                    for label in range(train_args.num_classes)
                )
                ignored_pixels = int(label_counts[CITYSCAPES_VOID_IGNORE_INDEX].item())
                evaluated_pixels = int(label_counts.sum().item()) - ignored_pixels
                print(f"First batch GT label pixel counts: {label_counts_text}")
                print(
                    "Cityscapes 19-class eval excludes "
                    f"label {CITYSCAPES_VOID_IGNORE_INDEX} "
                    f"({CITYSCAPES_VOID_CLASS_NAME}): "
                    f"ignored_pixels={ignored_pixels}, "
                    f"evaluated_pixels={evaluated_pixels}"
                )
            B, _, H, W = img.shape
            global_start = batch_idx * args.batch_size
            expected_prob_shape = (B, train_args.num_classes, H, W)

            remaining_vis = max(0, args.num_visualize - visualized)
            take_vis = min(remaining_vis, B)
            visual_sample_preds: list[torch.Tensor] = []
            visual_ensemble_preds: Dict[int, torch.Tensor] = {}
            visual_entropy_maps: Dict[int, torch.Tensor] = {}

            with autocast_context(args):
                image_feat = model.encode_image(img)
                source_stats = compute_source_stats(
                    cfm=cfm,
                    img=img,
                    source_net=source_net,
                    dtype=img.dtype,
                )

                running_prob_sum = None
                running_votes = None
                if args.prediction_mode == "sample":
                    running_votes = torch.zeros(
                        B,
                        train_args.num_classes,
                        H,
                        W,
                        device=img.device,
                        dtype=torch.int32,
                    )

                for sample_idx in range(max_samples):
                    probs = sample_final_probs(
                        cfm=cfm,
                        model=model,
                        img=img,
                        source_net=source_net,
                        num_steps=args.num_steps,
                        image_feat=image_feat,
                        source_stats=source_stats,
                        use_cfg=args.use_cfg,
                        cfg_scale=args.cfg_scale,
                        cfg_null_condition=args.cfg_null_condition,
                    )
                    validate_probs(
                        probs=probs,
                        expected_shape=expected_prob_shape,
                        stats=probability_checks,
                        batch_idx=batch_idx,
                        sample_idx=sample_idx,
                    )

                    pred_mask = probs_to_mask(
                        probs,
                        mode=args.prediction_mode,
                    )
                    if args.prediction_mode == "sample":
                        if running_votes is None:
                            raise RuntimeError(
                                "running_votes is required when "
                                "prediction_mode='sample'"
                            )
                        sample_votes = F.one_hot(
                            pred_mask,
                            num_classes=train_args.num_classes,
                        ).permute(0, 3, 1, 2)
                        running_votes.add_(
                            sample_votes.to(
                                device=running_votes.device,
                                dtype=running_votes.dtype,
                            )
                        )

                    if args.save_all_predictions:
                        save_batch_masks(
                            pred_mask,
                            pred_dir / f"sample_{sample_idx:02d}",
                            global_start,
                        )
                    if take_vis > 0 and sample_idx < max_visual_samples:
                        visual_sample_preds.append(pred_mask[:take_vis].detach().cpu())

                    if running_prob_sum is None:
                        running_prob_sum = probs.detach().clone()
                    else:
                        running_prob_sum.add_(probs.detach())

                    current_n = sample_idx + 1
                    if current_n in sample_count_set:
                        mean_probs = running_prob_sum / float(current_n)
                        if args.prediction_mode == "argmax":
                            ensemble_pred = mean_probs.argmax(dim=1)
                        elif args.prediction_mode == "sample":
                            if running_votes is None:
                                raise RuntimeError(
                                    "running_votes is required when "
                                    "prediction_mode='sample'"
                                )
                            ensemble_pred = running_votes.argmax(dim=1).long()
                        else:
                            raise ValueError(
                                "unsupported prediction mode: "
                                f"{args.prediction_mode}"
                            )
                        metrics[current_n].update(ensemble_pred, gt_mask)

                        if args.save_all_predictions:
                            save_batch_masks(
                                ensemble_pred,
                                pred_dir / f"{aggregation_tag}_{current_n:02d}",
                                global_start,
                            )
                            if args.save_entropy:
                                save_batch_entropy(
                                    predictive_entropy(mean_probs),
                                    pred_dir / f"entropy_probmean_{current_n:02d}",
                                    global_start,
                                )

                        if take_vis > 0:
                            visual_ensemble_preds[current_n] = (
                                ensemble_pred[:take_vis].detach().cpu()
                            )
                            if args.save_entropy:
                                visual_entropy_maps[current_n] = (
                                    predictive_entropy(mean_probs)[:take_vis].detach().cpu()
                                )

            if take_vis > 0:
                for i in range(take_vis):
                    save_path = vis_dir / f"{args.split}_batch{batch_idx:04d}_idx{i:02d}.png"
                    save_multisample_grid(
                        img=img[i].detach().cpu(),
                        gt=gt_mask[i].detach().cpu(),
                        sample_preds=[preds[i] for preds in visual_sample_preds],
                        ensemble_preds={n: preds[i] for n, preds in visual_ensemble_preds.items()},
                        entropy_maps={n: ent[i] for n, ent in visual_entropy_maps.items()},
                        sample_counts=sample_counts,
                        aggregation_tag=aggregation_tag,
                        save_path=save_path,
                        imagenet_normalize=getattr(train_args, "imagenet_normalize", False),
                        save_entropy=args.save_entropy,
                    )
                    if (
                        wandb is not None
                        and args.wandb_log_images
                        and len(wandb_images) < args.wandb_num_images
                    ):
                        wandb_images.append(
                            wandb.Image(
                                str(save_path),
                                caption=f"{args.split}/batch_{batch_idx}/idx_{i}",
                            )
                        )
                visualized += take_vis

        results = {}
        for n in sample_counts:
            result = metrics[n].compute()
            result.update(
                metric_metadata(
                    args=args,
                    train_args=train_args,
                    ckpt_path=ckpt_path,
                    eval_image_size=eval_image_size,
                    samples=n,
                )
            )
            result["probability_checks"] = probability_checks
            results[n] = result
            save_json(
                save_dir / f"metrics_samples{n}_{aggregation_tag}.json",
                result,
            )

        summary = {
            "aggregation": aggregation_name,
            "prediction_mode": args.prediction_mode,
            "vote_tie_break": vote_tie_break,
            "entropy_source": entropy_source,
            "sample_counts": sample_counts,
            "max_samples_generated_per_image": max_samples,
            "split": args.split,
            "num_steps": args.num_steps,
            "checkpoint": str(ckpt_path),
            "image_size": list(eval_image_size),
            "seed": int(args.seed),
            "probability_checks": probability_checks,
            "results": {f"samples{n}": results[n] for n in sample_counts},
        }
        save_json(save_dir / "metrics_summary.json", summary)

        eval_config = {
            "aggregation": aggregation_name,
            "prediction_mode": args.prediction_mode,
            "vote_tie_break": vote_tie_break,
            "entropy_source": entropy_source,
            "eval_args": to_jsonable(vars(args)),
            "train_args": to_jsonable(vars(train_args)),
            "sample_counts": sample_counts,
            "max_samples_generated_per_image": max_samples,
            "checkpoint": str(ckpt_path),
            "save_dir": str(save_dir),
            "cfm_sample_return": {
                "return_intermediates_true": "[T,B,H,W] hard class indices from x.argmax(dim=1).cpu()",
                "return_intermediates_false": "[B,H,W] hard class indices from x.argmax(dim=1)",
                "probability_tensor_used_here": (
                    "final-step pi = softmax(logits, dim=1), shape [B,K,H,W]; "
                    "at t_next=1.0, CFM flow_map makes x equal to this pi."
                ),
            },
            "probability_checks": probability_checks,
        }
        save_json(save_dir / "eval_config.json", eval_config)

        with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"aggregation : {aggregation_name}\n")
            f.write(f"prediction_mode: {args.prediction_mode}\n")
            f.write(
                "vote_tie_break: "
                f"{vote_tie_break if vote_tie_break is not None else 'not_applicable'}\n"
            )
            f.write(f"entropy_source: {entropy_source}\n")
            f.write("evaluation : Cityscapes 19 classes\n")
            f.write("ignore_index: 19 (void)\n")
            f.write(f"split       : {args.split}\n")
            f.write(f"num_steps   : {args.num_steps}\n")
            f.write(f"samples     : {','.join(str(n) for n in sample_counts)}\n")
            f.write(f"checkpoint  : {ckpt_path}\n")
            f.write(f"image_size  : {eval_image_size[0]}x{eval_image_size[1]}\n")
            f.write(f"seed        : {args.seed}\n")
            for n in sample_counts:
                f.write("\n")
                f.write(f"[samples={n}]\n")
                f.write(f"pixel_acc   : {results[n]['pixel_acc']:.6f}\n")
                f.write(f"mIoU        : {results[n]['mIoU']:.6f}\n")
                f.write(f"mAcc        : {results[n]['mAcc']:.6f}\n")

        if wandb is not None:
            log_payload = {
                "eval_multisample/num_steps": args.num_steps,
                "eval_multisample/max_samples": max_samples,
            }
            for n in sample_counts:
                log_payload[f"eval_samples{n}/pixel_acc"] = results[n]["pixel_acc"]
                log_payload[f"eval_samples{n}/mIoU"] = results[n]["mIoU"]
                log_payload[f"eval_samples{n}/mAcc"] = results[n]["mAcc"]
            if args.wandb_log_images and wandb_images:
                log_payload["eval_multisample/images"] = wandb_images
            wandb.log(log_payload)

        print("Multisample evaluation finished.")
        print(f"aggregation: {aggregation_name}")
        print(f"prediction_mode: {args.prediction_mode}")
        print(
            "vote_tie_break: "
            f"{vote_tie_break if vote_tie_break is not None else 'not_applicable'}"
        )
        print(f"entropy_source: {entropy_source}")
        for n in sample_counts:
            print(
                f"samples={n}: "
                f"pixel_acc={results[n]['pixel_acc']:.6f}, "
                f"mIoU={results[n]['mIoU']:.6f}, "
                f"mAcc={results[n]['mAcc']:.6f}"
            )
        print(f"saved to : {save_dir}")
    finally:
        if wandb is not None:
            wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="segdiff_final.pth")
    parser.add_argument("--num_steps", type=positive_int, default=50)
    parser.add_argument("--batch_size", type=positive_int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_visualize", type=int, default=16)
    parser.add_argument("--num_snap_points", type=int, default=5)
    parser.add_argument("--source_fixed_std", type=float, default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument(
        "--cfg_null_condition",
        choices=["zero", "learned"],
        default=None,
    )
    parser.add_argument(
        "--eval_image_size",
        type=positive_int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Evaluation image size as H W. If not set, use train_args.image_size.",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        nargs="+",
        default=[1, 5, 10, 16],
        help="Positive sample counts. Duplicates are removed and values are sorted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prediction_mode",
        type=str,
        choices=["argmax", "sample"],
        default="argmax",
        help=(
            "Method used to convert class probabilities into segmentation masks. "
            "'argmax' selects the most probable class, while 'sample' draws from "
            "the per-pixel categorical distribution."
        ),
    )
    parser.add_argument("--save_all_predictions", action="store_true")
    parser.add_argument("--max_val_batches", type=positive_int, default=None)
    parser.add_argument("--save_entropy", action="store_true")

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
