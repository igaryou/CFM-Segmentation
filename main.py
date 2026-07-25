import argparse
import copy
import json
import math
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CFM import CategoricalFlowMaps
from dataset import Cityscapes20ClassDataset
from model import SegDiffModel, SegFormerSourceGenerator
from model_segformer import SegDiffSegFormerModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HISTORY_KEYS = [
    "loss",
    "loss_base",
    "inf",
    "distill",
    "ce_ec",
    "td",
    "loss_var",
    "loss_align",
    "weighted_var",
    "weighted_align",
]
PRIOR_LOG_KEYS = [
    "mu_abs",
    "mu_min",
    "mu_max",
    "logvar_mean",
    "sigma_mean",
    "x0_abs",
]
RESUME_OVERRIDE_KEYS = {
    "result_dir",
    "resume",
    "extra_epochs",
    "lr_mid_epoch",
    "lr_mid_min",
    "batch_size",
    "grad_accum_steps",
    "resume_eta_min",
    "num_workers",
    "use_wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_name",
    "wandb_group",
    "wandb_mode",
    "wandb_tags",
    "wandb_log_interval",
    "wandb_log_images",
    "wandb_num_images",
    "val_eval_epochs",
    "val_eval_split",
    "val_eval_num_steps",
    "val_eval_batch_size",
}


def parse_int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def parse_int_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, set):
        return {int(v) for v in value}
    if isinstance(value, (tuple, list)):
        return {int(v) for v in value}
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "off", "false"}:
        return set()
    return {int(v.strip()) for v in text.split(",") if v.strip()}


def parse_wandb_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def get_amp_dtype(args: argparse.Namespace):
    if args.amp_dtype == "bf16":
        return torch.bfloat16
    if args.amp_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unknown amp_dtype: {args.amp_dtype}")


def autocast_context(args: argparse.Namespace):
    if not args.amp or DEVICE != "cuda":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=get_amp_dtype(args),
    )


def build_grad_scaler(args: argparse.Namespace):
    use_scaler = args.amp and args.amp_dtype == "fp16" and DEVICE == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    return torch.cuda.amp.GradScaler(enabled=use_scaler)


def load_model_state_dict_compat(model: torch.nn.Module, state_dict) -> bool:
    current_state = model.state_dict()
    missing = sorted(set(current_state.keys()) - set(state_dict.keys()))
    unexpected = sorted(set(state_dict.keys()) - set(current_state.keys()))
    allowed_missing = {"null_image_feat"}

    if unexpected or any(key not in allowed_missing for key in missing):
        model.load_state_dict(state_dict)
        return False

    if missing:
        warnings.warn(
            "checkpoint is missing null_image_feat; initializing it from the current model.",
            RuntimeWarning,
        )
        state_dict = state_dict.copy()
        for key in missing:
            state_dict[key] = current_state[key]

    model.load_state_dict(state_dict)
    return bool(missing)


def load_optimizer_state_dict_compat(optimizer, state_dict, model_missing_null_image_feat: bool) -> None:
    try:
        optimizer.load_state_dict(state_dict)
        return
    except ValueError:
        if not model_missing_null_image_feat:
            raise

    current_state = optimizer.state_dict()
    patched_state = copy.deepcopy(state_dict)
    saved_groups = patched_state.get("param_groups", [])
    current_groups = current_state.get("param_groups", [])
    if len(saved_groups) != len(current_groups):
        optimizer.load_state_dict(state_dict)
        return

    group_size_diffs = [
        len(current_group["params"]) - len(saved_group["params"])
        for saved_group, current_group in zip(saved_groups, current_groups)
    ]
    if group_size_diffs != [1] + [0] * (len(group_size_diffs) - 1):
        optimizer.load_state_dict(state_dict)
        return

    existing_ids = set(patched_state.get("state", {}).keys())
    for group in saved_groups:
        existing_ids.update(group.get("params", []))
    dummy_id = -1
    while dummy_id in existing_ids:
        dummy_id -= 1

    saved_groups[0]["params"].insert(0, dummy_id)
    warnings.warn(
        "optimizer checkpoint is missing null_image_feat state; loading existing states and "
        "initializing the new optimizer state lazily.",
        RuntimeWarning,
    )
    optimizer.load_state_dict(patched_state)


def apply_cfg_image_feature_dropout(
    model: torch.nn.Module,
    image_feat: torch.Tensor,
    drop_prob: float,
    training: bool,
    null_condition: str = "learned",
) -> torch.Tensor:
    if (not training) or drop_prob <= 0.0:
        return image_feat

    keep = torch.rand(image_feat.size(0), device=image_feat.device) >= drop_prob
    keep = keep.to(dtype=image_feat.dtype).view(-1, 1, 1, 1)

    null_feat = model.get_null_image_feat(
        image_feat,
        null_condition=null_condition,
    )

    return keep * image_feat + (1.0 - keep) * null_feat


def json_safe_config(args: argparse.Namespace, optimizer_summary=None) -> Dict[str, object]:
    config = vars(args).copy()
    config["image_size"] = list(args.image_size)
    if getattr(args, "crop_size", None) is not None:
        config["crop_size"] = list(args.crop_size)
    config["unet_channel_mults"] = list(args.unet_channel_mults)
    config["attn_levels"] = list(args.attn_levels)
    config["val_eval_epochs"] = sorted(args.val_eval_epochs)
    config["eta_meaning"] = "vfm_loss_weight"
    config["distill_weight"] = 1.0 - args.eta
    config["use_ccdm_aug"] = bool(args.use_ccdm_aug)
    config["imagenet_normalize"] = bool(args.imagenet_normalize)
    config["grad_accum_steps"] = int(args.grad_accum_steps)
    config["effective_batch_size"] = int(args.batch_size * args.grad_accum_steps)
    config["hflip_prob"] = args.hflip_prob
    config["color_jitter_brightness"] = args.color_jitter_brightness
    config["color_jitter_contrast"] = args.color_jitter_contrast
    config["color_jitter_saturation"] = args.color_jitter_saturation
    config["color_jitter_hue"] = args.color_jitter_hue
    config["lr_mid_epoch"] = args.lr_mid_epoch
    config["lr_mid_min"] = args.lr_mid_min
    if optimizer_summary is not None:
        config["optimizer_param_groups"] = optimizer_summary
    return config


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_resume_config(args: argparse.Namespace, ckpt: Dict[str, object], result_dir: Path) -> argparse.Namespace:
    config = load_json(result_dir / "config.json")
    config.update(ckpt.get("config", {}) or {})
    if not config:
        return args

    cli_config = vars(args).copy()
    merged = cli_config.copy()
    for key, value in config.items():
        if key in merged:
            merged[key] = value
    for key in RESUME_OVERRIDE_KEYS:
        if key in cli_config:
            merged[key] = cli_config[key]
    return argparse.Namespace(**merged)


def resolve_project_simplex(args: argparse.Namespace) -> Optional[bool]:
    if args.project_simplex and args.no_project_simplex:
        raise ValueError("--project_simplex and --no_project_simplex cannot both be set")
    if args.project_simplex:
        return True
    if args.no_project_simplex:
        return False
    return None


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.root = os.path.expanduser(args.root)
    args.image_size = parse_int_tuple(args.image_size)
    if getattr(args, "crop_size", None) is not None:
        args.crop_size = parse_int_tuple(args.crop_size)
    else:
        args.crop_size = None
    args.unet_channel_mults = parse_int_tuple(args.unet_channel_mults)
    args.attn_levels = parse_int_tuple(args.attn_levels)
    args.val_eval_epochs = parse_int_set(getattr(args, "val_eval_epochs", "150,300,450,600"))
    args.val_eval_split = getattr(args, "val_eval_split", "val")
    args.val_eval_num_steps = int(getattr(args, "val_eval_num_steps", 1))
    args.val_eval_batch_size = getattr(args, "val_eval_batch_size", None)
    if args.val_eval_batch_size is not None:
        args.val_eval_batch_size = int(args.val_eval_batch_size)
    args.use_cfg = bool(getattr(args, "use_cfg", False))
    args.cfg_drop_prob = float(getattr(args, "cfg_drop_prob", 0.1))
    args.cfg_scale = float(getattr(args, "cfg_scale", 1.0))
    args.cfg_null_condition = getattr(args, "cfg_null_condition", "learned")
    args.use_ccdm_aug = bool(getattr(args, "use_ccdm_aug", False))
    args.imagenet_normalize = bool(getattr(args, "imagenet_normalize", False))
    args.hflip_prob = float(getattr(args, "hflip_prob", 0.5))
    args.color_jitter_brightness = float(getattr(args, "color_jitter_brightness", 0.2))
    args.color_jitter_contrast = float(getattr(args, "color_jitter_contrast", 0.2))
    args.color_jitter_saturation = float(getattr(args, "color_jitter_saturation", 0.2))
    args.color_jitter_hue = float(getattr(args, "color_jitter_hue", 0.1))
    args.max_iters = getattr(args, "max_iters", None)
    if args.max_iters is not None:
        args.max_iters = int(args.max_iters)
    args.grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    args.lr_mid_epoch = getattr(args, "lr_mid_epoch", None)
    if args.lr_mid_epoch is not None:
        args.lr_mid_epoch = int(args.lr_mid_epoch)
    args.lr_mid_min = getattr(args, "lr_mid_min", None)
    if args.lr_mid_min is not None:
        args.lr_mid_min = float(args.lr_mid_min)

    if not (0.0 <= args.cfg_drop_prob < 1.0):
        raise ValueError("--cfg_drop_prob must satisfy 0.0 <= p < 1.0")
    if args.cfg_scale < 0.0:
        raise ValueError("--cfg_scale must be non-negative")
    if args.cfg_null_condition not in {"zero", "learned"}:
        raise ValueError("--cfg_null_condition must be one of: zero, learned")
    if not (0.0 <= args.hflip_prob <= 1.0):
        raise ValueError("--hflip_prob must be in [0, 1]")
    if not (0.0 <= args.color_jitter_hue <= 0.5):
        raise ValueError("--color_jitter_hue must be in [0, 0.5]")
    if args.max_iters is not None and args.max_iters <= 0:
        raise ValueError("--max_iters must be positive")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.lr_mid_epoch is not None:
        if not (args.warmup_epochs < args.lr_mid_epoch < args.epochs):
            raise ValueError("--lr_mid_epoch must satisfy warmup_epochs < lr_mid_epoch < epochs")
        if args.lr_mid_min is None:
            raise ValueError("--lr_mid_min is required when --lr_mid_epoch is set")
        if args.lr_mid_min <= 0.0:
            raise ValueError("--lr_mid_min must be > 0")
        if args.lr <= 0.0:
            raise ValueError("--lr must be > 0 when --lr_mid_epoch is set")
        if args.lr_mid_min <= args.eta_min:
            warnings.warn(
                "--lr_mid_min <= --eta_min; lr_mid_min > eta_min is recommended "
                "for a decreasing 3-stage schedule.",
                RuntimeWarning,
            )

    if not args.use_cfg:
        args.cfg_drop_prob = 0.0

    args.backbone = getattr(args, "backbone", "unet")
    if args.backbone not in {"unet", "segformer"}:
        raise ValueError(f"Unknown backbone: {args.backbone}")
    endpoint_variant = getattr(args, "endpoint_segformer_variant", "b5")
    args.endpoint_segformer_variant = endpoint_variant
    args.endpoint_decoder_channels = getattr(args, "endpoint_decoder_channels", 256)
    args.endpoint_time_emb_dim = getattr(args, "endpoint_time_emb_dim", 512)
    args.endpoint_drop_path_rate = getattr(args, "endpoint_drop_path_rate", 0.1)
    if (
        args.backbone == "segformer"
        and endpoint_variant not in {"b1", "b2", "b3", "b4", "b5"}
    ):
        raise ValueError(
            "--endpoint_segformer_variant must be one of b1,b2,b3,b4,b5 "
            "when --backbone segformer"
        )

    if args.prior_type == "image_gaussian" and args.source_backbone != "segformer":
        raise ValueError("prior_type='image_gaussian' requires --source_backbone segformer")
    if args.prior_type != "image_gaussian" and args.use_loss_align:
        warnings.warn(
            "--use_loss_align is only active with --prior_type image_gaussian; loss_align will be 0.",
            RuntimeWarning,
        )
    if args.source_learned_logvar:
        args.source_fixed_std = None
    elif args.source_fixed_std is None or args.source_fixed_std <= 0:
        raise ValueError("--source_fixed_std must be positive when --source_learned_logvar is not set")
    return args


def build_model(args: argparse.Namespace) -> SegDiffModel:
    backbone = getattr(args, "backbone", "unet")
    if backbone == "unet":
        return SegDiffModel(
            image_channels=3,
            mask_channels=args.num_classes,
            fusion_channels=args.fusion_channels,
            rrdb_blocks=args.rrdb_blocks,
            rrdb_growth_channels=args.rrdb_growth_channels,
            rrdb_blocks_mask=args.rrdb_blocks_mask,
            rrdb_growth_channels_mask=args.rrdb_growth_channels_mask,
            unet_base_channels=args.unet_base_channels,
            unet_channel_mults=args.unet_channel_mults,
            num_res_blocks=args.num_res_blocks,
            time_emb_dim=args.time_emb_dim,
            attn_levels=args.attn_levels,
            dropout=args.dropout,
            num_heads=args.num_heads,
        ).to(DEVICE)

    if backbone == "segformer":
        return SegDiffSegFormerModel(
            image_channels=3,
            mask_channels=args.num_classes,
            fusion_channels=args.fusion_channels,
            rrdb_blocks=args.rrdb_blocks,
            rrdb_growth_channels=args.rrdb_growth_channels,
            endpoint_segformer_variant=args.endpoint_segformer_variant,
            endpoint_decoder_channels=args.endpoint_decoder_channels,
            endpoint_time_emb_dim=args.endpoint_time_emb_dim,
            endpoint_drop_path_rate=args.endpoint_drop_path_rate,
        ).to(DEVICE)

    raise ValueError(f"Unknown backbone: {backbone}")


def build_source_net(args: argparse.Namespace):
    if args.prior_type != "image_gaussian":
        return None
    fixed_std = None if args.source_learned_logvar else args.source_fixed_std
    return SegFormerSourceGenerator(
        num_classes=args.num_classes,
        variant=args.source_segformer_variant,
        pretrained=args.source_pretrained,
        decoder_channels=args.source_decoder_channels,
        freeze_encoder=args.source_freeze_encoder,
        learned_logvar=args.source_learned_logvar,
        fixed_std=fixed_std,
        mu_tanh_scale=args.source_mu_tanh_scale,
    ).to(DEVICE)


def build_cfm(args: argparse.Namespace) -> CategoricalFlowMaps:
    return CategoricalFlowMaps(
        num_classes=args.num_classes,
        eps=args.eps,
        label_smoothing=args.label_smoothing,
        device=DEVICE,
        prior_type=args.prior_type,
        prior_noise_std=args.prior_noise_std,
        project_simplex=resolve_project_simplex(args),
        use_loss_align=args.use_loss_align,
        align_weight=args.align_weight,
        var_weight=args.var_weight,
        align_eps=args.align_eps,
    )


def trainable_params(*modules) -> list[torch.nn.Parameter]:
    params = []
    for module in modules:
        if module is not None:
            params.extend([p for p in module.parameters() if p.requires_grad])
    return params


def build_optimizer(args: argparse.Namespace, model: SegDiffModel, source_net):
    model_params = [p for p in model.parameters() if p.requires_grad]
    source_params = [p for p in source_net.parameters() if p.requires_grad] if source_net is not None else []

    if source_net is not None and args.source_lr is not None:
        param_groups = [
            {"params": model_params, "lr": args.lr, "name": "model"},
            {"params": source_params, "lr": args.source_lr, "name": "source_net"},
        ]
    elif source_net is not None:
        param_groups = [
            {"params": model_params + source_params, "lr": args.lr, "name": "model_source_net"},
        ]
    else:
        param_groups = [{"params": model_params, "lr": args.lr, "name": "model"}]

    optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    return optimizer_cls(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def optimizer_summary(optimizer) -> list[Dict[str, object]]:
    summary = []
    for idx, group in enumerate(optimizer.param_groups):
        n_params = sum(p.numel() for p in group["params"] if p.requires_grad)
        summary.append(
            {
                "index": idx,
                "name": group.get("name", f"group_{idx}"),
                "lr": group["lr"],
                "num_params": n_params,
            }
        )
    return summary


def build_scheduler(args: argparse.Namespace, optimizer, resume: bool):
    if resume:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.extra_epochs, 1),
            eta_min=args.resume_eta_min,
        )

    if getattr(args, "lr_mid_epoch", None) is not None:
        def lr_lambda(epoch):
            warmup = args.warmup_epochs
            mid = args.lr_mid_epoch
            total = args.epochs

            mid_factor = args.lr_mid_min / args.lr
            final_factor = args.eta_min / args.lr

            if warmup > 0 and epoch < warmup:
                return 0.1 + 0.9 * (epoch / max(warmup, 1))

            if epoch < mid:
                progress = (epoch - warmup) / max(mid - warmup, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return mid_factor + (1.0 - mid_factor) * cosine

            progress = (epoch - mid) / max(total - mid, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_factor + (mid_factor - final_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.warmup_epochs > 0 and args.epochs > args.warmup_epochs:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=args.warmup_epochs,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(args.epochs - args.warmup_epochs, 1),
                    eta_min=args.eta_min,
                ),
            ],
            milestones=[args.warmup_epochs],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.eta_min,
    )


def init_history(saved=None) -> Dict[str, list[float]]:
    history = {key: [] for key in HISTORY_KEYS}
    if isinstance(saved, dict):
        for key in HISTORY_KEYS:
            history[key] = list(saved.get(key, []))
    elif isinstance(saved, list):
        history["loss"] = list(saved)
    return history


def plot_loss_curves(history: Dict[str, list[float]], save_dir: Path, epoch: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    xs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(10, 6))
    for key in HISTORY_KEYS:
        if history[key]:
            plt.plot(xs, history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f"loss_all_{epoch + 1}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    for key in ("inf", "distill", "ce_ec", "td"):
        if history[key]:
            plt.plot(xs, history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("CFM losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f"loss_cfm_{epoch + 1}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    for key in ("loss_var", "loss_align", "weighted_var", "weighted_align"):
        if history[key]:
            plt.plot(xs, history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Aux losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f"loss_aux_{epoch + 1}.png", bbox_inches="tight")
    plt.close()


def init_wandb(args: argparse.Namespace, config: Dict[str, object]):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("--use_wandb was set, but wandb is not installed.") from exc

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        tags=parse_wandb_tags(args.wandb_tags),
        config=config,
    )
    return wandb


def tensor_item(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


class SegmentationMetrics:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confmat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.view(-1).cpu()
        target = target.view(-1).cpu()
        valid = (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]
        idx = target * self.num_classes + pred
        bins = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confmat += bins.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, object]:
        conf = self.confmat.float()
        tp = conf.diag()
        gt = conf.sum(dim=1)
        pred = conf.sum(dim=0)
        union = gt + pred - tp
        iou = tp / union.clamp_min(1.0)
        acc_cls = tp / gt.clamp_min(1.0)
        pixel_acc = tp.sum() / conf.sum().clamp_min(1.0)
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


def save_checkpoint(
    path: Path,
    model: SegDiffModel,
    source_net,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    args: argparse.Namespace,
    loss_best: float,
    history: Dict[str, list[float]],
    config: Dict[str, object],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "num_classes": args.num_classes,
        "loss_best": loss_best,
        "losses": history,
        "config": config,
    }
    if source_net is not None:
        payload["source_net"] = source_net.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


@torch.no_grad()
def evaluate_val_metrics(
    args: argparse.Namespace,
    model: torch.nn.Module,
    source_net,
    cfm: CategoricalFlowMaps,
    epoch_num: int,
    result_dir: Path,
    wandb_module=None,
) -> Dict[str, object]:
    was_training = model.training
    source_was_training = source_net.training if source_net is not None else None

    model.eval()
    if source_net is not None:
        source_net.eval()

    dataset = Cityscapes20ClassDataset(
        root=args.root,
        split=args.val_eval_split,
        mode="fine",
        image_size=tuple(args.image_size),
        augment=False,
        color_jitter=False,
        imagenet_normalize=args.imagenet_normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.val_eval_batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    metrics = SegmentationMetrics(num_classes=args.num_classes)

    for img, _, gt_mask in tqdm(loader, desc=f"val epoch {epoch_num}"):
        img = img.to(DEVICE, non_blocking=True)
        gt_mask = gt_mask.to(DEVICE, non_blocking=True)

        with autocast_context(args):
            pred = cfm.sample(
                model=model,
                img=img,
                source_net=source_net,
                num_steps=args.val_eval_num_steps,
                return_intermediates=False,
                use_cfg=args.use_cfg,
                cfg_scale=args.cfg_scale,
                cfg_null_condition=args.cfg_null_condition,
            )

        metrics.update(pred, gt_mask)

    result = metrics.compute()
    result.update(
        {
            "epoch": epoch_num,
            "split": args.val_eval_split,
            "num_steps": args.val_eval_num_steps,
            "batch_size": args.val_eval_batch_size or args.batch_size,
            "backbone": getattr(args, "backbone", "unet"),
            "prior_type": args.prior_type,
            "source_backbone": getattr(args, "source_backbone", None),
            "source_segformer_variant": getattr(args, "source_segformer_variant", None),
            "endpoint_segformer_variant": getattr(args, "endpoint_segformer_variant", None),
            "fusion_channels": getattr(args, "fusion_channels", None),
            "rrdb_blocks": getattr(args, "rrdb_blocks", None),
            "rrdb_growth_channels": getattr(args, "rrdb_growth_channels", None),
            "source_fixed_std": getattr(args, "source_fixed_std", None),
            "use_cfg": bool(args.use_cfg),
            "cfg_scale": float(args.cfg_scale),
            "cfg_drop_prob": float(getattr(args, "cfg_drop_prob", 0.0)),
            "cfg_null_condition": getattr(args, "cfg_null_condition", "learned"),
        }
    )

    save_dir = result_dir / "val_metrics" / f"epoch_{epoch_num:03d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"epoch     : {result['epoch']}\n")
        f.write(f"split     : {result['split']}\n")
        f.write(f"num_steps : {result['num_steps']}\n")
        f.write(f"backbone  : {result['backbone']}\n")
        f.write(f"pixel_acc : {result['pixel_acc']:.6f}\n")
        f.write(f"mIoU      : {result['mIoU']:.6f}\n")
        f.write(f"mAcc      : {result['mAcc']:.6f}\n")

    print(
        f"val epoch:{epoch_num} "
        f"pixel_acc:{result['pixel_acc']:.6f} "
        f"mIoU:{result['mIoU']:.6f} "
        f"mAcc:{result['mAcc']:.6f} "
        f"num_steps:{args.val_eval_num_steps}"
    )

    with open(result_dir / "val_metrics_log.txt", "a", encoding="utf-8") as f:
        f.write(
            f"epoch:{epoch_num} "
            f"pixel_acc:{result['pixel_acc']:.6f} "
            f"mIoU:{result['mIoU']:.6f} "
            f"mAcc:{result['mAcc']:.6f} "
            f"num_steps:{args.val_eval_num_steps}\n"
        )

    if wandb_module is not None:
        wandb_module.log(
            {
                "val/pixel_acc": result["pixel_acc"],
                "val/mIoU": result["mIoU"],
                "val/mAcc": result["mAcc"],
                "val/num_steps": result["num_steps"],
                "val/epoch": epoch_num,
                "epoch": epoch_num,
            }
        )

    if was_training:
        model.train()
    if source_net is not None and source_was_training:
        source_net.train()

    return result


def train(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "train_log.txt"

    ckpt = None
    if args.resume:
        ckpt_path = result_dir / "segdiff_final.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        args = merge_resume_config(args, ckpt, result_dir)

    args = normalize_args(args)
    model = build_model(args)
    source_net = build_source_net(args)
    cfm = build_cfm(args)
    optimizer = build_optimizer(args, model, source_net)
    scaler = build_grad_scaler(args)
    config = json_safe_config(args, optimizer_summary(optimizer))
    save_json(result_dir / "config.json", config)

    if ckpt is not None:
        model_missing_null_image_feat = load_model_state_dict_compat(model, ckpt["model"])
        if source_net is not None and "source_net" in ckpt:
            source_net.load_state_dict(ckpt["source_net"])
        elif source_net is not None:
            warnings.warn("source_net is configured but checkpoint has no source_net state.", RuntimeWarning)
        load_optimizer_state_dict_compat(
            optimizer,
            ckpt["optimizer"],
            model_missing_null_image_feat=model_missing_null_image_feat,
        )
        if scaler.is_enabled() and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        end_epoch = start_epoch + args.extra_epochs
        history = init_history(ckpt.get("losses", []))
        loss_best = float(ckpt.get("loss_best", float("inf")))
        print(f"Resume training from: {result_dir / 'segdiff_final.pth'}")
        print(f"start_epoch: {start_epoch + 1}")
        print(f"end_epoch  : {end_epoch}")
    else:
        start_epoch = 0
        end_epoch = args.epochs
        history = init_history()
        loss_best = float("inf")

    scheduler = build_scheduler(args, optimizer, resume=args.resume)
    wandb = init_wandb(args, config)

    dataset = Cityscapes20ClassDataset(
        root=args.root,
        split="train",
        mode="fine",
        image_size=None if args.crop_size is not None else tuple(args.image_size),
        crop_size=args.crop_size,
        augment=args.use_ccdm_aug,
        hflip_prob=args.hflip_prob,
        color_jitter=args.use_ccdm_aug,
        color_jitter_brightness=args.color_jitter_brightness,
        color_jitter_contrast=args.color_jitter_contrast,
        color_jitter_saturation=args.color_jitter_saturation,
        color_jitter_hue=args.color_jitter_hue,
        imagenet_normalize=args.imagenet_normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    global_step = 0
    optimizer_step = 0
    last_epoch = start_epoch - 1
    stop_training = False
    vis_epochs = {10, 30, 50, 70, 90, 110, 130}
    try:
        for epoch in range(start_epoch, end_epoch):
            model.train()
            if source_net is not None:
                source_net.train()

            sums = {key: 0.0 for key in HISTORY_KEYS + PRIOR_LOG_KEYS + ["x1_abs"]}
            cnt = 0

            optimizer.zero_grad(set_to_none=True)
            num_batches = len(loader)

            for batch_idx, (img, x_1, masks) in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}")):
                img = img.to(DEVICE, non_blocking=True)
                x_1 = x_1.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True).long()

                B = img.size(0)
                if B == 0:
                    continue

                group_start = (batch_idx // args.grad_accum_steps) * args.grad_accum_steps
                normal_group_end = min(group_start + args.grad_accum_steps, num_batches)
                if args.max_iters is not None:
                    global_step_at_group_start = global_step - (batch_idx - group_start)
                    max_group_end = group_start + max(args.max_iters - global_step_at_group_start, 0)
                    group_end = min(normal_group_end, max_group_end)
                else:
                    group_end = normal_group_end
                accum_steps_this_update = max(group_end - group_start, 1)
                should_step = (batch_idx + 1) == group_end
                will_stop_after_this_batch = (
                    args.max_iters is not None and (global_step + 1) >= args.max_iters
                )

                with autocast_context(args):
                    B, _, H, W = x_1.shape

                    x0, prior_stats = cfm.sample_prior(
                        B=B,
                        H=H,
                        W=W,
                        device=DEVICE,
                        dtype=x_1.dtype,
                        img=img,
                        source_net=source_net,
                        target_x1=x_1,
                        return_stats=True,
                    )
                    
                    image_feat = model.encode_image(img)
                    image_feat = apply_cfg_image_feature_dropout(
                        model=model,
                        image_feat=image_feat,
                        drop_prob=args.cfg_drop_prob if args.use_cfg else 0.0,
                        training=model.training,
                        null_condition=args.cfg_null_condition,
                    )

                    t_vfm = torch.rand(B, device=DEVICE)
                    x_t = cfm.path(x0, x_1, t_vfm)

                    loss_inf, _, _ = cfm.vfm_loss(
                        model=model,
                        x_t=x_t,
                        img=img,
                        t=t_vfm,
                        mask=masks,
                        image_feat=image_feat
                    )

                    if args.distill_loss == "psd":
                        times = torch.rand(B, 3, device=DEVICE)
                        times, _ = torch.sort(times, dim=1)

                        s = times[:, 0]
                        u = times[:, 1]
                        t = times[:, 2]

                        x_s = cfm.path(x0, x_1, s)
                        
                        
                        loss_distill, distill_stats = cfm.psd_loss(
                            model=model,
                            x_s=x_s,
                            img=img,
                            s=s,
                            u=u,
                            t=t,
                            image_feat=image_feat,
                        )

                    elif args.distill_loss == "ecld":
                        a = torch.rand(B, device=DEVICE)
                        b = torch.rand(B, device=DEVICE)
                        s = torch.minimum(a, b)
                        t = torch.maximum(a, b)

                        x_s = cfm.path(x0, x_1, s)

                        loss_distill, distill_stats = cfm.ecld_loss(
                            model=model,
                            x_s=x_s,
                            img=img,
                            s=s,
                            t=t,
                            lambda_td=args.lambda_td,
                            image_feat=image_feat,
                        )

                    else:
                        raise ValueError(f"Unknown distill_loss: {args.distill_loss}")

                    loss_inf = loss_inf.float()
                    loss_distill = loss_distill.float()

                    loss_base = (
                        args.eta * loss_inf
                        + (1.0 - args.eta) * args.lambda_distill * loss_distill
                    )

                    loss_var = prior_stats["loss_var"].float()
                    loss_align = prior_stats["loss_align"].float()
                    weighted_var = prior_stats["weighted_var"].float()
                    weighted_align = prior_stats["weighted_align"].float()

                    loss = (loss_base + weighted_var + weighted_align).float()

                loss_for_backward = loss / accum_steps_this_update

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                    if should_step:
                        if args.grad_clip is not None and args.grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                trainable_params(model, source_net),
                                max_norm=args.grad_clip,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                else:
                    loss_for_backward.backward()
                    if should_step:
                        if args.grad_clip is not None and args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                trainable_params(model, source_net),
                                max_norm=args.grad_clip,
                            )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1

                batch_metrics = {
                    "loss": tensor_item(loss),
                    "loss_base": tensor_item(loss_base),
                    "inf": tensor_item(loss_inf),
                    "distill": tensor_item(loss_distill),
                    "ce_ec": tensor_item(distill_stats["loss_ce_ec"]),
                    "td": tensor_item(distill_stats["loss_td"]),
                    "loss_var": tensor_item(loss_var),
                    "loss_align": tensor_item(loss_align),
                    "weighted_var": tensor_item(weighted_var),
                    "weighted_align": tensor_item(weighted_align),
                }

                for key in PRIOR_LOG_KEYS:
                    batch_metrics[key] = tensor_item(prior_stats[key])
                batch_metrics["x1_abs"] = tensor_item(x_1.abs().mean())

                for key, value in batch_metrics.items():
                    sums[key] += value
                cnt += 1

                if wandb is not None and args.wandb_log_interval > 0 and global_step % args.wandb_log_interval == 0:
                    wandb.log(
                        {
                            "batch/loss": batch_metrics["loss"],
                            "batch/loss_base": batch_metrics["loss_base"],
                            "batch/loss_inf": batch_metrics["inf"],
                            "batch/loss_distill": batch_metrics["distill"],
                            "batch/loss_ce_ec": batch_metrics["ce_ec"],
                            "batch/loss_td": batch_metrics["td"],
                            "batch/loss_var": batch_metrics["loss_var"],
                            "batch/loss_align": batch_metrics["loss_align"],
                            "batch/distill_loss_type": args.distill_loss,
                            "batch/grad_accum_steps": args.grad_accum_steps,
                            "batch/effective_batch_size": args.batch_size * args.grad_accum_steps,
                            "batch/accum_steps_this_update": accum_steps_this_update,
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                        }
                    )
                global_step += 1
                if will_stop_after_this_batch:
                    stop_training = True
                    break

            if cnt == 0:
                raise RuntimeError("No training batches were processed.")

            avg = {key: value / cnt for key, value in sums.items()}
            for key in HISTORY_KEYS:
                history[key].append(avg[key])

            current_lr = optimizer.param_groups[0]["lr"]
            log_line = (
                f"epoch:{epoch + 1} "
                f"loss_avg:{avg['loss']:.6f} "
                f"loss_base:{avg['loss_base']:.6f} "
                f"inf:{avg['inf']:.6f} "
                f"distill:{avg['distill']:.6f} "
                f"distill_type:{args.distill_loss} "
                f"ce_ec:{avg['ce_ec']:.6f} "
                f"td:{avg['td']:.6f} "
                f"loss_var:{avg['loss_var']:.6f} "
                f"loss_align:{avg['loss_align']:.6f} "
                f"weighted_var:{avg['weighted_var']:.6f} "
                f"weighted_align:{avg['weighted_align']:.6f} "
                f"mu_abs:{avg['mu_abs']:.6f} "
                f"mu_min:{avg['mu_min']:.6f} "
                f"mu_max:{avg['mu_max']:.6f} "
                f"logvar_mean:{avg['logvar_mean']:.6f} "
                f"sigma_mean:{avg['sigma_mean']:.6f} "
                f"x0_abs:{avg['x0_abs']:.6f} "
                f"x1_abs:{avg['x1_abs']:.6f} "
                f"grad_accum_steps:{args.grad_accum_steps} "
                f"effective_batch_size:{args.batch_size * args.grad_accum_steps} "
                f"optimizer_step:{optimizer_step} "
                f"lr:{current_lr:.8e}"
            )
            print(log_line)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": avg["loss"],
                        "train/loss_base": avg["loss_base"],
                        "train/loss_inf": avg["inf"],
                        "train/loss_distill": avg["distill"],
                        "train/distill_loss_type": args.distill_loss,
                        "train/loss_ce_ec": avg["ce_ec"],
                        "train/loss_td": avg["td"],
                        "train/loss_var": avg["loss_var"],
                        "train/loss_align": avg["loss_align"],
                        "train/weighted_var": avg["weighted_var"],
                        "train/weighted_align": avg["weighted_align"],
                        "train/lr": current_lr,
                        "train/mu_abs": avg["mu_abs"],
                        "train/mu_min": avg["mu_min"],
                        "train/mu_max": avg["mu_max"],
                        "train/logvar_mean": avg["logvar_mean"],
                        "train/sigma_mean": avg["sigma_mean"],
                        "train/x0_abs": avg["x0_abs"],
                        "train/x1_abs": avg["x1_abs"],
                        "train/grad_accum_steps": args.grad_accum_steps,
                        "train/effective_batch_size": args.batch_size * args.grad_accum_steps,
                        "epoch": epoch + 1,
                        "optimizer_step": optimizer_step,
                    }
                )

            scheduler.step()
            last_epoch = epoch

            if avg["loss"] < loss_best:
                loss_best = avg["loss"]
                save_checkpoint(
                    result_dir / "segdiff_best.pth",
                    model,
                    source_net,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    args,
                    loss_best,
                    history,
                    config,
                )

            if (epoch + 1) % 10 == 0 or (epoch + 1) == end_epoch:
                plot_loss_curves(history, result_dir / "loss_curves", epoch)
                
            epoch_num = epoch + 1
            if epoch_num in args.val_eval_epochs:
                save_checkpoint(
                    result_dir / f"segdiff_epoch_{epoch_num:03d}.pth",
                    model,
                    source_net,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    args,
                    loss_best,
                    history,
                    config,
                )
                evaluate_val_metrics(
                    args=args,
                    model=model,
                    source_net=source_net,
                    cfm=cfm,
                    epoch_num=epoch_num,
                    result_dir=result_dir,
                    wandb_module=wandb,
                )

            if epoch_num in vis_epochs:
                cfm.run_inference_examples(
                    model=model,
                    source_net=source_net,
                    root=args.root,
                    split="val",
                    batch_size=args.batch_size,
                    num_steps=1,
                    save_dir=str(result_dir / "infer_val" / f"epoch_{epoch_num:03d}"),
                    image_size=args.image_size,
                    num_workers=args.num_workers,
                    use_wandb=wandb is not None and args.wandb_log_images,
                    wandb_module=wandb,
                    wandb_prefix=f"eval_inference/epoch_{epoch_num:03d}",
                    max_images=args.wandb_num_images if args.wandb_log_images else args.batch_size,
                    use_cfg=args.use_cfg,
                    cfg_scale=args.cfg_scale,
                    cfg_null_condition=args.cfg_null_condition,
                    imagenet_normalize=args.imagenet_normalize,
                )

            if stop_training:
                break
                

        save_checkpoint(
            result_dir / "segdiff_final.pth",
            model,
            source_net,
            optimizer,
            scheduler,
            scaler,
            last_epoch,
            args,
            loss_best,
            history,
            config,
        )

        cfm.run_inference_examples(
            model=model,
            source_net=source_net,
            root=args.root,
            split="val",
            batch_size=args.batch_size,
            num_steps=1,
            save_dir=str(result_dir / "infer_val"),
            image_size=args.image_size,
            num_workers=args.num_workers,
            use_wandb=args.use_wandb and args.wandb_log_images,
            wandb_module=wandb,
            wandb_prefix="train_inference",
            max_images=args.wandb_num_images if args.wandb_log_images else args.batch_size,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            cfg_null_condition=args.cfg_null_condition,
            imagenet_normalize=args.imagenet_normalize,
        )
    finally:
        if wandb is not None:
            wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="~/datasets/cityscapes")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--image_size", nargs=2, type=int, default=(128, 256))
    parser.add_argument("--crop_size", nargs=2, type=int, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before one optimizer step.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--use_ccdm_aug",
        action="store_true",
        help="Use CCDM-style Cityscapes train augmentation: random horizontal flip + color jitter.",
    )
    parser.add_argument(
        "--imagenet_normalize",
        action="store_true",
        help="Apply ImageNet normalization to input images in train/val/eval datasets.",
    )
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--color_jitter_brightness", type=float, default=0.2)
    parser.add_argument("--color_jitter_contrast", type=float, default=0.2)
    parser.add_argument("--color_jitter_saturation", type=float, default=0.2)
    parser.add_argument("--color_jitter_hue", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra_epochs", type=int, default=50)
    parser.add_argument(
        "--val_eval_epochs",
        type=str,
        default="75,150,200,300",
        help="Comma-separated epochs for full validation metrics. Use 'none' to disable.",
    )
    parser.add_argument("--val_eval_split", type=str, default="val")
    parser.add_argument("--val_eval_num_steps", type=int, default=1)
    parser.add_argument("--val_eval_batch_size", type=int, default=None)
    parser.add_argument(
        "--backbone",
        choices=["unet", "segformer"],
        default="unet",
        help=(
            "Endpoint predictor backbone. 'unet' uses the existing SegDiffModel, "
            "'segformer' uses SegDiffSegFormerModel."
        ),
    )

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--lr_mid_epoch", type=int, default=None)
    parser.add_argument("--lr_mid_min", type=float, default=None)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--resume_eta_min", type=float, default=1e-8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--amp_ecld", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--cfg_null_condition",
        choices=["zero", "learned"],
        default="learned",
        help="Null image condition used for classifier-free guidance.",
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="Weight for VFM loss in full-batch VFM+distill training. Distill weight is 1-eta.",
    )
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--lambda_td", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--prior_type",
        choices=["dirichlet", "gaussian", "image_gaussian"],
        default="gaussian",
    )
    parser.add_argument(
        "--distill_loss",
        choices=["ecld", "psd"],
        default="psd",
    )
    parser.add_argument("--prior_noise_std", type=float, default=1.0)
    parser.add_argument("--project_simplex", action="store_true")
    parser.add_argument("--no_project_simplex", action="store_true")

    parser.add_argument("--use_loss_align", action="store_true")
    parser.add_argument("--align_weight", type=float, default=0.25)
    parser.add_argument("--var_weight", type=float, default=0.0)
    parser.add_argument("--align_eps", type=float, default=1e-8)

    parser.add_argument("--source_backbone", choices=["none", "segformer"], default="none")
    parser.add_argument(
        "--source_segformer_variant",
        choices=["b0", "b1", "b2", "b3", "b4", "b5"],
        default="b3",
    )
    parser.add_argument("--source_pretrained", action="store_true")
    parser.add_argument("--source_freeze_encoder", action="store_true")
    parser.add_argument("--source_decoder_channels", type=int, default=128)
    parser.add_argument("--source_learned_logvar", action="store_true")
    parser.add_argument("--source_fixed_std", type=float, default=1.0)
    parser.add_argument("--source_mu_tanh_scale", type=float, default=0.0)
    parser.add_argument("--source_lr", type=float, default=None)

    parser.add_argument("--fusion_channels", type=int, default=128)
    parser.add_argument("--rrdb_blocks", type=int, default=15)
    parser.add_argument("--rrdb_growth_channels", type=int, default=32)
    parser.add_argument("--rrdb_blocks_mask", type=int, default=3)
    parser.add_argument("--rrdb_growth_channels_mask", type=int, default=16)
    parser.add_argument("--unet_base_channels", type=int, default=128)
    parser.add_argument("--unet_channel_mults", type=str, default="1,2,4,8,8")
    parser.add_argument("--num_res_blocks", type=int, default=3)
    parser.add_argument("--time_emb_dim", type=int, default=512)
    parser.add_argument("--attn_levels", type=str, default="3,4")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument(
        "--endpoint_segformer_variant",
        choices=["b1", "b2", "b3", "b4", "b5"],
        default="b2",
    )
    parser.add_argument("--endpoint_decoder_channels", type=int, default=256)
    parser.add_argument("--endpoint_time_emb_dim", type=int, default=512)
    parser.add_argument("--endpoint_drop_path_rate", type=float, default=0.1)

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
    parser.add_argument("--wandb_log_interval", type=int, default=50)
    parser.add_argument("--wandb_log_images", action="store_true")
    parser.add_argument("--wandb_num_images", type=int, default=4)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
