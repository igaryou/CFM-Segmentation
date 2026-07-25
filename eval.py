#Categorical Flow Maps with Better Source

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Cityscapes20ClassDataset
from main import (
    DEVICE,
    autocast_context,
    build_cfm,
    build_model,
    build_parser as build_train_parser,
    build_source_net,
    load_model_state_dict_compat,
    load_json,
    normalize_args,
    parse_wandb_tags,
)
from visualization import save_trajectory_grid


class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 19,
    ) -> None:
        self.num_classes = num_classes  # 20
        self.ignore_index = ignore_index
        self.confmat = torch.zeros(
            num_classes,
            num_classes,
            dtype=torch.int64,
        )

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        pred = pred.reshape(-1).cpu()
        target = target.reshape(-1).cpu()

        valid = (
            (target >= 0)
            & (target < self.num_classes)
            & (target != self.ignore_index)
            & (pred >= 0)
            & (pred < self.num_classes)
        )

        pred = pred[valid]
        target = target[valid]

        idx = target * self.num_classes + pred
        bins = torch.bincount(
            idx,
            minlength=self.num_classes ** 2,
        )
        self.confmat += bins.reshape(
            self.num_classes,
            self.num_classes,
        )

    def compute(self):
        conf = self.confmat.float()

        tp = conf.diag()
        gt = conf.sum(dim=1)
        pred_count = conf.sum(dim=0)
        union = gt + pred_count - tp

        iou = tp / union.clamp_min(1.0)
        acc_cls = tp / gt.clamp_min(1.0)

        # void=19を平均対象から除外
        evaluated = torch.arange(self.num_classes) != self.ignore_index

        pixel_acc = tp.sum() / conf.sum().clamp_min(1.0)
        miou = iou[evaluated].mean()
        macc = acc_cls[evaluated].mean()

        return {
            "pixel_acc": float(pixel_acc.item()),
            "mIoU": float(miou.item()),
            "mAcc": float(macc.item()),
            "IoU_per_class": [
                float(iou[k].item())
                for k in range(self.num_classes)
                if k != self.ignore_index
            ],
            "Acc_per_class": [
                float(acc_cls[k].item())
                for k in range(self.num_classes)
                if k != self.ignore_index
            ],
            "confusion_matrix": conf.to(torch.int64).tolist(),
            "evaluated_classes": 19,
            "ignore_index": self.ignore_index,
        }


def parser_defaults(parser: argparse.ArgumentParser) -> Dict[str, object]:
    defaults = {}
    for action in parser._actions:
        if action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


def load_train_config(result_dir: Path, ckpt: Dict[str, object]) -> argparse.Namespace:
    config = parser_defaults(build_train_parser())
    config["result_dir"] = str(result_dir)
    config.update(load_json(result_dir / "config.json"))
    config.update(ckpt.get("config", {}) or {})
    config.setdefault("backbone", "unet")
    allowed = set(parser_defaults(build_train_parser()).keys())
    config = {key: value for key, value in config.items() if key in allowed}
    return normalize_args(argparse.Namespace(**config))


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


def resolve_eval_amp_args(args: argparse.Namespace, train_args: argparse.Namespace) -> argparse.Namespace:
    if args.amp is None:
        args.amp = bool(getattr(train_args, "amp", False))
        args.amp_dtype = getattr(train_args, "amp_dtype", "bf16")
    return args


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
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

    source_std = getattr(train_args, "source_fixed_std", None)

    if source_std is None:
        std_tag = "std_learned"
    else:
        std_tag = f"std{str(source_std)}"

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
    
    if args.eval_image_size is None:
        eval_image_size = tuple(train_args.image_size)
    else:
        eval_image_size = tuple(args.eval_image_size)
    
    size_tag = f"size{eval_image_size[0]}x{eval_image_size[1]}"

    save_dir = (
        result_dir
        / f"eval_{args.split}_{args.num_steps}steps_{backbone_tag}_{endpoint_tag}_{std_tag}_{cfg_tag}_{size_tag}"
    )
    vis_dir = save_dir / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

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
    wandb = init_wandb(args, wandb_config)

    metrics = SegmentationMetrics(num_classes=train_args.num_classes)
    visualized = 0
    wandb_images = []

    try:
        for batch_idx, (img, _, gt_mask) in enumerate(tqdm(loader, desc=f"eval:{args.split}")):
            img = img.to(DEVICE, non_blocking=True)
            gt_mask = gt_mask.to(DEVICE, non_blocking=True)

            with autocast_context(args):
                traj = cfm.sample(
                    model=model,
                    img=img,
                    source_net=source_net,
                    num_steps=args.num_steps,
                    return_intermediates=True,
                    use_cfg=args.use_cfg,
                    cfg_scale=args.cfg_scale,
                    cfg_null_condition=args.cfg_null_condition,
                )
            pred = traj[-1]
            metrics.update(pred, gt_mask)

            remaining = max(0, args.num_visualize - visualized)
            if remaining > 0:
                take = min(remaining, img.size(0))
                for i in range(take):
                    save_path = vis_dir / f"{args.split}_batch{batch_idx:04d}_idx{i:02d}.png"
                    save_trajectory_grid(
                        img=img[i].cpu(),
                        gt=gt_mask[i].cpu(),
                        traj=traj[:, i].cpu(),
                        save_path=save_path,
                        num_snap_points=args.num_snap_points,
                        imagenet_normalize=getattr(train_args, "imagenet_normalize", False),
                    )
                    if (
                        wandb is not None
                        and args.wandb_log_images
                        and len(wandb_images) < args.wandb_num_images
                    ):
                        wandb_images.append(
                            wandb.Image(
                                str(save_path),
                                caption=f"{args.split}/batch_{batch_idx}/sample_{i}",
                            )
                        )
                visualized += take

        result = metrics.compute()
        result.update(
            {
                "split": args.split,
                "num_steps": args.num_steps,
                "batch_size": args.batch_size,
                "num_visualize": args.num_visualize,
                "num_snap_points": args.num_snap_points,
                "checkpoint": str(ckpt_path),
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
            }
        )

        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"split      : {result['split']}\n")
            f.write(f"num_steps  : {result['num_steps']}\n")
            f.write(f"backbone   : {result['backbone']}\n")
            f.write(f"source     : {result['source_backbone']}-{result['source_segformer_variant']}\n")
            if result["backbone"] == "segformer":
                f.write(f"endpoint   : SegFormer-{result['endpoint_segformer_variant']}\n")
            f.write(f"pixel_acc  : {result['pixel_acc']:.6f}\n")
            f.write(f"mIoU       : {result['mIoU']:.6f}\n")
            f.write(f"mAcc       : {result['mAcc']:.6f}\n")

        if wandb is not None:
            log_payload = {
                "eval/pixel_acc": result["pixel_acc"],
                "eval/mIoU": result["mIoU"],
                "eval/mAcc": result["mAcc"],
                "eval/num_steps": args.num_steps,
            }
            if args.wandb_log_images and wandb_images:
                log_payload["eval/images"] = wandb_images
            wandb.log(log_payload)

        print("Evaluation finished.")
        print(f"pixel_acc: {result['pixel_acc']:.6f}")
        print(f"mIoU     : {result['mIoU']:.6f}")
        print(f"mAcc     : {result['mAcc']:.6f}")
        print(f"saved to : {save_dir}")
    finally:
        if wandb is not None:
            wandb.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="segdiff_final.pth")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
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
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Evaluation image size as H W. If not set, use train_args.image_size.",
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
