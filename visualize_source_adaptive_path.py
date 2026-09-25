import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import Cityscapes20ClassDataset
from eval import load_train_config
from main import build_source_net
from visualization import colorize_mask, image_to_numpy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_TIMES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]


def average_rank(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.full_like(values, 0.5)

    order = torch.argsort(values, stable=True)
    sorted_values = values[order]

    _, counts = torch.unique_consecutive(
        sorted_values,
        return_counts=True,
    )

    ends = counts.cumsum(0)
    starts = ends - counts

    avg = (starts + ends - 1).to(values.dtype) * 0.5
    sorted_ranks = torch.repeat_interleave(avg, counts)

    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks

    return ranks / float(values.numel() - 1)


def rank_normalize_entropy(
    entropy: torch.Tensor,
) -> torch.Tensor:
    """
    entropy: [B,H,W]
    return: difficulty d in [-1,1]
    """
    output = torch.zeros_like(entropy, dtype=torch.float32)

    for b in range(entropy.shape[0]):
        values = entropy[b].float().reshape(-1)

        d = 2.0 * average_rank(values) - 1.0
        d = d - d.mean()

        output[b] = d.reshape_as(entropy[b]).clamp(-1.0, 1.0)

    return output


def source_entropy(mu: torch.Tensor) -> torch.Tensor:
    """
    H(softmax(mu))
    mu: [B,C,H,W]
    """
    probs = torch.softmax(mu.float(), dim=1)

    return -(
        probs * probs.clamp_min(1e-8).log()
    ).sum(dim=1)


def adaptive_lambda(
    t: float,
    difficulty: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    lambda(t,d) = t ** exp(beta*d)

    return [B,H,W]
    """
    exponent = torch.exp(
        float(beta) * difficulty.float()
    )

    return torch.tensor(
        float(t),
        device=difficulty.device,
        dtype=torch.float32,
    ).pow(exponent)


def adaptive_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: float,
    difficulty: torch.Tensor,
    beta: float,
):
    lam = adaptive_lambda(
        t,
        difficulty,
        beta,
    )

    lam_state = lam[:, None].to(x0.dtype)

    xt = (
        (1.0 - lam_state) * x0
        + lam_state * x1
    )

    return xt, lam


def state_to_mask(x: torch.Tensor) -> torch.Tensor:
    return x.argmax(dim=1)


def save_trajectory(
    path: Path,
    image: torch.Tensor,
    gt: torch.Tensor,
    mu: torch.Tensor,
    x0: torch.Tensor,
    states: list[torch.Tensor],
    times: list[float],
    imagenet_normalize: bool,
):
    panels = [
        (
            "Input",
            image_to_numpy(
                image.cpu(),
                imagenet_normalize=imagenet_normalize,
            ),
            True,
        ),
        ("GT", gt.cpu(), False),
        (
            "Source argmax(mu)",
            mu.argmax(dim=1)[0].cpu(),
            False,
        ),
        (
            "x0 = mu + sigma eps",
            x0.argmax(dim=1)[0].cpu(),
            False,
        ),
    ]

    for t, xt in zip(times, states):
        if t in {0.0, 1.0}:
            continue

        panels.append(
            (
                f"t={t:g}",
                xt.argmax(dim=1)[0].cpu(),
                False,
            )
        )

    columns = 4
    rows = math.ceil(len(panels) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
    )

    axes = np.asarray(axes).reshape(-1)

    for ax, (title, value, rgb) in zip(
        axes,
        panels,
        strict=False,
    ):
        if rgb:
            ax.imshow(value)
        else:
            ax.imshow(colorize_mask(value))

        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_scheduler(
    path: Path,
    image: torch.Tensor,
    mu: torch.Tensor,
    entropy: torch.Tensor,
    difficulty: torch.Tensor,
    lambda_maps: list[torch.Tensor],
    times: list[float],
    imagenet_normalize: bool,
):
    panels = 4 + len(lambda_maps)

    columns = 4
    rows = math.ceil(panels / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
    )

    axes = np.asarray(axes).reshape(-1)

    axes[0].imshow(
        image_to_numpy(
            image.cpu(),
            imagenet_normalize=imagenet_normalize,
        )
    )
    axes[0].set_title("Input")

    axes[1].imshow(
        colorize_mask(
            mu.argmax(dim=1)[0].cpu()
        )
    )
    axes[1].set_title("Source")

    p = axes[2].imshow(
        entropy[0].cpu(),
        cmap="viridis",
    )
    axes[2].set_title("Entropy H")
    fig.colorbar(
        p,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    p = axes[3].imshow(
        difficulty[0].cpu(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[3].set_title("Difficulty d")

    fig.colorbar(
        p,
        ax=axes[3],
        fraction=0.046,
        pad=0.04,
    )

    for i, (t, lam) in enumerate(
        zip(times, lambda_maps),
        start=4,
    ):
        p = axes[i].imshow(
            lam[0].cpu(),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        axes[i].set_title(
            f"lambda(t={t:g})\n"
            f"mean={lam.mean().item():.3f}"
        )

        fig.colorbar(
            p,
            ax=axes[i],
            fraction=0.046,
            pad=0.04,
        )

    for ax in axes:
        ax.axis("off")

    for ax in axes[panels:]:
        ax.set_visible(False)

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


@torch.no_grad()
def main(args):
    checkpoint_path = Path(args.checkpoint)

    ckpt = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    result_dir = checkpoint_path.parent

    train_args = load_train_config(
        result_dir,
        ckpt,
    )

    source_net = build_source_net(
        train_args,
        DEVICE,
    )

    if source_net is None:
        raise RuntimeError(
            "Checkpoint does not use image_gaussian source."
        )

    if "source_net" not in ckpt:
        raise RuntimeError(
            "Checkpoint does not contain source_net."
        )

    source_net.load_state_dict(
        ckpt["source_net"],
        strict=True,
    )

    source_net.eval()

    dataset = Cityscapes20ClassDataset(
        root=train_args.root,
        split=args.split,
        mode="fine",
        image_size=tuple(train_args.image_size),
        augment=False,
        color_jitter=False,
        imagenet_normalize=getattr(
            train_args,
            "imagenet_normalize",
            False,
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    times = args.times

    for index in args.indices:
        image, x1, gt = dataset[index]

        image = image[None].to(DEVICE)
        x1 = x1[None].to(DEVICE)
        gt = gt.to(DEVICE)

        source_out = source_net(image)

        if (
            isinstance(source_out, (tuple, list))
            and len(source_out) == 3
        ):
            _, mu, logvar = source_out
        elif (
            isinstance(source_out, (tuple, list))
            and len(source_out) == 2
        ):
            mu, logvar = source_out
        else:
            raise RuntimeError(
                "Unexpected source output."
            )

        if mu.shape[-2:] != x1.shape[-2:]:
            mu = F.interpolate(
                mu.float(),
                size=x1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            logvar = F.interpolate(
                logvar.float(),
                size=x1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        #
        # IMPORTANT:
        # Use exactly the trained CFM source:
        #
        # x0 = mu + sigma * epsilon
        #

        source_module = getattr(
            source_net,
            "module",
            source_net,
        )

        fixed_std = getattr(
            source_module,
            "fixed_std",
            None,
        )

        torch.manual_seed(
            args.seed + index
        )

        epsilon = torch.randn_like(mu)

        if fixed_std is not None:
            sigma = float(fixed_std)
        else:
            sigma = torch.exp(
                0.5 * logvar
            )

        x0 = mu + sigma * epsilon

        entropy = source_entropy(mu)

        difficulty = rank_normalize_entropy(
            entropy
        )

        states = []
        lambda_maps = []

        for t in times:
            xt, lam = adaptive_path(
                x0,
                x1,
                t,
                difficulty,
                args.beta,
            )

            states.append(xt)
            lambda_maps.append(lam)

        #
        # sanity checks
        #
        assert torch.allclose(
            states[0],
            x0,
            atol=1e-5,
            rtol=1e-5,
        )

        assert torch.allclose(
            states[-1],
            x1,
            atol=1e-5,
            rtol=1e-5,
        )

        sample_dir = (
            output_dir
            / f"sample_{index:04d}"
        )

        save_trajectory(
            sample_dir / "trajectory.png",
            image[0],
            gt,
            mu,
            x0,
            states,
            times,
            getattr(
                train_args,
                "imagenet_normalize",
                False,
            ),
        )

        scheduler_times = []
        scheduler_maps = []

        for t, lam in zip(
            times,
            lambda_maps,
        ):
            if t in {0.0, 1.0}:
                continue

            scheduler_times.append(t)
            scheduler_maps.append(lam)

        save_scheduler(
            sample_dir / "scheduler.png",
            image[0],
            mu,
            entropy,
            difficulty,
            scheduler_maps,
            scheduler_times,
            getattr(
                train_args,
                "imagenet_normalize",
                False,
            ),
        )

        print(
            f"[{index}] "
            f"mu={tuple(mu.shape)} "
            f"x0={tuple(x0.shape)} "
            f"x1={tuple(x1.shape)} "
            f"sigma="
            f"{fixed_std if fixed_std is not None else 'learned'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="val",
    )

    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 10, 25, 100],
    )

    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=DEFAULT_TIMES,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    main(parser.parse_args())