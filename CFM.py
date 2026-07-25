import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.func import jvp
from torch.utils.data import DataLoader

from dataset import Cityscapes20ClassDataset
from visualization import save_prediction_triplet


class CategoricalFlowMaps:
    def __init__(
        self,
        num_classes: int = 20,
        eps: float = 1e-5,
        label_smoothing: float = 0.1,
        device="cuda",
        prior_type: str = "gaussian",
        prior_noise_std: float = 1.0,
        project_simplex: Optional[bool] = None,
        use_loss_align: bool = False,
        align_weight: float = 0.25,
        var_weight: float = 0.0,
        align_eps: float = 1e-8,
    ):
        if prior_type not in {"dirichlet", "gaussian", "image_gaussian"}:
            raise ValueError(f"Unknown prior_type: {prior_type}")
        if prior_noise_std <= 0:
            raise ValueError("prior_noise_std must be positive")

        self.num_classes = num_classes
        self.eps = eps
        self.label_smoothing = label_smoothing
        self.device = device
        self.prior_type = prior_type
        self.prior_noise_std = prior_noise_std
        self.project_simplex = prior_type == "dirichlet" if project_simplex is None else bool(project_simplex)
        self.use_loss_align = use_loss_align
        self.align_weight = align_weight
        self.var_weight = var_weight
        self.align_eps = align_eps

    def _zero_prior_stats(self, device, dtype, x0: Optional[torch.Tensor] = None):
        zero = torch.zeros((), device=device, dtype=dtype)
        x0_abs = x0.abs().mean() if x0 is not None else zero
        return {
            "loss_var": zero,
            "loss_align": zero,
            "weighted_var": zero,
            "weighted_align": zero,
            "mu_abs": zero,
            "mu_min": zero,
            "mu_max": zero,
            "logvar_mean": zero,
            "logvar_min": zero,
            "logvar_max": zero,
            "sigma_mean": zero,
            "sigma_min": zero,
            "sigma_max": zero,
            "x0_abs": x0_abs,
        }

    def sample_prior(
        self,
        B: int,
        H: int,
        W: int,
        device,
        dtype,
        img: Optional[torch.Tensor] = None,
        source_net: Optional[torch.nn.Module] = None,
        target_x1: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        if self.prior_type == "dirichlet":
            dist = torch.distributions.Dirichlet(
                torch.ones(self.num_classes, device=device, dtype=dtype)
            )
            x0 = dist.sample((B, H, W)).permute(0, 3, 1, 2).contiguous()
            if return_stats:
                return x0, self._zero_prior_stats(device, dtype, x0=x0)
            return x0

        if self.prior_type == "gaussian":
            x0 = torch.randn(B, self.num_classes, H, W, device=device, dtype=dtype)
            if return_stats:
                return x0, self._zero_prior_stats(device, dtype, x0=x0)
            return x0

        if source_net is None:
            raise RuntimeError("prior_type='image_gaussian' requires source_net.")
        if img is None:
            raise RuntimeError("prior_type='image_gaussian' requires img.")
        if self.use_loss_align and target_x1 is None and return_stats:
            raise RuntimeError("use_loss_align=True requires target_x1 for image_gaussian prior.")

        source_out = source_net(img)
        if isinstance(source_out, (tuple, list)) and len(source_out) == 3:
            _, mu, logvar = source_out
        elif isinstance(source_out, (tuple, list)) and len(source_out) == 2:
            mu, logvar = source_out
        else:
            raise RuntimeError("source_net(img) must return (x0, mu, logvar) or (mu, logvar).")

        mu = mu.to(device=device, dtype=dtype)
        logvar = logvar.to(device=device, dtype=dtype)
        if mu.shape[-2:] != (H, W):
            mu = F.interpolate(mu, size=(H, W), mode="bilinear", align_corners=False)
        if logvar.shape[-2:] != (H, W):
            logvar = F.interpolate(logvar, size=(H, W), mode="bilinear", align_corners=False)
        if mu.shape[1] != self.num_classes:
            raise RuntimeError(
                f"source_net mu has {mu.shape[1]} channels, expected {self.num_classes}."
            )

        zero = torch.zeros((), device=device, dtype=dtype)
        source_module = getattr(source_net, "module", source_net)
        fixed_std = getattr(source_module, "fixed_std", None)
        if fixed_std is not None:
            std = float(fixed_std)
            eps = torch.randn_like(mu)
            sigma = torch.full_like(mu, std)
            logvar = torch.full_like(mu, math.log(std ** 2))
            x0 = mu + sigma * eps
            loss_var = zero
        else:
            sigma = torch.exp(0.5 * logvar)
            eps = torch.randn_like(mu)
            x0 = mu + sigma * eps
            loss_var = 0.5 * torch.mean(torch.exp(logvar) - logvar - 1.0)

        if target_x1 is not None:
            target_x1 = target_x1.to(device=device, dtype=dtype)
            if target_x1.shape[-2:] != (H, W):
                target_x1 = F.interpolate(target_x1, size=(H, W), mode="nearest")
            if target_x1.shape[1] != self.num_classes:
                raise RuntimeError(
                    f"target_x1 has {target_x1.shape[1]} channels, expected {self.num_classes}."
                )
            mu_n = F.normalize(mu, p=2, dim=1, eps=self.align_eps)
            x1_n = F.normalize(target_x1, p=2, dim=1, eps=self.align_eps)
            loss_align = F.mse_loss(mu_n, x1_n)
        else:
            loss_align = zero

        stats = {
            "loss_var": loss_var,
            "loss_align": loss_align,
            "weighted_var": self.var_weight * loss_var,
            "weighted_align": self.align_weight * loss_align if self.use_loss_align else zero ,
            "mu_abs": mu.abs().mean(),
            "mu_min": mu.min(),
            "mu_max": mu.max(),
            "logvar_mean": logvar.mean(),
            "logvar_min": logvar.min(),
            "logvar_max": logvar.max(),
            "sigma_mean": sigma.mean(),
            "sigma_min": sigma.min(),
            "sigma_max": sigma.max(),
            "x0_abs": x0.abs().mean(),
        }
        if return_stats:
            return x0, stats
        return x0

    def path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        """
        x_t = (1 - t) x_0 + t x_1
        x0, x1: [B, K, H, W]
        t: [B]
        """
        t_view = t[:, None, None, None]
        return (1.0 - t_view) * x0 + t_view * x1

    def gamma(self, s: torch.Tensor, t: torch.Tensor):
        """
        gamma = (t - s) / (1 - s)
        """
        return (t - s) / (1.0 - s).clamp_min(self.eps)

    def flow_map(self, x_s: torch.Tensor, pi_st: torch.Tensor, s: torch.Tensor, t: torch.Tensor):
        """
        X_{s,t}(x_s) = x_s + gamma (pi_{s,t}(x_s) - x_s)
        """
        gamma = self.gamma(s, t)[:, None, None, None]
        return x_s + gamma * (pi_st - x_s)

    def vfm_loss(
        self,
        model,
        x_t: torch.Tensor,
        img: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        image_feat=None
    ):
        """
        diagonal s=t の L_inf。
        model(x_t, img, t, t) -> logits_tt, pi_tt
        mask: [B, H, W]
        """
        if image_feat is None:
            logits_tt, pi_tt = model(x_t, img, t, t)
            
        else:
            logits_tt, pi_tt = model.forward_with_image_feat(
                x_t,
                image_feat,
                t,
                t,
            )

        loss = F.cross_entropy(
                logits_tt,
                mask,
                label_smoothing=self.label_smoothing,
            )
        return loss, logits_tt, pi_tt

    def ecld_loss(
        self,
        model,
        x_s: torch.Tensor,
        img: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        lambda_td: float = 1.0,
        image_feat=None,
    ):
        """
        ECLD = 4 * CE-EC + 2 * TD
        """
        dtdt = torch.ones_like(t)

        def logits_fn(t_in):
            if image_feat is None:
                return model.forward_logits(x_s, img, s, t_in)
            return model.forward_logits_with_image_feat(x_s, image_feat, s, t_in)

        logits_st, dlogits_dt = jvp(
            logits_fn,
            (t,),
            (dtdt,),
        )
        pi_st = torch.softmax(logits_st, dim=1)

        x_st = self.flow_map(x_s, pi_st, s, t)

        with torch.no_grad():
            if image_feat is None:
                _, pi_tgt = model(x_st, img, t, t)
            else:
                _, pi_tgt = model.forward_with_image_feat(
                    x_st,
                    image_feat.detach(),
                    t,
                    t,
                )

        log_pi_st = F.log_softmax(logits_st, dim=1)
        loss_ce_ec = -(pi_tgt * log_pi_st).sum(dim=1).mean()

        inner = (pi_st * dlogits_dt).sum(dim=1, keepdim=True)
        dpi_dt = pi_st * (dlogits_dt - inner)

        gamma = self.gamma(s, t)[:, None, None, None]
        loss_td = (gamma * dpi_dt.pow(2)).sum(dim=1).mean()

        loss_ecld = 4.0 * loss_ce_ec + 2.0 * lambda_td * loss_td

        stats = {
            "loss_ecld": loss_ecld.detach(),
            "loss_ce_ec": loss_ce_ec.detach(),
            "loss_td": loss_td.detach(),
        }

        return loss_ecld, stats
    
    def psd_lambda(self, s: torch.Tensor, u: torch.Tensor, t: torch.Tensor, eps: float = 1e-5):
        """
        flow_map gamma=(t-s)/(1-s) と整合する PSD teacher の混合係数．

        pi_st = lam * pi_su + (1 - lam) * pi_ut
        lam = (1-t)(u-s) / ((1-u)(t-s)).

        これは X_{s,t}=X_{u,t} o X_{s,u} に現在の endpoint
        parametrisation を代入して pi_{s,t} について解いた係数である．
        """
        num = (1.0 - t) * (u - s)
        den = (1.0 - u).clamp_min(eps) * (t - s).clamp_min(eps)
        lam = num / den
        return lam[:, None, None, None].clamp(0.0, 1.0)

    def psd_loss(self, model, x_s, img, s, u, t, eps: float = 1e-5, image_feat=None):
        """
        Posterior Self-Distillation loss.

        s < u < t として，
          1. pi_su で x_s -> x_u を作る
          2. x_u から pi_ut を予測する
          3. pi_su と pi_ut の semigroup-consistent な混合を teacher にする
          4. 直接予測 pi_st を teacher に合わせる

        後方互換のため，既存実装どおり全画素・全20クラス上の soft-target
        cross entropy CE(target, pi_st) を使う．Discrete Flow Maps の forward
        KL(target || pi_st) とは teacher entropy の定数分だけ異なるが，detach
        した student 勾配は同じである．
        """
        def model_forward(x_in, s_in, t_in, feat):
            if feat is None:
                return model(x_in, img, s_in, t_in)
            return model.forward_with_image_feat(x_in, feat, s_in, t_in)
        
        teacher_feat = image_feat.detach() if image_feat is not None else None
        with torch.no_grad():
            _, pi_su = model_forward(x_s, s, u, teacher_feat)
            x_su = self.flow_map(x_s, pi_su, s, u)
            _, pi_ut = model_forward(x_su, u, t, teacher_feat)

            lam = self.psd_lambda(s, u, t, eps=eps)
            target = lam * pi_su + (1.0 - lam) * pi_ut

            # 数値安定化．soft-label CE 用に確率分布として正規化しておく．
            target = target.clamp_min(1e-8)
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)

        logits_st, _ = model_forward(x_s, s, t, image_feat)
        log_pi_st = F.log_softmax(logits_st, dim=1)
        loss_psd = -(target * log_pi_st).sum(dim=1).mean()

        stats = {
            "loss_psd": loss_psd.detach(),
            "loss_ce_ec": loss_psd.detach(),  # 既存ログ互換用．PSDではCE成分として扱う
            "loss_td": torch.zeros_like(loss_psd.detach()),
        }

        return loss_psd, stats

    @torch.no_grad()
    def sample(
        self,
        model,
        img: torch.Tensor,
        source_net=None,
        num_steps: int = 100,
        return_intermediates: bool = False,
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        cfg_null_condition: str = "learned",
    ):
        device = img.device
        B, _, H, W = img.shape
        pred_traj = []

        x = self.sample_prior(
            B=B,
            H=H,
            W=W,
            device=device,
            dtype=img.dtype,
            img=img,
            source_net=source_net,
        )
        image_feat = model.encode_image(img)
        ts = torch.linspace(0.03, 1.0, num_steps + 1, device=img.device)

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
            x = self.flow_map(x, pi, s_batch, t_batch)

            if self.prior_type == "dirichlet" and self.project_simplex:
                x = x.clamp_min(1e-8)
                x = x / x.sum(dim=1, keepdim=True).clamp_min(1e-8)

            if return_intermediates:
                pred_traj.append(x.argmax(dim=1).cpu())

        if return_intermediates:
            return torch.stack(pred_traj, dim=0)

        return x.argmax(dim=1)

    @torch.no_grad()
    def run_inference_examples(
        self,
        model,
        source_net=None,
        root=None,
        split: str = "val",
        batch_size: int = 4,
        num_steps: int = 100,
        save_dir: str = "./infer_samples",
        image_size=(128, 256),
        num_workers: int = 4,
        use_wandb: bool = False,
        wandb_module=None,
        wandb_prefix: str = "inference",
        max_images: Optional[int] = None,
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        cfg_null_condition: str = "learned",
        imagenet_normalize: bool = False,
    ):
        if root is None:
            raise ValueError("root is required for run_inference_examples")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = Cityscapes20ClassDataset(
            root=root,
            split=split,
            mode="fine",
            image_size=tuple(image_size),
            augment=False,
            color_jitter=False,
            imagenet_normalize=imagenet_normalize,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        was_training = model.training
        model.eval()
        source_was_training = None
        if source_net is not None:
            source_was_training = source_net.training
            source_net.eval()

        img, _, gt_mask = next(iter(loader))
        img = img.to(self.device)
        pred = self.sample(
            model=model,
            img=img,
            source_net=source_net,
            num_steps=num_steps,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
            cfg_null_condition=cfg_null_condition,
        ).cpu()

        img = img.cpu()
        gt_mask = gt_mask.cpu()
        limit = img.size(0) if max_images is None else min(max_images, img.size(0))
        wandb_images = []

        for i in range(limit):
            save_path = save_dir / f"{split}_{i}.png"
            save_prediction_triplet(
                img=img[i],
                gt=gt_mask[i],
                pred=pred[i],
                save_path=save_path,
                title=f"{split} #{i}",
                imagenet_normalize=imagenet_normalize,
            )
            if use_wandb and wandb_module is not None:
                wandb_images.append(
                    wandb_module.Image(str(save_path), caption=f"{split}/sample_{i}")
                )

        if use_wandb and wandb_module is not None and wandb_images:
            wandb_module.log({f"{wandb_prefix}/{split}_examples": wandb_images})

        if was_training:
            model.train()
        if source_net is not None and source_was_training:
            source_net.train()

        print(f"saved to: {save_dir}")
        return wandb_images
