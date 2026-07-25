from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.func import jvp


def sample_consistency_times(
    loss_type: str,
    batch_size: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Use the existing PSD/ECLD time laws for every consistency objective."""
    if loss_type == "psd":
        times = torch.rand(batch_size, 3, device=device)
        times, _ = torch.sort(times, dim=1)
        return times[:, 0], times[:, 1], times[:, 2]

    if loss_type in {"csd", "ecld", "esd"}:
        times = torch.rand(batch_size, 2, device=device)
        times, _ = torch.sort(times, dim=1)
        return times[:, 0], None, times[:, 1]

    if loss_type == "none":
        zero = torch.zeros(batch_size, device=device)
        return zero, None, zero

    raise ValueError(f"Unknown consistency loss: {loss_type}")


def _autocast_disabled(tensor: torch.Tensor):
    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def _autocast_enabled(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype],
):
    if dtype is None or tensor.device.type not in {"cpu", "cuda"}:
        return nullcontext()
    if dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError(f"Unsupported ECLD AMP dtype: {dtype}")
    if tensor.device.type == "cpu" and dtype != torch.bfloat16:
        return nullcontext()
    return torch.autocast(
        device_type=tensor.device.type,
        dtype=dtype,
        enabled=True,
    )


def _forward_logits(
    model,
    x: torch.Tensor,
    image: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    image_feat: Optional[torch.Tensor],
) -> torch.Tensor:
    if image_feat is None:
        return model.forward_logits(x, image, s, t)
    return model.forward_logits_with_image_feat(x, image_feat, s, t)


def _normalize_probability(prob: torch.Tensor, eps: float) -> torch.Tensor:
    prob = prob.float().clamp_min(eps)
    return prob / prob.sum(dim=1, keepdim=True).clamp_min(eps)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def _detached_stats(loss: torch.Tensor, **stats: torch.Tensor) -> Dict[str, torch.Tensor]:
    result = {"loss_consistency": loss.detach()}
    result.update({key: value.detach() for key, value in stats.items()})
    return result


def _finite_or_raise(
    loss_type: str,
    loss: torch.Tensor,
    *,
    s: torch.Tensor,
    t: torch.Tensor,
    diagnostics: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    if bool(torch.isfinite(loss).all()):
        return

    details = {
        "s_min": float(s.detach().min().cpu()),
        "s_max": float(s.detach().max().cpu()),
        "t_min": float(t.detach().min().cpu()),
        "t_max": float(t.detach().max().cpu()),
    }
    if diagnostics is not None:
        for key, value in diagnostics.items():
            value = value.detach()
            if value.numel() == 1:
                details[key] = float(value.cpu())
    raise FloatingPointError(
        f"Non-finite {loss_type} consistency loss. Diagnostics: {details}"
    )


def _validate_probability(name: str, prob: torch.Tensor, atol: float = 1e-4) -> None:
    if not bool(torch.isfinite(prob).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")
    if bool((prob < 0).any()):
        raise FloatingPointError(f"{name} contains negative probabilities")
    sums = prob.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=atol):
        max_error = float((sums - 1.0).abs().max().detach().cpu())
        raise FloatingPointError(
            f"{name} is not normalized over class dim=1 (max error={max_error})"
        )


def _csd_loss(
    *,
    cfm,
    model,
    image: torch.Tensor,
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    image_feat: Optional[torch.Tensor],
    eps: float,
    debug: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Categorical Flow Maps, Eq. (14), with the paper's empirical w_t = 1."""
    with _autocast_disabled(x_s):
        x_s_fp32 = x_s.float()
        image_fp32 = image.float()
        image_feat_fp32 = image_feat.float() if image_feat is not None else None
        s_fp32 = s.float()
        t_fp32 = t.float()

        def flow_at(t_in: torch.Tensor) -> torch.Tensor:
            logits = _forward_logits(
                model,
                x_s_fp32,
                image_fp32,
                s_fp32,
                t_in,
                image_feat_fp32,
            )
            pred_prob = torch.softmax(logits.float(), dim=1)
            return cfm.flow_map(x_s_fp32, pred_prob, s_fp32, t_in)

        x_st, dx_st_dt = jvp(flow_at, (t_fp32,), (torch.ones_like(t_fp32),))

        # CFM Algorithm 3 applies stop-gradient to the complete instantaneous
        # endpoint teacher, including its dependence on the transported state.
        with torch.no_grad():
            teacher_logits = _forward_logits(
                model,
                x_st.detach(),
                image_fp32,
                t_fp32,
                t_fp32,
                image_feat_fp32.detach() if image_feat_fp32 is not None else None,
            )
            teacher_prob = _normalize_probability(
                torch.softmax(teacher_logits.float(), dim=1),
                eps,
            )

        one_minus_t = (1.0 - t_fp32)[:, None, None, None]
        residual = one_minus_t * dx_st_dt.float() - teacher_prob + x_st.float()
        loss_csd = residual.square().sum(dim=1).mean()
        residual_norm = residual.square().sum(dim=1).sqrt().mean()

    if debug:
        _validate_probability("CSD teacher", teacher_prob)
    _finite_or_raise(
        "csd",
        loss_csd,
        s=s,
        t=t,
        diagnostics={"csd_residual_norm": residual_norm},
    )
    zero = _zero(loss_csd)
    return loss_csd, _detached_stats(
        loss_csd,
        loss_csd=loss_csd,
        csd_residual_norm=residual_norm,
        loss_ce_ec=zero,
        loss_td=zero,
    )


def _ecld_debug_dtype_stats(
    *,
    student_logits: torch.Tensor,
    dlogits_dt: torch.Tensor,
    student_prob: torch.Tensor,
    student_log_prob: torch.Tensor,
    dprob_dt: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Return scalar probes whose dtypes can be asserted in debug tests."""
    tensors = {
        "ecld_debug_student_logits": student_logits,
        "ecld_debug_dlogits_dt": dlogits_dt,
        "ecld_debug_student_prob": student_prob,
        "ecld_debug_student_log_prob": student_log_prob,
        "ecld_debug_dprob_dt": dprob_dt,
        "ecld_debug_teacher_logits": teacher_logits,
        "ecld_debug_teacher_prob": teacher_prob,
    }
    return {
        name: tensor.detach().new_zeros(())
        for name, tensor in tensors.items()
    }


def _ecld_loss_fp32(
    *,
    cfm,
    model,
    image: torch.Tensor,
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    image_feat: Optional[torch.Tensor],
    eps: float,
    time_eps: float,
    ec_weight: float,
    td_weight: float,
    time_weighting: str,
    debug: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Original full-FP32 ECLD path kept for numerical compatibility."""
    with _autocast_disabled(x_s):
        x_s_fp32 = x_s.float()
        image_fp32 = image.float()
        image_feat_fp32 = image_feat.float() if image_feat is not None else None
        s_fp32 = s.float()
        t_fp32 = t.float()

        def logits_at(t_in: torch.Tensor) -> torch.Tensor:
            return _forward_logits(
                model,
                x_s_fp32,
                image_fp32,
                s_fp32,
                t_in,
                image_feat_fp32,
            ).float()

        student_logits, dlogits_dt = jvp(
            logits_at,
            (t_fp32,),
            (torch.ones_like(t_fp32),),
        )
        student_prob = torch.softmax(student_logits, dim=1)
        student_log_prob = F.log_softmax(student_logits, dim=1)

        # Exact softmax JVP, without constructing a class-by-class Jacobian.
        prob_dot_logits = (student_prob * dlogits_dt).sum(dim=1, keepdim=True)
        dprob_dt = student_prob * (dlogits_dt - prob_dot_logits)

        x_st = cfm.flow_map(x_s_fp32, student_prob, s_fp32, t_fp32)
        with torch.no_grad():
            teacher_logits = _forward_logits(
                model,
                x_st.detach(),
                image_fp32,
                t_fp32,
                t_fp32,
                image_feat_fp32.detach() if image_feat_fp32 is not None else None,
            )
            teacher_prob = _normalize_probability(
                torch.softmax(teacher_logits.float(), dim=1),
                eps,
            )

        loss_ec_pixel = -(teacher_prob * student_log_prob).sum(dim=1)
        if time_weighting == "none":
            temporal_weight = torch.ones_like(t_fp32)
        elif time_weighting == "inverse_square":
            temporal_weight = (1.0 - t_fp32).clamp_min(time_eps).pow(-2)
        else:
            raise ValueError(f"Unknown ECLD time weighting: {time_weighting}")
        loss_ec = (
            loss_ec_pixel * temporal_weight[:, None, None]
        ).mean()

        gamma = cfm.gamma(s_fp32, t_fp32)
        loss_td = (
            gamma.square()[:, None, None]
            * dprob_dt.square().sum(dim=1)
        ).mean()
        dt_prob_norm = dprob_dt.square().sum(dim=1).sqrt().mean()
        loss_ecld = ec_weight * loss_ec + td_weight * loss_td

    debug_dtype_stats = {}
    if debug:
        _validate_probability("ECLD teacher", teacher_prob)
        debug_dtype_stats = _ecld_debug_dtype_stats(
            student_logits=student_logits,
            dlogits_dt=dlogits_dt,
            student_prob=student_prob,
            student_log_prob=student_log_prob,
            dprob_dt=dprob_dt,
            teacher_logits=teacher_logits,
            teacher_prob=teacher_prob,
        )
    _finite_or_raise(
        "ecld",
        loss_ecld,
        s=s,
        t=t,
        diagnostics={
            "loss_ecld_ec": loss_ec,
            "loss_ecld_td": loss_td,
            "ecld_dt_prob_norm": dt_prob_norm,
        },
    )
    return loss_ecld, _detached_stats(
        loss_ecld,
        loss_ecld=loss_ecld,
        loss_ecld_ec=loss_ec,
        loss_ecld_td=loss_td,
        ecld_dt_prob_norm=dt_prob_norm,
        loss_ce_ec=loss_ec,
        loss_td=loss_td,
        **debug_dtype_stats,
    )


def _ecld_loss_amp(
    *,
    cfm,
    model,
    image: torch.Tensor,
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    image_feat: Optional[torch.Tensor],
    eps: float,
    time_eps: float,
    ec_weight: float,
    td_weight: float,
    time_weighting: str,
    amp_dtype: torch.dtype,
    debug: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ECLD with AMP model/JVP forwards and FP32 probabilities/losses."""
    s_fp32 = s.float()
    t_fp32 = t.float()

    with _autocast_enabled(x_s, amp_dtype):
        def logits_at(t_in: torch.Tensor) -> torch.Tensor:
            logits = _forward_logits(
                model,
                x_s,
                image,
                s_fp32,
                t_in,
                image_feat,
            )
            return logits.to(dtype=amp_dtype)

        student_logits, dlogits_dt = jvp(
            logits_at,
            (t_fp32,),
            (torch.ones_like(t_fp32),),
        )

    with _autocast_disabled(x_s):
        student_logits_fp32 = student_logits.float()
        dlogits_dt_fp32 = dlogits_dt.float()
        student_log_prob = F.log_softmax(student_logits_fp32, dim=1)
        student_prob = student_log_prob.exp()

        # Exact softmax JVP in FP32, without constructing a full Jacobian.
        prob_dot_logits = (
            student_prob * dlogits_dt_fp32
        ).sum(dim=1, keepdim=True)
        dprob_dt = student_prob * (
            dlogits_dt_fp32 - prob_dot_logits
        )

        # Keep this FP32 transport for the student-side objective. Only the
        # detached teacher input is converted back to the model AMP dtype.
        x_st = cfm.flow_map(x_s, student_prob, s_fp32, t_fp32)
        x_st_teacher = x_st.detach().to(dtype=amp_dtype)

    with torch.no_grad():
        detached_feat = image_feat.detach() if image_feat is not None else None
        with _autocast_enabled(x_s, amp_dtype):
            teacher_logits = _forward_logits(
                model,
                x_st_teacher,
                image,
                t_fp32,
                t_fp32,
                detached_feat,
            ).to(dtype=amp_dtype)

        with _autocast_disabled(x_s):
            teacher_prob = _normalize_probability(
                torch.softmax(teacher_logits.float(), dim=1),
                eps,
            )

    with _autocast_disabled(x_s):
        loss_ec_pixel = -(teacher_prob * student_log_prob).sum(dim=1)
        if time_weighting == "none":
            temporal_weight = torch.ones_like(t_fp32)
        elif time_weighting == "inverse_square":
            temporal_weight = (1.0 - t_fp32).clamp_min(time_eps).pow(-2)
        else:
            raise ValueError(f"Unknown ECLD time weighting: {time_weighting}")
        loss_ec = (
            loss_ec_pixel * temporal_weight[:, None, None]
        ).mean()

        gamma = cfm.gamma(s_fp32, t_fp32)
        loss_td = (
            gamma.square()[:, None, None]
            * dprob_dt.square().sum(dim=1)
        ).mean()
        dt_prob_norm = dprob_dt.square().sum(dim=1).sqrt().mean()
        loss_ecld = (
            ec_weight * loss_ec + td_weight * loss_td
        ).float()

    debug_dtype_stats = {}
    if debug:
        _validate_probability("ECLD teacher", teacher_prob)
        debug_dtype_stats = _ecld_debug_dtype_stats(
            student_logits=student_logits,
            dlogits_dt=dlogits_dt,
            student_prob=student_prob,
            student_log_prob=student_log_prob,
            dprob_dt=dprob_dt,
            teacher_logits=teacher_logits,
            teacher_prob=teacher_prob,
        )
    _finite_or_raise(
        "ecld",
        loss_ecld,
        s=s,
        t=t,
        diagnostics={
            "loss_ecld_ec": loss_ec,
            "loss_ecld_td": loss_td,
            "ecld_dt_prob_norm": dt_prob_norm,
        },
    )
    return loss_ecld, _detached_stats(
        loss_ecld,
        loss_ecld=loss_ecld,
        loss_ecld_ec=loss_ec,
        loss_ecld_td=loss_td,
        ecld_dt_prob_norm=dt_prob_norm,
        loss_ce_ec=loss_ec,
        loss_td=loss_td,
        **debug_dtype_stats,
    )


def _ecld_loss(
    *,
    cfm,
    model,
    image: torch.Tensor,
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    image_feat: Optional[torch.Tensor],
    eps: float,
    time_eps: float,
    ec_weight: float,
    td_weight: float,
    time_weighting: str,
    amp_ecld: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    debug: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Categorical Flow Maps ECLD, Eqs. (16), (18), and (19)."""
    common = {
        "cfm": cfm,
        "model": model,
        "image": image,
        "x_s": x_s,
        "s": s,
        "t": t,
        "image_feat": image_feat,
        "eps": eps,
        "time_eps": time_eps,
        "ec_weight": ec_weight,
        "td_weight": td_weight,
        "time_weighting": time_weighting,
        "debug": debug,
    }
    amp_device_supported = (
        x_s.device.type == "cuda"
        or (x_s.device.type == "cpu" and amp_dtype == torch.bfloat16)
    )
    if amp_ecld and amp_dtype is not None and amp_device_supported:
        return _ecld_loss_amp(
            **common,
            amp_dtype=amp_dtype,
        )
    return _ecld_loss_fp32(**common)


def _esd_loss(
    *,
    cfm,
    model,
    image: torch.Tensor,
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    image_feat: Optional[torch.Tensor],
    eps: float,
    debug: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Stable logit-space ESD teacher from Discrete Flow Maps, Eq. (43)."""
    with _autocast_disabled(x_s):
        x_s_fp32 = x_s.float()
        image_fp32 = image.float()
        image_feat_fp32 = image_feat.float() if image_feat is not None else None
        s_fp32 = s.float()
        t_fp32 = t.float()

        # The DFM teacher is entirely stop-gradient. Forward-mode JVP computes
        # D_s z = partial_s z + J_x z b_s without materializing a full Jacobian.
        with torch.no_grad():
            detached_feat = (
                image_feat_fp32.detach() if image_feat_fp32 is not None else None
            )
            logits_ss = _forward_logits(
                model,
                x_s_fp32,
                image_fp32,
                s_fp32,
                s_fp32,
                detached_feat,
            ).float()
            prob_ss = torch.softmax(logits_ss, dim=1)

            # Current local Flow Map uses d_s=max(1-s, cfm.eps), so its
            # diagonal drift is b_s=(psi_ss-x_s)/d_s. For gamma=(t-s)/d_s,
            # kappa^{-1}=d_s(t-s)/(d_s-(t-s)). Removing that scalar
            # denominator inside softmax gives the stable log_arg below.
            denominator_s = (1.0 - s_fp32).clamp_min(float(cfm.eps))
            b_s = (
                prob_ss - x_s_fp32
            ) / denominator_s[:, None, None, None]

            def logits_along_flow(
                x_in: torch.Tensor,
                s_in: torch.Tensor,
            ) -> torch.Tensor:
                return _forward_logits(
                    model,
                    x_in,
                    image_fp32,
                    s_in,
                    t_fp32,
                    detached_feat,
                ).float()

            logits_st_teacher, dlogits_ds = jvp(
                logits_along_flow,
                (x_s_fp32, s_fp32),
                (b_s, torch.ones_like(s_fp32)),
            )
            prob_st_teacher = torch.softmax(logits_st_teacher, dim=1)
            mean_dlogits_ds = (
                prob_st_teacher * dlogits_ds
            ).sum(dim=1, keepdim=True)
            delta_st = dlogits_ds - mean_dlogits_ds

            delta_time = t_fp32 - s_fp32
            stable_scale = denominator_s - delta_time
            log_arg_raw = (
                stable_scale[:, None, None, None]
                - (
                    denominator_s * delta_time
                )[:, None, None, None] * delta_st
            )
            log_arg_min = log_arg_raw.amin()
            nonfinite_ratio = (~torch.isfinite(log_arg_raw)).float().mean()
            clamp_ratio = (log_arg_raw < eps).float().mean()
            log_arg = log_arg_raw.clamp_min(eps)

            teacher_logits = logits_ss - torch.log(log_arg)
            teacher_prob = _normalize_probability(
                torch.softmax(teacher_logits, dim=1),
                eps,
            ).detach()
            teacher_entropy = -(
                teacher_prob * teacher_prob.clamp_min(eps).log()
            ).sum(dim=1).mean()
            teacher_min = teacher_prob.amin()
            teacher_max = teacher_prob.amax()

        student_logits = _forward_logits(
            model,
            x_s_fp32,
            image_fp32,
            s_fp32,
            t_fp32,
            image_feat_fp32,
        ).float()
        student_log_prob = F.log_softmax(student_logits, dim=1)
        loss_esd_pixel = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="none",
        ).sum(dim=1)
        loss_esd = loss_esd_pixel.mean()

    if debug:
        _validate_probability("ESD teacher", teacher_prob)
    _finite_or_raise(
        "esd",
        loss_esd,
        s=s,
        t=t,
        diagnostics={
            "esd_log_arg_min": log_arg_min,
            "esd_clamp_ratio": clamp_ratio,
            "esd_nonfinite_ratio": nonfinite_ratio,
            "esd_teacher_min": teacher_min,
            "esd_teacher_max": teacher_max,
        },
    )
    zero = _zero(loss_esd)
    return loss_esd, _detached_stats(
        loss_esd,
        loss_esd=loss_esd,
        esd_log_arg_min=log_arg_min,
        esd_clamp_ratio=clamp_ratio,
        esd_nonfinite_ratio=nonfinite_ratio,
        esd_teacher_entropy=teacher_entropy,
        esd_teacher_min=teacher_min,
        esd_teacher_max=teacher_max,
        loss_ce_ec=zero,
        loss_td=zero,
    )


def compute_consistency_loss(
    loss_type: str,
    *,
    cfm,
    model,
    image: torch.Tensor,
    x_s: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    image_feat: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    time_eps: float = 1e-4,
    ecld_ec_weight: float = 4.0,
    ecld_td_weight: float = 2.0,
    ecld_time_weighting: str = "none",
    amp_ecld: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    debug: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Common loss interface; class probabilities always use dim=1."""
    if eps <= 0.0:
        raise ValueError("consistency eps must be positive")
    if time_eps <= 0.0:
        raise ValueError("consistency time eps must be positive")

    if loss_type == "none":
        loss = _zero(x_s)
        stats = _detached_stats(
            loss,
            loss_ce_ec=loss,
            loss_td=loss,
        )
    elif loss_type == "psd":
        if u is None:
            raise ValueError("PSD requires an intermediate time u")
        # This intentionally calls the pre-existing implementation unchanged:
        # it is the repository's backward-compatible PSD soft-target CE mode.
        loss, legacy_stats = cfm.psd_loss(
            model=model,
            x_s=x_s,
            img=image,
            s=s,
            u=u,
            t=t,
            image_feat=image_feat,
        )
        stats = dict(legacy_stats)
        stats["loss_consistency"] = loss.detach()
    elif loss_type == "csd":
        loss, stats = _csd_loss(
            cfm=cfm,
            model=model,
            image=image,
            x_s=x_s,
            s=s,
            t=t,
            image_feat=image_feat,
            eps=eps,
            debug=debug,
        )
    elif loss_type == "ecld":
        loss, stats = _ecld_loss(
            cfm=cfm,
            model=model,
            image=image,
            x_s=x_s,
            s=s,
            t=t,
            image_feat=image_feat,
            eps=eps,
            time_eps=time_eps,
            ec_weight=ecld_ec_weight,
            td_weight=ecld_td_weight,
            time_weighting=ecld_time_weighting,
            amp_ecld=amp_ecld,
            amp_dtype=amp_dtype,
            debug=debug,
        )
    elif loss_type == "esd":
        loss, stats = _esd_loss(
            cfm=cfm,
            model=model,
            image=image,
            x_s=x_s,
            s=s,
            t=t,
            image_feat=image_feat,
            eps=eps,
            debug=debug,
        )
    else:
        raise ValueError(f"Unknown consistency loss: {loss_type}")

    _finite_or_raise(loss_type, loss, s=s, t=t, diagnostics=stats)
    return loss, stats
