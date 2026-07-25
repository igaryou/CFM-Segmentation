import warnings

import pytest
import torch
from torch import nn
from torch.func import jvp

from CFM import CategoricalFlowMaps
from consistency_losses import compute_consistency_loss
from main import build_parser, json_safe_config, normalize_args


class TinyEndpointModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.state = nn.Conv2d(num_classes, num_classes, 1, bias=False)
        self.image = nn.Conv2d(3, num_classes, 1, bias=False)
        self.slope_s = nn.Parameter(torch.randn(1, num_classes, 1, 1) * 0.1)
        self.slope_t = nn.Parameter(torch.randn(1, num_classes, 1, 1) * 0.1)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image(image)

    def forward_logits_with_image_feat(self, x_s, image_feat, s, t):
        return (
            self.state(x_s)
            + image_feat
            + self.slope_s * s[:, None, None, None]
            + self.slope_t * t[:, None, None, None]
        )

    def forward_logits(self, x_s, image, s, t):
        return self.forward_logits_with_image_feat(
            x_s,
            self.encode_image(image),
            s,
            t,
        )

    def forward_with_image_feat(self, x_s, image_feat, s, t):
        logits = self.forward_logits_with_image_feat(x_s, image_feat, s, t)
        return logits, torch.softmax(logits, dim=1)

    def forward(self, x_s, image, s, t):
        logits = self.forward_logits(x_s, image, s, t)
        return logits, torch.softmax(logits, dim=1)


class ConstantEndpointModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.linspace(-0.4, 0.4, num_classes))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            image.shape[0],
            1,
            image.shape[2],
            image.shape[3],
            device=image.device,
            dtype=image.dtype,
        )

    def _logits(self, x_s):
        return self.logits[None, :, None, None].expand(
            x_s.shape[0],
            -1,
            x_s.shape[2],
            x_s.shape[3],
        )

    def forward_logits_with_image_feat(self, x_s, image_feat, s, t):
        return self._logits(x_s)

    def forward_logits(self, x_s, image, s, t):
        return self._logits(x_s)

    def forward_with_image_feat(self, x_s, image_feat, s, t):
        logits = self._logits(x_s)
        return logits, torch.softmax(logits, dim=1)

    def forward(self, x_s, image, s, t):
        logits = self._logits(x_s)
        return logits, torch.softmax(logits, dim=1)


class SplitTeacherStudentModel(nn.Module):
    """Separate diagonal/off-diagonal parameters to audit stop-gradient."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.teacher_logits = nn.Parameter(torch.randn(num_classes) * 0.1)
        self.student_logits = nn.Parameter(torch.randn(num_classes) * 0.1)

    def _expanded(self, parameter, x_s):
        return parameter[None, :, None, None].expand(
            x_s.shape[0],
            -1,
            x_s.shape[2],
            x_s.shape[3],
        )

    def forward_logits_with_image_feat(self, x_s, image_feat, s, t):
        parameter = self.teacher_logits if torch.equal(s, t) else self.student_logits
        return self._expanded(parameter, x_s)

    def forward_logits(self, x_s, image, s, t):
        return self.forward_logits_with_image_feat(x_s, image, s, t)

    def forward_with_image_feat(self, x_s, image_feat, s, t):
        logits = self.forward_logits_with_image_feat(x_s, image_feat, s, t)
        return logits, torch.softmax(logits, dim=1)

    def forward(self, x_s, image, s, t):
        logits = self.forward_logits(x_s, image, s, t)
        return logits, torch.softmax(logits, dim=1)


@pytest.fixture
def tiny_case():
    torch.manual_seed(13)
    batch, classes, height, width = 2, 4, 3, 2
    model = TinyEndpointModel(classes)
    cfm = CategoricalFlowMaps(
        num_classes=classes,
        eps=0.05,
        label_smoothing=0.0,
        device="cpu",
    )
    image = torch.randn(batch, 3, height, width)
    x_s = torch.randn(batch, classes, height, width)
    s = torch.tensor([0.12, 0.28])
    u = torch.tensor([0.42, 0.55])
    t = torch.tensor([0.73, 0.81])
    return model, cfm, image, x_s, s, u, t


@pytest.mark.parametrize("loss_type", ["psd", "csd", "ecld", "esd"])
def test_consistency_losses_are_scalar_finite_and_backward(tiny_case, loss_type):
    model, cfm, image, x_s, s, u, t = tiny_case
    image_feat = model.encode_image(image)
    loss, stats = compute_consistency_loss(
        loss_type,
        cfm=cfm,
        model=model,
        image=image,
        image_feat=image_feat,
        x_s=x_s,
        s=s,
        u=u if loss_type == "psd" else None,
        t=t,
        debug=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert stats["loss_consistency"].ndim == 0
    assert not stats["loss_consistency"].requires_grad

    loss.backward()
    finite_grads = [
        grad
        for parameter in model.parameters()
        if (grad := parameter.grad) is not None and torch.isfinite(grad).all()
    ]
    assert finite_grads
    assert any(float(grad.abs().sum()) > 0.0 for grad in finite_grads)


def test_psd_common_interface_is_exact_legacy_regression(tiny_case):
    model, cfm, image, x_s, s, u, t = tiny_case
    image_feat = model.encode_image(image)

    legacy_loss, legacy_stats = cfm.psd_loss(
        model=model,
        x_s=x_s,
        img=image,
        s=s,
        u=u,
        t=t,
        image_feat=image_feat,
    )
    common_loss, common_stats = compute_consistency_loss(
        "psd",
        cfm=cfm,
        model=model,
        image=image,
        image_feat=image_feat,
        x_s=x_s,
        s=s,
        u=u,
        t=t,
    )

    torch.testing.assert_close(common_loss, legacy_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        common_stats["loss_psd"],
        legacy_stats["loss_psd"],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("loss_type", ["csd", "esd"])
def test_constant_self_consistent_model_has_near_zero_residual(loss_type):
    classes = 4
    model = ConstantEndpointModel(classes)
    cfm = CategoricalFlowMaps(num_classes=classes, eps=0.05, device="cpu")
    image = torch.randn(2, 3, 2, 3)
    x_s = torch.randn(2, classes, 2, 3)
    s = torch.tensor([0.1, 0.2])
    t = torch.tensor([0.6, 0.8])

    loss, _ = compute_consistency_loss(
        loss_type,
        cfm=cfm,
        model=model,
        image=image,
        x_s=x_s,
        s=s,
        t=t,
        debug=True,
    )
    assert float(loss.detach()) < 2e-6


@pytest.mark.parametrize(
    ("loss_type", "entropy_factor"),
    [("psd", 1.0), ("ecld", 4.0)],
)
def test_self_consistent_soft_ce_has_entropy_constant_and_zero_gradient(
    loss_type,
    entropy_factor,
):
    classes = 4
    model = ConstantEndpointModel(classes)
    cfm = CategoricalFlowMaps(num_classes=classes, eps=0.05, device="cpu")
    image = torch.randn(2, 3, 2, 3)
    x_s = torch.randn(2, classes, 2, 3)
    s = torch.tensor([0.1, 0.2])
    u = torch.tensor([0.4, 0.5])
    t = torch.tensor([0.6, 0.8])

    loss, _ = compute_consistency_loss(
        loss_type,
        cfm=cfm,
        model=model,
        image=image,
        x_s=x_s,
        s=s,
        u=u if loss_type == "psd" else None,
        t=t,
    )
    probability = torch.softmax(model.logits.detach(), dim=0)
    entropy = -(probability * probability.log()).sum()
    torch.testing.assert_close(
        loss.detach(),
        entropy_factor * entropy,
        rtol=1e-5,
        atol=1e-6,
    )

    loss.backward()
    assert model.logits.grad is not None
    assert float(model.logits.grad.abs().max()) < 1e-6


def test_endpoint_logit_jvp_matches_central_difference(tiny_case):
    model, _, image, x_s, s, _, t = tiny_case
    image_feat = model.encode_image(image)

    def logits_at(t_in):
        return model.forward_logits_with_image_feat(x_s, image_feat, s, t_in)

    _, tangent = jvp(logits_at, (t,), (torch.ones_like(t),))
    step = 1e-3
    finite_difference = (logits_at(t + step) - logits_at(t - step)) / (2 * step)
    torch.testing.assert_close(tangent, finite_difference, rtol=2e-3, atol=2e-4)


def test_esd_teacher_probability_diagnostics(tiny_case):
    model, cfm, image, x_s, s, _, t = tiny_case
    loss, stats = compute_consistency_loss(
        "esd",
        cfm=cfm,
        model=model,
        image=image,
        image_feat=model.encode_image(image),
        x_s=x_s,
        s=s,
        t=t,
        debug=True,
    )
    assert torch.isfinite(loss)
    assert 0.0 <= float(stats["esd_teacher_min"])
    assert float(stats["esd_teacher_max"]) <= 1.0
    assert float(stats["esd_nonfinite_ratio"]) == 0.0


@pytest.mark.parametrize("loss_type", ["csd", "ecld", "esd"])
def test_diagonal_teacher_branch_is_stop_gradient(loss_type):
    classes = 4
    model = SplitTeacherStudentModel(classes)
    cfm = CategoricalFlowMaps(num_classes=classes, eps=0.05, device="cpu")
    image = torch.randn(2, 3, 2, 3)
    x_s = torch.randn(2, classes, 2, 3)
    s = torch.tensor([0.1, 0.2])
    t = torch.tensor([0.6, 0.8])

    loss, _ = compute_consistency_loss(
        loss_type,
        cfm=cfm,
        model=model,
        image=image,
        x_s=x_s,
        s=s,
        t=t,
    )
    loss.backward()

    assert model.teacher_logits.grad is None
    assert model.student_logits.grad is not None
    assert torch.isfinite(model.student_logits.grad).all()


def test_cli_defaults_and_legacy_priority():
    parser = build_parser()

    default_args = normalize_args(parser.parse_args(["--result_dir", "/tmp/test"]))
    assert default_args.consistency_loss == "psd"
    assert default_args.consistency_weight == 1.0
    assert default_args.ecld_ec_weight == 4.0
    assert default_args.ecld_td_weight == 2.0
    config = json_safe_config(default_args)
    assert config["consistency_loss"] == "psd"
    assert config["consistency_weight"] == 1.0
    assert config["ecld_ec_weight"] == 4.0
    assert config["ecld_td_weight"] == 2.0

    legacy_args = normalize_args(
        parser.parse_args(
            [
                "--result_dir",
                "/tmp/test",
                "--distill_loss",
                "ecld",
                "--lambda_distill",
                "0.7",
                "--lambda_td",
                "1.5",
            ]
        )
    )
    assert legacy_args.consistency_loss == "ecld"
    assert legacy_args.consistency_weight == 0.7
    assert legacy_args.ecld_td_weight == 3.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preferred_args = normalize_args(
            parser.parse_args(
                [
                    "--result_dir",
                    "/tmp/test",
                    "--consistency_loss",
                    "csd",
                    "--distill_loss",
                    "psd",
                    "--consistency_weight",
                    "0.4",
                    "--lambda_distill",
                    "0.8",
                ]
            )
        )
    assert preferred_args.consistency_loss == "csd"
    assert preferred_args.consistency_weight == 0.4
    assert len(caught) >= 2
