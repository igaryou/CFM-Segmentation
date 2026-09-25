import types

import torch

from CFM import CategoricalFlowMaps
from eval import SegmentationMetrics
from eval_multisample import probs_to_mask


class VoidDominantModel(torch.nn.Module):
    def __init__(self, num_classes: int = 20) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.observed_state_shapes = []

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        return img

    def forward_with_image_feat(self, x, image_feat, s, t):
        self.observed_state_shapes.append(tuple(x.shape))
        pi = torch.zeros_like(x)
        pi[:, 18] = 0.1
        pi[:, 19] = 0.9
        return pi.log().nan_to_num(neginf=-100.0), pi


def make_void_dominant_cfm() -> CategoricalFlowMaps:
    cfm = CategoricalFlowMaps(num_classes=20, device="cpu")

    def fixed_prior(self, B, H, W, device, dtype, **kwargs):
        x = torch.zeros(B, self.num_classes, H, W, device=device, dtype=dtype)
        x[:, 18] = 1.0
        x[:, 19] = 2.0
        return x

    cfm.sample_prior = types.MethodType(fixed_prior, cfm)
    return cfm


def test_decode_prediction_option_preserves_legacy_behavior() -> None:
    cfm = make_void_dominant_cfm()
    x = cfm.sample_prior(1, 2, 3, "cpu", torch.float32)

    assert torch.all(cfm.decode_prediction(x, exclude_void=True) == 18)
    assert torch.all(cfm.decode_prediction(x, exclude_void=False) == 19)
    assert x.shape == (1, 20, 2, 3)


def test_sample_excludes_void_from_every_trajectory_step_and_final_prediction() -> None:
    cfm = make_void_dominant_cfm()
    model = VoidDominantModel()
    img = torch.zeros(2, 3, 2, 3)

    trajectory = cfm.sample(
        model=model,
        img=img,
        num_steps=3,
        return_intermediates=True,
        exclude_void=True,
    )
    final_prediction = cfm.sample(
        model=model,
        img=img,
        num_steps=2,
        exclude_void=True,
    )

    assert trajectory.shape == (4, 2, 2, 3)
    assert not torch.any(trajectory == 19)
    assert not torch.any(final_prediction == 19)
    assert model.observed_state_shapes
    assert all(shape == (2, 20, 2, 3) for shape in model.observed_state_shapes)


def test_multisample_argmax_and_sampling_exclude_void() -> None:
    probs = torch.zeros(2, 20, 2, 3)
    probs[:, 7] = 0.1
    probs[:, 19] = 0.9

    argmax_prediction = probs_to_mask(probs, mode="argmax", exclude_void=True)
    sampled_prediction = probs_to_mask(probs, mode="sample", exclude_void=True)

    assert torch.all(argmax_prediction == 7)
    assert torch.all(sampled_prediction == 7)
    assert probs.shape == (2, 20, 2, 3)


def test_multisample_default_keeps_legacy_void_argmax() -> None:
    probs = torch.zeros(1, 20, 1, 1)
    probs[:, 18] = 0.1
    probs[:, 19] = 0.9

    assert probs_to_mask(probs, mode="argmax").item() == 19


def test_cityscapes_metrics_ignore_void_gt_and_use_19_classes() -> None:
    metrics = SegmentationMetrics(num_classes=20, ignore_index=19)
    metrics.update(
        pred=torch.tensor([[0, 1, 18, 4]]),
        target=torch.tensor([[0, 1, 18, 19]]),
    )
    result = metrics.compute()

    assert metrics.confmat.shape == (19, 19)
    assert result["evaluated_classes"] == 19
    assert result["pixel_acc"] == 1.0


def test_cityscapes_metrics_reject_void_prediction_on_valid_gt() -> None:
    metrics = SegmentationMetrics(num_classes=20, ignore_index=19)

    try:
        metrics.update(pred=torch.tensor([[19]]), target=torch.tensor([[0]]))
    except RuntimeError:
        pass
    else:
        raise AssertionError("void prediction on valid GT must not be silently ignored")
