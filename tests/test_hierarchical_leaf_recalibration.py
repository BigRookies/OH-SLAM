import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.hierarchical_leaf_recalibration import (
    BaselineAnchoredActivationGate,
    bounded_class_weights,
    class_adaptive_radii,
    clamp_decoder_rows,
    stable_distillation_loss,
    temporally_stratified_class_coverage,
    validation_score,
)


def test_rare_classes_get_more_weight_and_room_to_move():
    counts = torch.tensor([1000.0, 100.0, 10.0])
    weights = bounded_class_weights(counts)
    radii = class_adaptive_radii(counts)
    assert weights[2] > weights[1] > weights[0]
    assert radii[2] > radii[1] > radii[0]


def test_row_projection_obeys_individual_radii():
    baseline = torch.zeros(2, 3, 1, 1)
    current = torch.tensor([[[[3.0]], [[4.0]], [[0.0]]], [[[0.0]], [[0.0]], [[2.0]]]])
    radii = torch.tensor([1.0, 0.5])
    clamp_decoder_rows(current, baseline, radii)
    assert torch.allclose(current.flatten(1).norm(dim=1), radii)


def test_distillation_ignores_wrong_teacher_pixels():
    baseline = torch.tensor([[[3.0, 3.0]], [[0.0, 0.0]]])
    logits = torch.tensor([[[2.0, 2.0]], [[1.0, 1.0]]], requires_grad=True)
    target = torch.tensor([[0, 1]])
    valid = torch.ones_like(target, dtype=torch.bool)
    loss, stable = stable_distillation_loss(
        logits, baseline, target, valid, confidence_threshold=0.9
    )
    assert stable.tolist() == [[False, False]]  # both pixels are boundary pixels
    assert loss == 0.0


def test_validation_score_rewards_balanced_improvement():
    good = {
        "miou_delta_pp": 1.0,
        "mbiou_delta_pp": 0.8,
        "miou_median_delta_pp": 0.2,
        "miou_p10_delta_pp": -0.5,
        "miou_negative_rate": 0.2,
    }
    risky = dict(good, miou_p10_delta_pp=-8.0, miou_negative_rate=0.7)
    assert validation_score(good) > validation_score(risky)


def _decoder(weight):
    module = torch.nn.Conv2d(2, weight.shape[0], 1, bias=False)
    module.weight.data.copy_(weight[:, :, None, None])
    return module


def test_activation_gate_reverts_unsupported_new_class():
    baseline_weight = torch.tensor([[1.0, 0.0], [0.0, 0.0]])[:, :, None, None]
    calibrated = _decoder(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    gate = BaselineAnchoredActivationGate(
        calibrated, baseline_weight, minimum_pixels=2, minimum_area_ratio=0.0
    )
    semantic = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 0.0]]]])
    assert gate(semantic).argmax(1).eq(0).all()


def test_activation_gate_keeps_spatially_supported_new_class():
    baseline_weight = torch.tensor([[1.0, 0.0], [0.0, 0.0]])[:, :, None, None]
    calibrated = _decoder(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    gate = BaselineAnchoredActivationGate(
        calibrated, baseline_weight, minimum_pixels=2, minimum_area_ratio=0.0
    )
    semantic = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [0.0, 0.0]]]])
    assert int((gate(semantic).argmax(1) == 1).sum()) == 2


def test_temporal_sampler_enforces_equal_bin_quotas():
    histograms = {
        frame: torch.tensor([100.0, float(frame % 7 == 0), float(frame % 11 == 0)])
        for frame in range(80)
    }
    selected, by_bin = temporally_stratified_class_coverage(
        histograms, 16, num_bins=4
    )
    assert len(selected) == 16
    assert [len(by_bin[index]) for index in range(4)] == [4, 4, 4, 4]
    assert all(0 <= frame < 20 for frame in by_bin[0])
    assert all(60 <= frame < 80 for frame in by_bin[3])


if __name__ == "__main__":
    test_rare_classes_get_more_weight_and_room_to_move()
    test_row_projection_obeys_individual_radii()
    test_distillation_ignores_wrong_teacher_pixels()
    test_validation_score_rewards_balanced_improvement()
    test_activation_gate_reverts_unsupported_new_class()
    test_activation_gate_keeps_spatially_supported_new_class()
    test_temporal_sampler_enforces_equal_bin_quotas()
    print("HCLR CPU tests passed")
