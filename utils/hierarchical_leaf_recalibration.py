"""Objectives and safeguards for Hierarchical Class-balanced Leaf Recalibration.

HCLR changes only the 1x1 leaf decoder after mapping.  The 3-level semantic
Gaussian map, geometry, appearance, opacity, and trajectory remain frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def split_tree_logits(
    rendered_semantic: torch.Tensor,
    widths: Sequence[int],
) -> Tuple[torch.Tensor, ...]:
    """Split a ``[sum(widths), H, W]`` semantic image into tree levels."""
    if rendered_semantic.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(rendered_semantic.shape)}")
    if rendered_semantic.shape[0] != sum(widths):
        raise ValueError(
            f"Semantic width {rendered_semantic.shape[0]} does not match {tuple(widths)}"
        )
    return tuple(torch.split(rendered_semantic, tuple(int(v) for v in widths), dim=0))


def build_leaf_to_level_matrices(
    label_mapping_tree: Dict[str, Sequence[int]],
    widths: Sequence[int],
    num_leaf_classes: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, ...]:
    """Build fixed matrices that aggregate leaf probabilities to tree levels."""
    matrices = [
        torch.zeros((int(width), num_leaf_classes), device=device, dtype=dtype)
        for width in widths
    ]
    seen = torch.zeros(num_leaf_classes, device=device, dtype=torch.bool)
    for leaf_key, path in label_mapping_tree.items():
        leaf = int(leaf_key)
        if not 0 <= leaf < num_leaf_classes:
            continue
        if len(path) < len(widths):
            raise ValueError(f"Leaf {leaf} has incomplete path {path}")
        for level, width in enumerate(widths):
            node = int(path[level])
            if not 0 <= node < int(width):
                raise ValueError(
                    f"Leaf {leaf}, level {level}: node {node} outside width {width}"
                )
            matrices[level][node, leaf] = 1.0
        seen[leaf] = True
    if not bool(seen.all()):
        missing = torch.nonzero(~seen, as_tuple=False).flatten().tolist()
        raise ValueError(f"Tree has no paths for leaf classes: {missing}")
    return tuple(matrices)


def semantic_boundary_mask(labels: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Return a dilated four-neighbour boundary mask for ``[H,W]`` labels."""
    if labels.ndim != 2:
        raise ValueError(f"Expected [H,W] labels, got {tuple(labels.shape)}")
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    if radius > 0:
        kernel = 2 * int(radius) + 1
        boundary = F.max_pool2d(
            boundary[None, None].float(),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        )[0, 0].bool()
    return boundary


def hierarchical_consistency_loss(
    tree_levels: Sequence[torch.Tensor],
    leaf_probabilities: torch.Tensor,
    leaf_to_level: Sequence[torch.Tensor],
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric JS agreement between tree heads and aggregated leaf probabilities."""
    epsilon = 1e-6
    flat_leaf = leaf_probabilities.flatten(1)
    flat_valid = valid_mask.flatten()
    losses = []
    for logits, matrix in zip(tree_levels, leaf_to_level):
        tree_probability = logits.softmax(dim=0).flatten(1)
        aggregate = matrix @ flat_leaf
        aggregate = aggregate / aggregate.sum(dim=0, keepdim=True).clamp_min(epsilon)
        mixture = 0.5 * (tree_probability + aggregate)
        js = 0.5 * (
            (
                tree_probability
                * ((tree_probability + epsilon).log() - (mixture + epsilon).log())
            ).sum(dim=0)
            + (
                aggregate
                * ((aggregate + epsilon).log() - (mixture + epsilon).log())
            ).sum(dim=0)
        )
        losses.append(
            js[flat_valid].mean() if bool(flat_valid.any()) else js.mean() * 0.0
        )
    return torch.stack(losses).mean()


def soft_class_iou_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Macro soft-IoU over target classes present in ``mask``."""
    active_target = target[mask]
    if active_target.numel() == 0:
        return probabilities.sum() * 0.0
    losses = []
    mask_float = mask.to(probabilities.dtype)
    for class_id in torch.unique(active_target):
        class_index = int(class_id)
        probability = probabilities[class_index] * mask_float
        truth = (target == class_index).to(probabilities.dtype) * mask_float
        intersection = (probability * truth).sum()
        union = probability.sum() + truth.sum() - intersection
        losses.append(1.0 - (intersection + 1e-6) / (union + 1e-6))
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class HCLRWeights:
    focal: float = 1.0
    soft_iou: float = 0.25
    hierarchy: float = 0.03
    distillation: float = 0.05
    trust: float = 0.01
    focal_gamma: float = 1.5


class BaselineAnchoredActivationGate(torch.nn.Module):
    """Reject spatially unsupported classes introduced by recalibration.

    HCLR is a residual calibration of an already trained decoder.  Classes
    already active in the baseline prediction may change freely.  A class
    absent from the baseline frame must collect a small, resolution-aware
    amount of spatial evidence before it is allowed to become active.
    """

    def __init__(
        self,
        calibrated_decoder: torch.nn.Module,
        baseline_weight: torch.Tensor,
        *,
        minimum_pixels: int = 9,
        minimum_area_ratio: float = 1e-5,
    ) -> None:
        super().__init__()
        self.calibrated_decoder = calibrated_decoder
        self.register_buffer("baseline_weight", baseline_weight.detach().clone())
        self.minimum_pixels = int(minimum_pixels)
        self.minimum_area_ratio = float(minimum_area_ratio)

    def support_threshold(self, height: int, width: int) -> int:
        return max(
            self.minimum_pixels,
            int(math.ceil(self.minimum_area_ratio * int(height) * int(width))),
        )

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        baseline_logits = F.conv2d(semantic, self.baseline_weight)
        calibrated_logits = self.calibrated_decoder(semantic)
        output = calibrated_logits.clone()
        threshold = self.support_threshold(semantic.shape[-2], semantic.shape[-1])
        with torch.no_grad():
            baseline_prediction = baseline_logits.argmax(dim=1)
            calibrated_prediction = calibrated_logits.argmax(dim=1)
            revert_masks = []
            for batch_index in range(semantic.shape[0]):
                baseline_classes = set(
                    int(value) for value in baseline_prediction[batch_index].unique().tolist()
                )
                revert = torch.zeros_like(
                    calibrated_prediction[batch_index], dtype=torch.bool
                )
                for class_id in calibrated_prediction[batch_index].unique().tolist():
                    class_id = int(class_id)
                    if class_id in baseline_classes:
                        continue
                    class_mask = calibrated_prediction[batch_index] == class_id
                    if int(class_mask.sum()) < threshold:
                        revert |= class_mask
                revert_masks.append(revert)
        for batch_index, revert in enumerate(revert_masks):
            if bool(revert.any()):
                output[batch_index, :, revert] = baseline_logits[batch_index, :, revert]
        return output


def temporally_stratified_class_coverage(
    histograms: Dict[int, torch.Tensor],
    count: int,
    *,
    num_bins: int = 8,
) -> Tuple[Sequence[int], Dict[int, Sequence[int]]]:
    """Select equal temporal quotas, then optimize rare-class coverage per bin."""
    frames = sorted(int(frame) for frame in histograms)
    count = min(int(count), len(frames))
    if count <= 0:
        return [], {}
    active_bins = min(int(num_bins), count, len(frames))
    frame_bins = [list(map(int, values)) for values in np.array_split(frames, active_bins)]
    base_quota, remainder = divmod(count, active_bins)
    quotas = [base_quota + int(index < remainder) for index in range(active_bins)]
    total = torch.stack([histograms[frame].double() for frame in frames]).sum(dim=0)
    present = total > 0
    rarity = torch.zeros_like(total)
    rarity[present] = 1.0 / torch.sqrt(total[present])
    rarity /= rarity[present].mean().clamp_min(1e-12)
    global_coverage = torch.zeros_like(total)
    selected_by_bin: Dict[int, list[int]] = {index: [] for index in range(active_bins)}

    for round_index in range(max(quotas)):
        for bin_index, candidates in enumerate(frame_bins):
            if round_index >= quotas[bin_index]:
                continue
            already = selected_by_bin[bin_index]
            best_frame = None
            best_utility = -float("inf")
            span = max(1, candidates[-1] - candidates[0])
            target_position = candidates[0] + (
                (round_index + 0.5) / quotas[bin_index]
            ) * span
            for frame in candidates:
                if frame in already:
                    continue
                class_presence = histograms[frame] > 0
                class_utility = (
                    rarity[class_presence]
                    / torch.sqrt(1.0 + global_coverage[class_presence])
                ).sum().item()
                anchor_utility = 1.0 - abs(frame - target_position) / span
                separation = (
                    min(abs(frame - other) for other in already) / span
                    if already else anchor_utility
                )
                utility = class_utility + 0.15 * anchor_utility + 0.10 * separation
                if utility > best_utility:
                    best_frame, best_utility = frame, utility
            selected_by_bin[bin_index].append(int(best_frame))
            global_coverage += (histograms[int(best_frame)] > 0).double()

    selected = sorted(frame for values in selected_by_bin.values() for frame in values)
    return selected, {key: sorted(value) for key, value in selected_by_bin.items()}


def bounded_class_weights(
    counts: torch.Tensor,
    minimum: float = 0.5,
    maximum: float = 3.0,
) -> torch.Tensor:
    """Square-root inverse-frequency weights, bounded for stable calibration."""
    counts = counts.float()
    result = torch.ones_like(counts)
    present = counts > 0
    if bool(present.any()):
        median = counts[present].median().clamp_min(1.0)
        result[present] = torch.sqrt(median / counts[present].clamp_min(1.0))
        result[present].clamp_(float(minimum), float(maximum))
    return result


def class_adaptive_radii(
    counts: torch.Tensor,
    minimum: float = 0.05,
    maximum: float = 0.15,
) -> torch.Tensor:
    """Give rare decoder rows more freedom while tightly protecting frequent rows."""
    counts = counts.float()
    radii = torch.full_like(counts, float(minimum))
    present = counts > 0
    if bool(present.any()):
        median = counts[present].median().clamp_min(1.0)
        rarity = torch.sqrt(median / counts[present].clamp_min(1.0)).clamp(0.5, 2.0)
        normalized = (rarity - 0.5) / 1.5
        radii[present] = float(minimum) + normalized * (float(maximum) - float(minimum))
    return radii


@torch.no_grad()
def clamp_decoder_rows(
    weight: torch.Tensor,
    baseline_weight: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Project each leaf-class row into its independent L2 trust region."""
    delta = weight - baseline_weight
    norm = delta.flatten(1).norm(dim=1)
    scale = (radii / norm.clamp_min(1e-12)).clamp_max(1.0)
    weight.copy_(baseline_weight + delta * scale.view(-1, 1, 1, 1))
    return norm


def stable_distillation_loss(
    logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    confidence_threshold: float = 0.90,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preserve baseline distributions only where the baseline is confidently correct."""
    with torch.no_grad():
        teacher_probability = (baseline_logits / temperature).softmax(dim=0)
        confidence, prediction = teacher_probability.max(dim=0)
        stable = (
            valid_mask
            & (prediction == target)
            & (confidence >= float(confidence_threshold))
            & ~semantic_boundary_mask(target, radius=1)
        )
    if not bool(stable.any()):
        return logits.sum() * 0.0, stable
    student_log_probability = (logits / temperature).log_softmax(dim=0)
    per_pixel = F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="none",
    ).sum(dim=0) * (temperature ** 2)
    return per_pixel[stable].mean(), stable


def decoder_trust_loss(
    weight: torch.Tensor,
    baseline_weight: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    delta_norm = (weight - baseline_weight).flatten(1).norm(dim=1)
    return ((delta_norm / radii.clamp_min(1e-6)) ** 2).mean()


def compute_hclr_loss(
    rendered_semantic: torch.Tensor,
    decoder: torch.nn.Module,
    baseline_weight: torch.Tensor,
    semantic_gt: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor,
    row_radii: torch.Tensor,
    leaf_to_level: Sequence[torch.Tensor],
    *,
    widths: Sequence[int] = (10, 4, 7),
    weights: HCLRWeights = HCLRWeights(),
    confidence_threshold: float = 0.90,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the bounded leaf recalibration objective for one rendered view."""
    target = semantic_gt[-1].long()
    logits = decoder(rendered_semantic.unsqueeze(0)).squeeze(0)
    probabilities = logits.softmax(dim=0)
    target_probability = probabilities.gather(0, target.unsqueeze(0)).squeeze(0)
    ce = F.cross_entropy(
        logits.unsqueeze(0), target.unsqueeze(0), weight=class_weights, reduction="none"
    )[0]
    focal = (((1.0 - target_probability) ** weights.focal_gamma) * ce)[valid_mask].mean()
    soft_iou = soft_class_iou_loss(probabilities, target, valid_mask)

    tree_levels = split_tree_logits(rendered_semantic.detach(), widths)
    hierarchy = hierarchical_consistency_loss(
        tree_levels, probabilities, leaf_to_level, valid_mask
    )
    baseline_logits = F.conv2d(rendered_semantic.detach().unsqueeze(0), baseline_weight).squeeze(0)
    distillation, stable_mask = stable_distillation_loss(
        logits,
        baseline_logits,
        target,
        valid_mask,
        confidence_threshold=confidence_threshold,
    )
    trust = decoder_trust_loss(decoder.weight, baseline_weight, row_radii)
    total = (
        weights.focal * focal
        + weights.soft_iou * soft_iou
        + weights.hierarchy * hierarchy
        + weights.distillation * distillation
        + weights.trust * trust
    )
    diagnostics = {
        "loss": total.detach(),
        "focal": focal.detach(),
        "soft_iou": soft_iou.detach(),
        "hierarchy": hierarchy.detach(),
        "distillation": distillation.detach(),
        "trust": trust.detach(),
        "valid_ratio": valid_mask.float().mean().detach(),
        "stable_ratio": stable_mask.float().mean().detach(),
    }
    return total, diagnostics


def _mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape
    dilation = max(1, int(round(dilation_ratio * np.sqrt(height ** 2 + width ** 2))))
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=dilation)
    return mask - eroded[1 : height + 1, 1 : width + 1]


def frame_semantic_scores(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> Tuple[float, float]:
    """Return frame macro IoU and boundary IoU using the project's convention."""
    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    ious = []
    boundary_ious = []
    for class_id in range(int(num_classes)):
        predicted = prediction_np == class_id
        truth = target_np == class_id
        union = np.logical_or(predicted, truth).sum()
        if union == 0:
            continue
        ious.append(float(np.logical_and(predicted, truth).sum() / union))
        predicted_boundary = _mask_to_boundary(predicted)
        truth_boundary = _mask_to_boundary(truth)
        boundary_union = np.logical_or(predicted_boundary, truth_boundary).sum()
        boundary_ious.append(
            float(np.logical_and(predicted_boundary, truth_boundary).sum() / boundary_union)
            if boundary_union else 1.0
        )
    return float(np.mean(ious)), float(np.mean(boundary_ious))


def validation_score(paired: Dict[str, float]) -> float:
    """Rank checkpoints by gain while penalizing broad or severe regressions."""
    lower_tail_penalty = max(0.0, -float(paired["miou_p10_delta_pp"]))
    negative_rate_penalty = max(0.0, float(paired["miou_negative_rate"]) - 0.40)
    return (
        float(paired["miou_delta_pp"])
        + 0.35 * float(paired["mbiou_delta_pp"])
        + 0.15 * float(paired["miou_median_delta_pp"])
        - 0.03 * lower_tail_penalty
        - 0.50 * negative_rate_penalty
    )
