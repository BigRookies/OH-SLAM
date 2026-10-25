#!/usr/bin/env python3
"""Post-map Hierarchical Class-balanced Leaf Recalibration (HCLR).

The command consumes a completed PST+TCBR map.  Only the 21-to-leaf
Conv2d decoder is optimized; every Gaussian parameter and every camera pose is
frozen.  Official eval frames are excluded from training and model selection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "hierslam-diff-gaussian-rasterization-w-depth"))

from scripts.postmap_semantic_io import (
    baseline_metric_directory,
    build_dataset,
    file_sha256,
    frame_data,
    json_dump,
    load_config,
    load_decoder,
    load_params,
    make_camera,
    read_metric_directory,
    reliable_mask,
    render_frame,
    run_official_evaluation,
)
from utils.common_utils import seed_everything
from utils.hierarchical_leaf_recalibration import (
    BaselineAnchoredActivationGate,
    HCLRWeights,
    build_leaf_to_level_matrices,
    bounded_class_weights,
    class_adaptive_radii,
    clamp_decoder_rows,
    compute_hclr_loss,
    frame_semantic_scores,
    temporally_stratified_class_coverage,
    validation_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config-override",
        default=None,
        help="Frozen self-contained config used when a legacy source config imports a removed module",
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=160)
    parser.add_argument("--selected-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=48)
    parser.add_argument("--temporal-bins", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--minimum-row-radius", type=float, default=0.05)
    parser.add_argument("--maximum-row-radius", type=float, default=0.15)
    parser.add_argument("--protect-confidence", type=float, default=0.90)
    parser.add_argument("--activation-minimum-pixels", type=int, default=9)
    parser.add_argument("--activation-minimum-area-ratio", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def split_protocol(
    num_frames: int,
    eval_every: int,
    candidate_count: int,
    validation_count: int,
) -> Tuple[List[int], List[int], List[int]]:
    official = {0} | {
        index for index in range(1, num_frames) if (index + 1) % eval_every == 0
    }
    eligible = [index for index in range(1, num_frames) if index not in official]
    validation_positions = np.linspace(
        0, len(eligible) - 1, min(validation_count, len(eligible))
    ).round().astype(int)
    validation = sorted({eligible[int(position)] for position in validation_positions})
    validation_set = set(validation)
    training_pool = [index for index in eligible if index not in validation_set]
    candidate_positions = np.linspace(
        0, len(training_pool) - 1, min(candidate_count, len(training_pool))
    ).round().astype(int)
    candidates = sorted({training_pool[int(position)] for position in candidate_positions})
    assert not official.intersection(validation)
    assert not official.intersection(candidates)
    return candidates, validation, training_pool


def collect_frame_histograms(dataset, frames: Sequence[int], num_classes: int):
    histograms: Dict[int, torch.Tensor] = {}
    total = torch.zeros(num_classes, dtype=torch.float64)
    for ordinal, index in enumerate(frames, 1):
        target = frame_data(dataset, index)[-1][-1].detach().cpu()
        histogram = torch.bincount(target.flatten(), minlength=num_classes).double()
        histograms[int(index)] = histogram
        total += histogram
        if ordinal == 1 or ordinal % 40 == 0:
            print(f"[HCLR scan {ordinal}/{len(frames)}] frame={index}", flush=True)
    return histograms, total


@torch.no_grad()
def evaluate_validation(
    dataset,
    params,
    evaluation_decoder,
    baseline_weight: torch.Tensor,
    camera,
    frames: Sequence[int],
) -> Dict[str, float]:
    baseline_miou = []
    current_miou = []
    baseline_mbiou = []
    current_mbiou = []
    for index in frames:
        target = frame_data(dataset, index)[-1][-1]
        rendered_semantic = render_frame(params, camera, index)[2]
        baseline_prediction = torch.nn.functional.conv2d(
            rendered_semantic.unsqueeze(0), baseline_weight
        ).squeeze(0).argmax(dim=0)
        current_prediction = evaluation_decoder(
            rendered_semantic.unsqueeze(0)
        ).squeeze(0).argmax(dim=0)
        base_iou, base_biou = frame_semantic_scores(
            baseline_prediction, target, dataset.num_semantic_class
        )
        current_iou, current_biou = frame_semantic_scores(
            current_prediction, target, dataset.num_semantic_class
        )
        baseline_miou.append(base_iou * 100.0)
        current_miou.append(current_iou * 100.0)
        baseline_mbiou.append(base_biou * 100.0)
        current_mbiou.append(current_biou * 100.0)
    base_iou_array = np.asarray(baseline_miou)
    current_iou_array = np.asarray(current_miou)
    base_biou_array = np.asarray(baseline_mbiou)
    current_biou_array = np.asarray(current_mbiou)
    iou_delta = current_iou_array - base_iou_array
    biou_delta = current_biou_array - base_biou_array
    result = {
        "frames": len(frames),
        "baseline_miou_percent": float(base_iou_array.mean()),
        "current_miou_percent": float(current_iou_array.mean()),
        "miou_delta_pp": float(iou_delta.mean()),
        "baseline_mbiou_percent": float(base_biou_array.mean()),
        "current_mbiou_percent": float(current_biou_array.mean()),
        "mbiou_delta_pp": float(biou_delta.mean()),
        "miou_median_delta_pp": float(np.median(iou_delta)),
        "miou_p10_delta_pp": float(np.percentile(iou_delta, 10)),
        "miou_minimum_delta_pp": float(iou_delta.min()),
        "miou_maximum_delta_pp": float(iou_delta.max()),
        "miou_positive_frames": int((iou_delta > 0).sum()),
        "miou_negative_frames": int((iou_delta < 0).sum()),
        "miou_negative_rate": float((iou_delta < 0).mean()),
        "mbiou_median_delta_pp": float(np.median(biou_delta)),
    }
    temporal_bins = min(8, len(frames))
    bin_statistics = []
    for bin_index, positions in enumerate(np.array_split(np.arange(len(frames)), temporal_bins)):
        bin_statistics.append({
            "bin": bin_index,
            "first_frame": int(frames[int(positions[0])]),
            "last_frame": int(frames[int(positions[-1])]),
            "miou_delta_pp": float(iou_delta[positions].mean()),
            "mbiou_delta_pp": float(biou_delta[positions].mean()),
        })
    result["temporal_bins"] = bin_statistics
    result["worst_temporal_miou_delta_pp"] = min(
        item["miou_delta_pp"] for item in bin_statistics
    )
    result["worst_temporal_mbiou_delta_pp"] = min(
        item["mbiou_delta_pp"] for item in bin_statistics
    )
    result["selection_score"] = (
        validation_score(result)
        + 0.15 * min(0.0, result["worst_temporal_miou_delta_pp"])
        + 0.10 * min(0.0, result["worst_temporal_mbiou_delta_pp"])
    )
    return result


def load_saved_decoder(decoder, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    decoder.load_state_dict(state, strict=True)
    decoder.eval()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "RUNNING").write_text(str(os.getpid()), encoding="utf-8")
    started = time.time()
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.candidate_count = min(args.candidate_count, 6)
        args.selected_count = min(args.selected_count, 2)
        args.validation_count = min(args.validation_count, 2)
        args.validate_every = 1
        args.skip_official_eval = True

    device = torch.device(args.device)
    seed_everything(args.seed)
    if args.config_override:
        config_path = Path(args.config_override).resolve()
        config = SourceFileLoader(
            f"hclr_override_{source.name}", str(config_path)
        ).load_module().config
        if "tree_conflict_refinement" in config.get("mapping", {}):
            config["mapping"]["tree_conflict_refinement"]["enabled"] = False
    else:
        config = load_config(source)
    config["model"]["eval_gt_transfer"] = False
    params = load_params(source, device)
    dataset = build_dataset(config, source, device)
    num_frames_cfg = int(config["data"].get("num_frames", -1))
    num_frames = len(dataset) if num_frames_cfg == -1 else min(num_frames_cfg, len(dataset))
    decoder = load_decoder(source, dataset.num_semantic_class, device)
    camera = make_camera(dataset)
    widths = tuple(int(value) for value in dataset.num_semantic[:-1])
    if widths != (10, 4, 7):
        raise ValueError(f"HCLR requires PST widths (10,4,7), got {widths}")

    baseline_metrics = read_metric_directory(baseline_metric_directory(source))
    source_baseline_weight = decoder.weight.detach().clone()
    saved_decoder = output / "hclr_decoder.pt"
    if args.evaluate_only:
        if not saved_decoder.exists():
            raise FileNotFoundError(saved_decoder)
        load_saved_decoder(decoder, saved_decoder, device)
        evaluation_decoder = BaselineAnchoredActivationGate(
            decoder,
            source_baseline_weight,
            minimum_pixels=args.activation_minimum_pixels,
            minimum_area_ratio=args.activation_minimum_area_ratio,
        ).to(device).eval()
        official_metrics = run_official_evaluation(
            dataset, params, evaluation_decoder, config, output, num_frames
        )
        report_path = output / "hclr_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update({
            "baseline_metrics": baseline_metrics,
            "hclr_metrics": official_metrics,
            "delta": {
                key: official_metrics[key] - baseline_metrics[key]
                for key in official_metrics.keys() & baseline_metrics.keys()
            },
            "evaluation_resumed": True,
        })
        json_dump(report_path, report)
        (output / "RUNNING").unlink(missing_ok=True)
        (output / "RUN_COMPLETE").touch()
        return

    eval_every = int(config.get("eval_every", 5))
    candidates, validation_frames, training_pool = split_protocol(
        num_frames,
        eval_every,
        args.candidate_count,
        args.validation_count,
    )
    histograms, candidate_counts = collect_frame_histograms(
        dataset, candidates, dataset.num_semantic_class
    )
    selected_frames, selected_frames_by_bin = temporally_stratified_class_coverage(
        histograms,
        args.selected_count,
        num_bins=args.temporal_bins,
    )
    selected_counts = torch.stack([histograms[index] for index in selected_frames]).sum(dim=0)
    class_weights = bounded_class_weights(selected_counts.float()).to(device)
    row_radii = class_adaptive_radii(
        selected_counts.float(), args.minimum_row_radius, args.maximum_row_radius
    ).to(device)
    leaf_to_level = build_leaf_to_level_matrices(
        dataset.label_mapping_tree,
        widths,
        dataset.num_semantic_class,
        device=device,
    )

    for parameter in decoder.parameters():
        parameter.requires_grad_(True)
    decoder.train()
    baseline_weight = source_baseline_weight
    evaluation_decoder = BaselineAnchoredActivationGate(
        decoder,
        baseline_weight,
        minimum_pixels=args.activation_minimum_pixels,
        minimum_area_ratio=args.activation_minimum_area_ratio,
    ).to(device)
    baseline_state = {
        key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()
    }
    baseline_validation = evaluate_validation(
        dataset, params, evaluation_decoder, baseline_weight, camera, validation_frames
    )
    best_state = baseline_state
    best_step = 0
    best_validation = baseline_validation
    best_score = 0.0
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.0)
    weights = HCLRWeights()
    events = []
    generator = np.random.default_rng(args.seed)
    epoch_order: List[int] = []

    for step in range(1, args.steps + 1):
        if not epoch_order:
            epoch_order = [int(value) for value in generator.permutation(selected_frames)]
        frame = epoch_order.pop()
        _, gt_depth, _, _, semantic_gt = frame_data(dataset, frame)
        rendered = render_frame(params, camera, frame)
        rendered_semantic, rendered_depth, opacity = rendered[2], rendered[3], rendered[5]
        valid = reliable_mask(gt_depth, rendered_depth, opacity)
        loss, diagnostics = compute_hclr_loss(
            rendered_semantic,
            decoder,
            baseline_weight,
            semantic_gt,
            valid,
            class_weights,
            row_radii,
            leaf_to_level,
            widths=widths,
            weights=weights,
            confidence_threshold=args.protect_confidence,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0))
        optimizer.step()
        unclamped_norm = clamp_decoder_rows(decoder.weight, baseline_weight, row_radii)
        event = {
            "step": step,
            "frame": frame,
            "grad_norm": grad_norm,
            "maximum_unclamped_row_delta": float(unclamped_norm.max()),
            **{key: float(value) for key, value in diagnostics.items()},
        }
        events.append(event)
        if step == 1 or step % 8 == 0:
            print(
                f"[HCLR {step}/{args.steps}] frame={frame} loss={event['loss']:.5f} "
                f"focal={event['focal']:.5f} iou={event['soft_iou']:.5f} "
                f"stable={event['stable_ratio']:.4f}",
                flush=True,
            )
        del rendered_semantic, rendered, loss

        if step % args.validate_every == 0 or step == args.steps:
            evaluation_decoder.eval()
            validation = evaluate_validation(
                dataset,
                params,
                evaluation_decoder,
                baseline_weight,
                camera,
                validation_frames,
            )
            decoder.train()
            event["validation"] = validation
            print(
                f"[HCLR validation] step={step} d_mIoU={validation['miou_delta_pp']:.4f}pp "
                f"d_mBIoU={validation['mbiou_delta_pp']:.4f}pp "
                f"median={validation['miou_median_delta_pp']:.4f}pp "
                f"negative={validation['miou_negative_rate']:.3f} "
                f"score={validation['selection_score']:.4f}",
                flush=True,
            )
            eligible = (
                validation["miou_delta_pp"] >= 0.0
                and validation["mbiou_delta_pp"] >= 0.0
                and validation["selection_score"] > best_score + 1e-7
            )
            if eligible:
                best_score = validation["selection_score"]
                best_step = step
                best_validation = validation
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in decoder.state_dict().items()
                }

    decoder.load_state_dict(best_state, strict=True)
    evaluation_decoder.eval()
    torch.save(best_state, saved_decoder)
    torch.save(best_state, output / "Semantic.pth")
    torch.save(baseline_state, output / "Semantic_baseline.pth")
    torch.save(
        {
            "module": "HCLR",
            "baseline_weight": baseline_weight.detach().cpu(),
            "calibrated_decoder": best_state,
            "activation_minimum_pixels": args.activation_minimum_pixels,
            "activation_minimum_area_ratio": args.activation_minimum_area_ratio,
        },
        output / "HCLR_runtime.pt",
    )
    shutil.copy2(source / "config.py", output / "config.py")
    official_metrics = {}
    if not args.skip_official_eval:
        official_metrics = run_official_evaluation(
            dataset, params, evaluation_decoder, config, output, num_frames
        )
    report = {
        "module": "HCLR",
        "scene": config["data"]["sequence"],
        "source": str(source),
        "source_params_sha256": file_sha256(source / "params.npz"),
        "source_decoder_sha256": file_sha256(source / "Semantic.pth"),
        "frozen_components": list(params.keys()),
        "optimized_component": "leaf_decoder.Conv2d(21,102,1,bias=False)",
        "runtime_checkpoint": "HCLR_runtime.pt",
        "tree_widths": widths,
        "official_test_frames_used_during_training": False,
        "candidate_frames": candidates,
        "selected_frames": selected_frames,
        "selected_frames_by_temporal_bin": selected_frames_by_bin,
        "validation_frames": validation_frames,
        "training_pool_size": len(training_pool),
        "class_pixel_counts": selected_counts.tolist(),
        "class_weights": class_weights.detach().cpu().tolist(),
        "row_trust_radii": row_radii.detach().cpu().tolist(),
        "activation_gate": {
            "baseline_anchored": True,
            "minimum_pixels": args.activation_minimum_pixels,
            "minimum_area_ratio": args.activation_minimum_area_ratio,
            "threshold_at_dataset_resolution": evaluation_decoder.support_threshold(
                int(config["data"]["desired_image_height"]),
                int(config["data"]["desired_image_width"]),
            ),
        },
        "temporal_bin_count": args.temporal_bins,
        "steps_requested": args.steps,
        "best_step": best_step,
        "baseline_validation": baseline_validation,
        "best_validation": best_validation,
        "baseline_metrics": baseline_metrics,
        "hclr_metrics": official_metrics,
        "delta": {
            key: official_metrics[key] - baseline_metrics[key]
            for key in official_metrics.keys() & baseline_metrics.keys()
        },
        "events": events,
        "duration_seconds": time.time() - started,
        "smoke": bool(args.smoke),
    }
    json_dump(output / "hclr_report.json", report)
    (output / "RUNNING").unlink(missing_ok=True)
    (output / "RUN_COMPLETE").touch()
    print(json.dumps({key: report[key] for key in (
        "scene", "best_step", "best_validation", "baseline_metrics", "hclr_metrics", "delta"
    )}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
