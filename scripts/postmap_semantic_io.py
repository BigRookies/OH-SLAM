"""Shared I/O and rendering helpers for post-map semantic calibration."""

from __future__ import annotations

import hashlib
import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from datasets.gradslam_datasets import ReplicaDataset_semantic, load_dataset_config
from diff_gaussian_rasterization import GaussianRasterizer_semantic as RendererSemantic
from utils.recon_helpers import setup_camera
from utils.slam_helpers import transform_to_frame, transformed_params2rendervar_semantic


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def load_config(source: Path) -> Dict:
    config_path = source / "config.py"
    module = SourceFileLoader(
        f"postmap_source_{source.name}", str(config_path)
    ).load_module()
    return module.config


def load_params(source: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    arrays = dict(np.load(source / "params.npz", allow_pickle=True))
    params = {}
    for key, value in arrays.items():
        tensor = torch.as_tensor(value, device=device)
        if tensor.is_floating_point():
            tensor = tensor.float()
        params[key] = tensor.detach()
    if params["semantic"].ndim != 2 or params["semantic"].shape[1] != 21:
        raise ValueError(
            f"Post-map calibration requires [N,21] semantics, "
            f"got {tuple(params['semantic'].shape)}"
        )
    return params


def build_dataset(
    config: Dict, source: Path, device: torch.device
) -> ReplicaDataset_semantic:
    data = dict(config["data"])
    base_config = load_dataset_config(data["gradslam_data_cfg"])
    merged = {**base_config, **data}
    merged["results_dir"] = str(source)
    cache_name = merged.get(
        "semantic_tree_cache_name",
        f"semantic_labels_tree{merged.get('num_tree_level', 3)}level_test",
    )
    cache_dir = (
        Path(merged["basedir_sem"])
        / os.path.basename(merged["sequence"])
        / cache_name
    )
    merged["load_semantic_cache"] = cache_dir.is_dir()
    print(
        f"[Post-map] semantic cache: {cache_dir} "
        f"(enabled={merged['load_semantic_cache']})",
        flush=True,
    )
    merged.setdefault("ignore_bad", False)
    merged.setdefault("use_train_split", True)
    if merged["dataset_name"].lower() != "replica_semantic":
        raise ValueError("HCLR supports Replica semantic data")
    return ReplicaDataset_semantic(
        merged,
        merged["basedir"],
        os.path.basename(merged["sequence"]),
        start=merged["start"],
        end=merged["end"],
        stride=merged["stride"],
        desired_height=merged["desired_image_height"],
        desired_width=merged["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=merged["ignore_bad"],
        use_train_split=merged["use_train_split"],
    )


def load_decoder(
    source: Path, num_classes: int, device: torch.device
) -> torch.nn.Module:
    state = torch.load(source / "Semantic.pth", map_location=device)
    decoder = torch.nn.Conv2d(21, num_classes, kernel_size=1, bias=False).to(device)
    decoder.load_state_dict(state, strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder


def frame_data(dataset: ReplicaDataset_semantic, index: int):
    raw = dataset[index]
    color, depth, intrinsics, pose, semantic = raw[:5]
    return (
        color.permute(2, 0, 1) / 255.0,
        depth.permute(2, 0, 1),
        intrinsics[:3, :3],
        pose,
        semantic.long(),
    )


def make_camera(dataset: ReplicaDataset_semantic):
    color, _, intrinsics, pose, _ = frame_data(dataset, 0)
    first_w2c = torch.linalg.inv(pose)
    return setup_camera(
        color.shape[2],
        color.shape[1],
        intrinsics.cpu().numpy(),
        first_w2c.cpu().numpy(),
    )


def render_frame(
    params: Dict[str, torch.Tensor], camera, frame_index: int
):
    transformed = transform_to_frame(
        params, frame_index, gaussians_grad=False, camera_grad=False
    )
    render_vars = transformed_params2rendervar_semantic(params, transformed)
    return RendererSemantic(raster_settings=camera)(**render_vars)


def reliable_mask(
    gt_depth: torch.Tensor,
    rendered_depth: torch.Tensor,
    opacity: torch.Tensor,
    *,
    silhouette_threshold: float = 0.5,
    absolute_tolerance: float = 0.05,
    relative_tolerance: float = 0.03,
) -> torch.Tensor:
    gt = gt_depth.squeeze(0)
    estimate = rendered_depth.squeeze(0)
    alpha = opacity.squeeze(0)
    threshold = absolute_tolerance + relative_tolerance * gt.abs()
    valid = (
        (gt > 0)
        & torch.isfinite(gt)
        & torch.isfinite(estimate)
        & (alpha > silhouette_threshold)
        & ((estimate - gt).abs() <= threshold)
    )
    if float(valid.float().mean()) < 0.01:
        valid = (gt > 0) & torch.isfinite(estimate) & (alpha > silhouette_threshold)
    return valid


def read_metric_directory(path: Path) -> Dict[str, float]:
    metrics = {}
    for name in ("psnr", "ssim", "lpips", "l1", "rmse", "miou", "mbiou"):
        candidate = path / f"{name}.txt"
        if candidate.exists():
            values = np.atleast_1d(np.loadtxt(candidate))
            value = float(np.mean(values))
            if name in {"miou", "mbiou", "l1", "rmse"}:
                value *= 100.0
            metrics[name] = value
    return metrics


def baseline_metric_directory(source: Path) -> Path:
    """Return the frozen mapping evaluation used as the HCLR baseline."""
    regular_eval = source / "eval"
    if regular_eval.exists():
        return regular_eval
    raise FileNotFoundError(f"No baseline evaluation directory under {source}")


def run_official_evaluation(
    dataset,
    params,
    decoder,
    config: Dict,
    output: Path,
    num_frames: int,
):
    from utils.eval_helpers import eval_semantic_tree_newrender

    eval_dir = output / "eval"
    with torch.no_grad():
        eval_semantic_tree_newrender(
            dataset,
            params,
            num_frames,
            str(eval_dir),
            sil_thres=config["mapping"]["sil_thres"],
            mapping_iters=config["mapping"]["num_iters"],
            add_new_gaussians=config["mapping"]["add_new_gaussians"],
            eval_every=config["eval_every"],
            save_frames=False,
            flag_printtxt=True,
            flag_mlp=1,
            mlp_func=decoder,
            gt_transfer=False,
        )
    return read_metric_directory(eval_dir)
