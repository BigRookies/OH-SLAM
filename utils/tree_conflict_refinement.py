"""Tree-path conflict driven boundary Gaussian refinement.

The module is deliberately independent from the rasterizer: it projects mature,
visible Gaussians explicitly, gathers RGB-D-supported tree paths around a frozen
boundary mask, and only refines a Gaussian after the same path conflict is seen
in multiple distinct frames.  It never consumes evaluation labels.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

from utils.slam_external import build_rotation, cat_params_to_optimizer, inverse_sigmoid


STATE_KEYS = (
    "tcbr_pair_code",
    "tcbr_pair_hits",
    "tcbr_last_frame",
    "tcbr_split_count",
)


@torch.no_grad()
def compute_tree_boundary_mask(
    semantic_labels: torch.Tensor,
    tree_widths: Sequence[int] = (10, 4, 7),
    boundary_width: int = 3,
) -> torch.Tensor:
    """Return a valid-pixel boundary band from hierarchical training labels.

    The mask compares complete tree paths between adjacent pixels, excludes
    invalid/void pairs, and then dilates the detected path transitions. The
    resulting band is the fixed boundary source used by TCBR.
    """
    widths = tuple(int(value) for value in tree_widths)
    if semantic_labels.ndim != 3 or semantic_labels.shape[0] < len(widths):
        raise ValueError(
            "Tree boundary extraction requires [levels, H, W] labels with "
            f"at least {len(widths)} levels"
        )
    paths = semantic_labels[: len(widths)].long()
    valid = torch.ones(paths.shape[1:], dtype=torch.bool, device=paths.device)
    for level, width in enumerate(widths):
        valid &= (paths[level] >= 0) & (paths[level] < width)

    boundary = torch.zeros_like(valid)
    if paths.shape[1] > 1:
        vertical = (
            valid[:-1, :]
            & valid[1:, :]
            & (paths[:, :-1, :] != paths[:, 1:, :]).any(dim=0)
        )
        boundary[:-1, :] |= vertical
        boundary[1:, :] |= vertical
    if paths.shape[2] > 1:
        horizontal = (
            valid[:, :-1]
            & valid[:, 1:]
            & (paths[:, :, :-1] != paths[:, :, 1:]).any(dim=0)
        )
        boundary[:, :-1] |= horizontal
        boundary[:, 1:] |= horizontal

    radius = max(0, int(boundary_width))
    if radius:
        boundary = F.max_pool2d(
            boundary.float()[None, None],
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[0, 0] > 0.5
    return boundary & valid


def path_strides(widths: Sequence[int]) -> Tuple[int, ...]:
    """Return mixed-radix strides for local tree-path indices."""
    widths = tuple(int(v) for v in widths)
    if not widths or any(v <= 0 for v in widths):
        raise ValueError(f"Invalid tree widths: {widths}")
    strides = []
    for idx in range(len(widths)):
        stride = 1
        for width in widths[idx + 1 :]:
            stride *= width
        strides.append(stride)
    return tuple(strides)


def encode_paths(paths: torch.Tensor, widths: Sequence[int]) -> torch.Tensor:
    """Encode [..., levels] local paths without merging reused child indices."""
    widths = tuple(int(v) for v in widths)
    if paths.shape[-1] != len(widths):
        raise ValueError(f"Expected {len(widths)} path levels, got {paths.shape[-1]}")
    strides = path_strides(widths)
    encoded = torch.zeros(paths.shape[:-1], dtype=torch.long, device=paths.device)
    for level, (width, stride) in enumerate(zip(widths, strides)):
        values = paths[..., level].long()
        if torch.any((values < 0) | (values >= width)):
            raise ValueError(f"Tree path level {level} is outside [0, {width})")
        encoded += values * stride
    return encoded


def decode_paths(codes: torch.Tensor, widths: Sequence[int]) -> torch.Tensor:
    widths = tuple(int(v) for v in widths)
    max_code = math.prod(widths)
    if torch.any((codes < 0) | (codes >= max_code)):
        raise ValueError(f"Tree code is outside [0, {max_code})")
    values = []
    for width, stride in zip(widths, path_strides(widths)):
        values.append(torch.div(codes, stride, rounding_mode="floor") % width)
    return torch.stack(values, dim=-1).long()


def first_divergent_level(path_a: torch.Tensor, path_b: torch.Tensor) -> torch.Tensor:
    if path_a.shape != path_b.shape or path_a.ndim != 2:
        raise ValueError("Paths must have the same [N, levels] shape")
    different = path_a != path_b
    if torch.any(~different.any(dim=1)):
        raise ValueError("Every refinement pair must contain two distinct paths")
    return different.to(torch.int64).argmax(dim=1)


def initialize_child_semantics(
    parent_semantic: torch.Tensor,
    path_a: torch.Tensor,
    path_b: torch.Tensor,
    widths: Sequence[int],
    boost: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preserve the shared prefix and separate only the conflicting suffix."""
    widths = tuple(int(v) for v in widths)
    if parent_semantic.shape[1] != sum(widths):
        raise ValueError("Semantic width does not match the configured tree")
    if path_a.shape != path_b.shape or path_a.shape[1] != len(widths):
        raise ValueError("Path tensors do not match the configured tree")
    semantic_a = parent_semantic.clone()
    semantic_b = parent_semantic.clone()
    first = first_divergent_level(path_a, path_b)
    rows = torch.arange(parent_semantic.shape[0], device=parent_semantic.device)
    offset = 0
    for level, width in enumerate(widths):
        active = first <= level
        active_rows = rows[active]
        if active_rows.numel():
            a = path_a[active, level]
            b = path_b[active, level]
            semantic_a[active_rows, offset + a] += boost
            semantic_a[active_rows, offset + b] -= boost
            semantic_b[active_rows, offset + b] += boost
            semantic_b[active_rows, offset + a] -= boost
        offset += width
    return semantic_a, semantic_b


def _resize_state(variables: Dict, count: int, device: torch.device) -> None:
    specs = {
        "tcbr_pair_code": (torch.int32, -1),
        "tcbr_pair_hits": (torch.uint8, 0),
        "tcbr_last_frame": (torch.int32, -1),
        "tcbr_split_count": (torch.uint8, 0),
    }
    existing = variables.get("tcbr_pair_code")
    old_count = int(existing.numel()) if isinstance(existing, torch.Tensor) else 0
    if old_count > count:
        # An external pruning operation changed the index map. Stale evidence is
        # unsafe, so discard it instead of associating it with the wrong Gaussian.
        old_count = 0
    for key, (dtype, fill) in specs.items():
        value = variables.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() != old_count:
            value = torch.full((old_count,), fill, dtype=dtype, device=device)
        else:
            value = value.to(device=device, dtype=dtype)
        if old_count < count:
            padding = torch.full((count - old_count,), fill, dtype=dtype, device=device)
            value = torch.cat((value, padding), dim=0)
        variables[key] = value


def _append_variable_rows(variables: Dict, new_count: int, time_idx: int, device: torch.device) -> None:
    for key in ("means2D_gradient_accum", "denom", "max_2D_radius"):
        if key in variables:
            variables[key] = torch.cat(
                (variables[key], torch.zeros(new_count, dtype=variables[key].dtype, device=device))
            )
    if "timestep" in variables:
        variables["timestep"] = torch.cat(
            (
                variables["timestep"],
                torch.full((new_count,), float(time_idx), dtype=variables["timestep"].dtype, device=device),
            )
        )
    for key, fill in (
        ("tcbr_pair_code", -1),
        ("tcbr_pair_hits", 0),
        ("tcbr_last_frame", -1),
        ("tcbr_split_count", 1),
    ):
        value = variables[key]
        variables[key] = torch.cat(
            (value, torch.full((new_count,), fill, dtype=value.dtype, device=device))
        )


def _sample_cap(indices: torch.Tensor, cap: int) -> torch.Tensor:
    if indices.numel() <= cap:
        return indices
    step = int(math.ceil(indices.numel() / float(cap)))
    return indices[::step][:cap]


def _split_gaussians(
    params: Dict[str, torch.Tensor],
    variables: Dict,
    optimizer: torch.optim.Optimizer,
    indices: torch.Tensor,
    directions_world: torch.Tensor,
    path_a: torch.Tensor,
    path_b: torch.Tensor,
    time_idx: int,
    cfg: Dict,
) -> Tuple[Dict, Dict]:
    if indices.numel() == 0:
        return params, variables
    device = params["means3D"].device
    widths = tuple(cfg.get("tree_widths", (10, 4, 7)))
    scale = torch.exp(params["log_scales"].detach()[indices]).max(dim=1).values
    shift = (scale * float(cfg.get("split_distance_scale", 0.5))).clamp(
        min=float(cfg.get("min_world_shift", 0.00025)),
        max=float(cfg.get("max_world_shift", 0.01)),
    )
    delta = F.normalize(directions_world, dim=1, eps=1e-8) * shift[:, None]
    original_means = params["means3D"].detach()[indices].clone()
    original_semantic = params["semantic"].detach()[indices].clone()
    semantic_a, semantic_b = initialize_child_semantics(
        original_semantic,
        path_a,
        path_b,
        widths,
        float(cfg.get("semantic_logit_boost", 1.0)),
    )
    shrink = float(cfg.get("scale_shrink", 0.7))
    if not 0.0 < shrink < 1.0:
        raise ValueError("scale_shrink must lie in (0, 1)")
    new_params = {
        key: value.detach()[indices].clone()
        for key, value in params.items()
        if key not in ("cam_unnorm_rots", "cam_trans")
    }
    new_params["means3D"] = original_means + delta
    new_params["semantic"] = semantic_b
    new_params["log_scales"] = new_params["log_scales"] + math.log(shrink)

    alpha = torch.sigmoid(params["logit_opacities"].detach()[indices])
    child_alpha = (1.0 - torch.sqrt((1.0 - alpha).clamp_min(1e-6))).clamp(1e-4, 1.0 - 1e-4)
    child_logit = inverse_sigmoid(child_alpha)
    new_params["logit_opacities"] = child_logit

    # Keep the original optimizer moments for the retained child, while the
    # appended child receives zero moments through cat_params_to_optimizer.
    params["means3D"].data[indices] = original_means - delta
    params["semantic"].data[indices] = semantic_a
    params["log_scales"].data[indices] += math.log(shrink)
    params["logit_opacities"].data[indices] = child_logit
    params = cat_params_to_optimizer(new_params, params, optimizer)

    variables["tcbr_split_count"][indices] = torch.clamp(
        variables["tcbr_split_count"][indices].to(torch.int16) + 1,
        max=255,
    ).to(torch.uint8)
    variables["tcbr_pair_code"][indices] = -1
    variables["tcbr_pair_hits"][indices] = 0
    variables["tcbr_last_frame"][indices] = int(time_idx)
    _append_variable_rows(variables, int(indices.numel()), time_idx, device)
    return params, variables


@torch.no_grad()
def observe_and_refine(
    params: Dict[str, torch.Tensor],
    variables: Dict,
    optimizer: torch.optim.Optimizer,
    curr_data: Dict,
    boundary_mask: torch.Tensor,
    time_idx: int,
    cfg: Dict,
    split_budget: int,
) -> Tuple[Dict, Dict, Dict[str, float]]:
    """Accumulate a frame observation and split confirmed conflict Gaussians."""
    started = time.perf_counter()
    device = params["means3D"].device
    count = int(params["means3D"].shape[0])
    _resize_state(variables, count, device)
    stats = {
        "visible_candidates": 0,
        "boundary_candidates": 0,
        "raw_path_conflicts": 0,
        "path_conflicts": 0,
        "confirmed": 0,
        "splits": 0,
        "milliseconds": 0.0,
    }
    if split_budget <= 0 or boundary_mask is None or count == 0:
        stats["milliseconds"] = (time.perf_counter() - started) * 1000.0
        return params, variables, stats
    labels = curr_data["semantic_label_gt"]
    widths = tuple(int(v) for v in cfg.get("tree_widths", (10, 4, 7)))
    if labels.ndim != 3 or labels.shape[0] < len(widths):
        raise ValueError("Refinement requires a [levels, H, W] semantic tree target")
    height, width = int(labels.shape[-2]), int(labels.shape[-1])
    boundary = boundary_mask.to(device=device, dtype=torch.float32)
    if boundary.shape != (height, width):
        boundary = F.interpolate(
            boundary[None, None], size=(height, width), mode="nearest"
        )[0, 0]

    visible = variables.get("seen")
    if not isinstance(visible, torch.Tensor) or visible.numel() != count:
        visible = torch.ones(count, dtype=torch.bool, device=device)
    else:
        visible = visible.to(device=device, dtype=torch.bool)
    mature = variables.get("timestep", torch.zeros(count, device=device)) <= (
        int(time_idx) - int(cfg.get("min_gaussian_age", 16))
    )
    opacity = torch.sigmoid(params["logit_opacities"].detach()).squeeze(-1)
    candidates = torch.nonzero(
        visible
        & mature
        & (opacity >= float(cfg.get("min_opacity", 0.05)))
        & (variables["tcbr_split_count"] < int(cfg.get("max_splits_per_gaussian", 1))),
        as_tuple=False,
    ).squeeze(1)
    candidates = _sample_cap(candidates, int(cfg.get("projection_candidate_cap", 200000)))
    stats["visible_candidates"] = int(candidates.numel())
    if candidates.numel() == 0:
        stats["milliseconds"] = (time.perf_counter() - started) * 1000.0
        return params, variables, stats

    cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
    rotation = build_rotation(cam_rot)[0]
    translation = params["cam_trans"][..., time_idx].detach().reshape(3)
    world = params["means3D"].detach()[candidates]
    camera = world @ rotation.T + translation
    z = camera[:, 2]
    intrinsics = curr_data["intrinsics"].to(device=device)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * camera[:, 0] / z.clamp_min(1e-6) + cx
    v = fy * camera[:, 1] / z.clamp_min(1e-6) + cy
    scale = torch.exp(params["log_scales"].detach()[candidates]).max(dim=1).values
    radius = (fx.abs() * scale / z.clamp_min(1e-6)).clamp(
        min=float(cfg.get("min_support_radius_px", 2.0)),
        max=float(cfg.get("max_support_radius_px", 6.0)),
    )
    max_radius = int(math.ceil(float(cfg.get("max_support_radius_px", 6.0))))
    boundary_near = F.max_pool2d(
        boundary[None, None], kernel_size=2 * max_radius + 1, stride=1, padding=max_radius
    )[0, 0]
    ui = u.round().long().clamp(0, width - 1)
    vi = v.round().long().clamp(0, height - 1)
    inside = (
        (z > 1e-4)
        & (u >= max_radius)
        & (u < width - max_radius)
        & (v >= max_radius)
        & (v < height - max_radius)
    )
    near = inside & (boundary_near[vi, ui] > 0.5)
    keep = torch.nonzero(near, as_tuple=False).squeeze(1)
    if keep.numel() == 0:
        stats["milliseconds"] = (time.perf_counter() - started) * 1000.0
        return params, variables, stats
    keep = _sample_cap(keep, int(cfg.get("boundary_candidate_cap", 4096)))
    gaussian_indices = candidates[keep]
    u, v, z, radius = u[keep], v[keep], z[keep], radius[keep]
    stats["boundary_candidates"] = int(gaussian_indices.numel())

    # Nine support samples: center, cardinal directions and diagonals. The
    # per-Gaussian radius follows its projected spatial support.
    template = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
         [0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, 0.7071], [-0.7071, -0.7071]],
        dtype=torch.float32,
        device=device,
    )
    sample_u = (u[:, None] + radius[:, None] * template[None, :, 0]).round().long().clamp(0, width - 1)
    sample_v = (v[:, None] + radius[:, None] * template[None, :, 1]).round().long().clamp(0, height - 1)
    sampled_paths = labels[: len(widths), sample_v, sample_u].permute(1, 2, 0).long()
    valid_path = torch.ones(sampled_paths.shape[:2], dtype=torch.bool, device=device)
    for level, level_width in enumerate(widths):
        valid_path &= (sampled_paths[..., level] >= 0) & (sampled_paths[..., level] < level_width)
    safe_paths = sampled_paths.clone()
    safe_paths[~valid_path] = 0
    codes = encode_paths(safe_paths, widths)
    depth = curr_data["depth"][0].to(device=device)
    sample_depth = depth[sample_v, sample_u]
    depth_tolerance = torch.maximum(
        torch.full_like(z, float(cfg.get("depth_absolute_tolerance", 0.025))),
        z.abs() * float(cfg.get("depth_relative_tolerance", 0.02)),
    )
    valid = valid_path & (sample_depth > 0) & (
        (sample_depth - z[:, None]).abs() <= depth_tolerance[:, None]
    )
    code_count = math.prod(widths)
    counts = torch.zeros((codes.shape[0], code_count), dtype=torch.int16, device=device)
    counts.scatter_add_(1, codes, valid.to(torch.int16))
    top_counts, top_codes = counts.topk(k=2, dim=1)
    minimum_support = int(cfg.get("min_support_per_path", 2))
    raw_conflict = top_counts[:, 1] >= minimum_support
    stats["raw_path_conflicts"] = int(raw_conflict.sum())
    support_ratio = top_counts[:, 1].to(torch.float32) / (
        top_counts[:, 0].clamp_min(1).to(torch.float32)
    )
    enough = (
        raw_conflict
        & (support_ratio >= float(cfg.get("min_secondary_support_ratio", 0.0)))
        & (top_counts.sum(dim=1) >= int(cfg.get("min_total_path_support", 0)))
    )
    if not enough.any():
        stats["milliseconds"] = (time.perf_counter() - started) * 1000.0
        return params, variables, stats
    gaussian_indices = gaussian_indices[enough]
    top_counts = top_counts[enough]
    top_codes = top_codes[enough]
    support_ratio = support_ratio[enough]
    sample_u, sample_v, sample_depth, codes, valid = (
        tensor[enough] for tensor in (sample_u, sample_v, sample_depth, codes, valid)
    )
    stats["path_conflicts"] = int(gaussian_indices.numel())

    code_a, code_b = top_codes[:, 0], top_codes[:, 1]
    low, high = torch.minimum(code_a, code_b), torch.maximum(code_a, code_b)
    pair_code = low * code_count + high
    last_frame = variables["tcbr_last_frame"][gaussian_indices]
    previous_pair = variables["tcbr_pair_code"][gaussian_indices].long()
    previous_hits = variables["tcbr_pair_hits"][gaussian_indices].to(torch.int16)
    new_frame = last_frame != int(time_idx)
    same_pair = previous_pair == pair_code
    hits = torch.where(same_pair, previous_hits + new_frame.to(torch.int16), new_frame.to(torch.int16))
    hits = hits.clamp(max=255).to(torch.uint8)
    variables["tcbr_pair_code"][gaussian_indices] = pair_code.to(torch.int32)
    variables["tcbr_pair_hits"][gaussian_indices] = hits
    variables["tcbr_last_frame"][gaussian_indices[new_frame]] = int(time_idx)
    confirmed_mask = hits >= int(cfg.get("min_confirmed_views", 2))
    stats["confirmed"] = int(confirmed_mask.sum())
    if not confirmed_mask.any():
        stats["milliseconds"] = (time.perf_counter() - started) * 1000.0
        return params, variables, stats

    gaussian_indices = gaussian_indices[confirmed_mask]
    code_a, code_b = code_a[confirmed_mask], code_b[confirmed_mask]
    top_counts = top_counts[confirmed_mask]
    support_ratio = support_ratio[confirmed_mask]
    sample_u, sample_v, sample_depth, codes, valid = (
        tensor[confirmed_mask] for tensor in (sample_u, sample_v, sample_depth, codes, valid)
    )
    # Prefer conflicts with stronger, more balanced support. The selection only
    # orders confirmed candidates; it does not change the hard acceptance tests.
    strength = top_counts[:, 1].to(torch.float32) + support_ratio
    # The project pins an older PyTorch release whose CUDA argsort does not
    # expose the newer ``stable`` keyword.
    order = torch.argsort(strength, descending=True)[: int(split_budget)]
    gaussian_indices = gaussian_indices[order]
    code_a, code_b = code_a[order], code_b[order]
    sample_u, sample_v, sample_depth, codes, valid = (
        tensor[order] for tensor in (sample_u, sample_v, sample_depth, codes, valid)
    )
    mask_a = valid & (codes == code_a[:, None])
    mask_b = valid & (codes == code_b[:, None])
    points_camera = torch.stack(
        (
            (sample_u.to(torch.float32) - cx) / fx * sample_depth,
            (sample_v.to(torch.float32) - cy) / fy * sample_depth,
            sample_depth,
        ),
        dim=-1,
    )
    mean_a = (points_camera * mask_a[..., None]).sum(dim=1) / mask_a.sum(dim=1).clamp_min(1)[:, None]
    mean_b = (points_camera * mask_b[..., None]).sum(dim=1) / mask_b.sum(dim=1).clamp_min(1)[:, None]
    directions_camera = mean_b - mean_a
    directions_world = directions_camera @ rotation
    valid_direction = torch.linalg.vector_norm(directions_world, dim=1) > 1e-6
    gaussian_indices = gaussian_indices[valid_direction]
    directions_world = directions_world[valid_direction]
    code_a, code_b = code_a[valid_direction], code_b[valid_direction]
    if gaussian_indices.numel() == 0:
        stats["milliseconds"] = (time.perf_counter() - started) * 1000.0
        return params, variables, stats
    path_a = decode_paths(code_a, widths)
    path_b = decode_paths(code_b, widths)
    params, variables = _split_gaussians(
        params,
        variables,
        optimizer,
        gaussian_indices,
        directions_world,
        path_a,
        path_b,
        time_idx,
        cfg,
    )
    stats["splits"] = int(gaussian_indices.numel())
    stats["milliseconds"] = (time.perf_counter() - started) * 1000.0
    return params, variables, stats
