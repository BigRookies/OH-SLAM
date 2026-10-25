#!/usr/bin/env python3
"""Headless visualization for OH-SLAM semantic Gaussians and camera poses."""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENE_ORDER = ("room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4")
TREE_WIDTHS = (10, 4, 7)


def sigmoid(values):
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def label_colormap(size):
    """Generate the VOC/imgviz-style deterministic label palette."""
    cmap = np.zeros((size, 3), dtype=np.uint8)
    for label in range(size):
        value = label
        bit = 0
        while value:
            cmap[label, 0] |= ((value >> 0) & 1) << (7 - bit)
            cmap[label, 1] |= ((value >> 1) & 1) << (7 - bit)
            cmap[label, 2] |= ((value >> 2) & 1) << (7 - bit)
            bit += 1
            value >>= 3
    return cmap


def quaternion_to_rotation(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def extract_camera_poses(params):
    rotations = np.asarray(params["cam_unnorm_rots"])
    translations = np.asarray(params["cam_trans"])
    frame_count = rotations.shape[-1]
    c2w_all = []
    for frame_idx in range(frame_count):
        quaternion = rotations[..., frame_idx].reshape(-1, 4)[0]
        translation = translations[..., frame_idx].reshape(-1, 3)[0]
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = quaternion_to_rotation(quaternion)
        w2c[:3, 3] = translation
        c2w_all.append(np.linalg.inv(w2c))
    return np.stack(c2w_all)


def find_conv_head(state, feature_dim):
    try:
        import torch
    except ImportError:
        return None, None, "tree_path_fallback"

    if isinstance(state, dict):
        for wrapper in ("state_dict", "model_state_dict", "model"):
            if wrapper in state and isinstance(state[wrapper], dict):
                state = state[wrapper]
                break
    if not isinstance(state, dict):
        return None, None, "tree_path_fallback"

    candidates = []
    for key, value in state.items():
        if not hasattr(value, "detach"):
            continue
        array = value.detach().cpu().numpy()
        if array.ndim == 4 and array.shape[1] == feature_dim and array.shape[2:] == (1, 1):
            candidates.append((key, array[:, :, 0, 0]))
        elif array.ndim == 2 and array.shape[1] == feature_dim:
            candidates.append((key, array))
    if not candidates:
        return None, None, "tree_path_fallback"

    key, weight = max(candidates, key=lambda item: item[1].shape[0])
    bias = None
    prefix = key.rsplit("weight", 1)[0]
    bias_key = prefix + "bias"
    if bias_key in state and hasattr(state[bias_key], "detach"):
        bias = state[bias_key].detach().cpu().numpy().reshape(-1)
    return weight.astype(np.float32), bias, "semantic_head:" + key


def semantic_colors(features, model_path, batch_size=100000):
    feature_dim = features.shape[1]
    weight = bias = None
    mode = "tree_path_fallback"
    if model_path.exists():
        try:
            import torch

            try:
                state = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location="cpu")
            weight, bias, mode = find_conv_head(state, feature_dim)
        except Exception as exc:
            mode = "tree_path_fallback:" + type(exc).__name__

    if weight is not None:
        labels = np.empty(features.shape[0], dtype=np.int32)
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            logits = features[start:end].astype(np.float32, copy=False) @ weight.T
            if bias is not None:
                logits += bias
            labels[start:end] = np.argmax(logits, axis=1)
        palette = label_colormap(weight.shape[0])
        return labels, palette[labels], mode, int(weight.shape[0])

    if feature_dim != sum(TREE_WIDTHS):
        raise ValueError(
            "No compatible semantic head and unexpected feature dimension: "
            f"got {feature_dim}, expected {sum(TREE_WIDTHS)}"
        )
    offsets = np.cumsum((0,) + TREE_WIDTHS)
    level_labels = [
        np.argmax(features[:, offsets[idx] : offsets[idx + 1]], axis=1)
        for idx in range(len(TREE_WIDTHS))
    ]
    labels = (level_labels[0] * TREE_WIDTHS[1] + level_labels[1]) * TREE_WIDTHS[2] + level_labels[2]
    palette = label_colormap(math.prod(TREE_WIDTHS))
    return labels.astype(np.int32), palette[labels], mode, int(math.prod(TREE_WIDTHS))


def robust_limits(points, low=0.5, high=99.5):
    return np.percentile(points, [low, high], axis=0)


def choose_up_axis(points, requested):
    names = ("x", "y", "z")
    if requested != "auto":
        return names.index(requested)
    limits = robust_limits(points)
    spans = limits[1] - limits[0]
    return int(np.argmin(spans))


def sample_indices(count, maximum, seed=0):
    if count <= maximum:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=maximum, replace=False))


def set_3d_equal(ax, points):
    limits = robust_limits(points)
    center = (limits[0] + limits[1]) * 0.5
    spans = np.maximum(limits[1] - limits[0], 1e-3)
    radius = float(np.max(spans) * 0.53)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def save_semantic_3d(path, scene, points, colors, trajectory):
    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255.0, s=0.18, alpha=0.80, linewidths=0, rasterized=True)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="#111111", linewidth=1.1, label="Estimated trajectory")
    ax.scatter(*trajectory[0], c="#00a651", s=38, marker="o", label="Start")
    ax.scatter(*trajectory[-1], c="#e31a1c", s=44, marker="X", label="End")
    set_3d_equal(ax, np.vstack((points, trajectory)))
    ax.view_init(elev=32, azim=-58)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{scene}: reconstructed semantic Gaussian map")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_topdown(path, scene, points, colors, trajectory, up_axis, overlay=True):
    names = ("X", "Y", "Z")
    horizontal = [axis for axis in range(3) if axis != up_axis]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    ax.scatter(points[:, horizontal[0]], points[:, horizontal[1]], c=colors / 255.0, s=0.20, alpha=0.82, linewidths=0, rasterized=True)
    if overlay:
        ax.plot(trajectory[:, horizontal[0]], trajectory[:, horizontal[1]], color="#111111", linewidth=1.4, label="Estimated trajectory")
        ax.scatter(trajectory[0, horizontal[0]], trajectory[0, horizontal[1]], c="#00a651", s=42, label="Start", zorder=5)
        ax.scatter(trajectory[-1, horizontal[0]], trajectory[-1, horizontal[1]], c="#e31a1c", s=48, marker="X", label="End", zorder=5)
        ax.legend(loc="best", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{names[horizontal[0]]} (m)")
    ax.set_ylabel(f"{names[horizontal[1]]} (m)")
    suffix = " with estimated trajectory" if overlay else ""
    ax.set_title(f"{scene}: top-down semantic map{suffix}")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_trajectory_figure(path, scene, c2w_all, up_axis):
    trajectory = c2w_all[:, :3, 3]
    names = ("X", "Y", "Z")
    horizontal = [axis for axis in range(3) if axis != up_axis]
    fig = plt.figure(figsize=(14, 6), dpi=180)
    ax3d = fig.add_subplot(121, projection="3d")
    frame_color = np.linspace(0.0, 1.0, len(trajectory))
    ax3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="#2c7fb8", linewidth=1.5)
    scatter = ax3d.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c=frame_color, cmap="viridis", s=3)
    ax3d.scatter(*trajectory[0], c="#00a651", s=42, label="Start")
    ax3d.scatter(*trajectory[-1], c="#e31a1c", s=48, marker="X", label="End")
    step = max(1, len(trajectory) // 20)
    forward = c2w_all[::step, :3, 2]
    origins = trajectory[::step]
    ax3d.quiver(origins[:, 0], origins[:, 1], origins[:, 2], forward[:, 0], forward[:, 1], forward[:, 2], length=0.12, normalize=True, color="#d95f0e", linewidth=0.7)
    set_3d_equal(ax3d, trajectory)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D estimated camera poses")
    ax3d.legend(loc="best")

    ax2d = fig.add_subplot(122)
    ax2d.plot(trajectory[:, horizontal[0]], trajectory[:, horizontal[1]], color="#2c7fb8", linewidth=1.4)
    ax2d.scatter(trajectory[:, horizontal[0]], trajectory[:, horizontal[1]], c=frame_color, cmap="viridis", s=4)
    ax2d.scatter(trajectory[0, horizontal[0]], trajectory[0, horizontal[1]], c="#00a651", s=42, label="Start")
    ax2d.scatter(trajectory[-1, horizontal[0]], trajectory[-1, horizontal[1]], c="#e31a1c", s=48, marker="X", label="End")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel(f"{names[horizontal[0]]} (m)")
    ax2d.set_ylabel(f"{names[horizontal[1]]} (m)")
    ax2d.set_title("Top-down estimated trajectory")
    ax2d.grid(alpha=0.2)
    ax2d.legend(loc="best")
    fig.colorbar(scatter, ax=[ax3d, ax2d], shrink=0.75, label="Normalized frame index")
    fig.suptitle(f"{scene}: estimated camera trajectory", fontsize=14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_point_ply(path, points, colors, labels, opacities):
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("semantic_label", "<i4"), ("opacity", "<f4"),
        ]
    )
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
    vertices["red"], vertices["green"], vertices["blue"] = colors.T.astype(np.uint8)
    vertices["semantic_label"] = labels.astype(np.int32)
    vertices["opacity"] = opacities.astype(np.float32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int semantic_label\nproperty float opacity\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def write_trajectory_ply(path, trajectory):
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(trajectory)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element edge {max(0, len(trajectory) - 1)}\n")
        handle.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for idx, point in enumerate(trajectory):
            fraction = idx / max(1, len(trajectory) - 1)
            color = plt.get_cmap("viridis")(fraction)[:3]
            rgb = (np.asarray(color) * 255).astype(int)
            handle.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {rgb[0]} {rgb[1]} {rgb[2]}\n")
        for idx in range(len(trajectory) - 1):
            handle.write(f"{idx} {idx + 1}\n")


def scene_name_from_dir(path):
    match = re.match(r"(room[0-9]+|office[0-9]+)", path.name)
    return match.group(1) if match else path.name


def process_scene(scene_dir, output_root, args):
    scene = scene_name_from_dir(scene_dir)
    output_dir = output_root / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = scene_dir / "params.npz"
    model_path = scene_dir / "Semantic.pth"
    with np.load(params_path, allow_pickle=True) as data:
        params = {key: data[key] for key in data.files}

    means = np.asarray(params["means3D"], dtype=np.float32)
    features = np.asarray(params["semantic"], dtype=np.float32)
    opacity = sigmoid(np.asarray(params["logit_opacities"], dtype=np.float32).reshape(-1))
    finite = np.isfinite(means).all(axis=1) & np.isfinite(features).all(axis=1) & np.isfinite(opacity)
    keep = finite & (opacity >= args.opacity_threshold)
    if not np.any(keep):
        raise RuntimeError(f"{scene}: no Gaussians remain after opacity filtering")
    means, features, opacity = means[keep], features[keep], opacity[keep]
    labels, colors, semantic_mode, class_count = semantic_colors(features, model_path)
    c2w_all = extract_camera_poses(params)
    trajectory = c2w_all[:, :3, 3]
    up_axis = choose_up_axis(means, args.up_axis)

    plot_idx = sample_indices(len(means), args.max_plot_points, seed=args.seed)
    plot_points, plot_colors = means[plot_idx], colors[plot_idx]
    save_semantic_3d(output_dir / "semantic_map_3d.png", scene, plot_points, plot_colors, trajectory)
    save_topdown(output_dir / "semantic_map_topdown.png", scene, plot_points, plot_colors, trajectory, up_axis, overlay=False)
    save_topdown(output_dir / "semantic_map_with_trajectory.png", scene, plot_points, plot_colors, trajectory, up_axis, overlay=True)
    save_trajectory_figure(output_dir / "estimated_camera_trajectory.png", scene, c2w_all, up_axis)
    write_point_ply(output_dir / "semantic_map.ply", means, colors, labels, opacity)
    write_trajectory_ply(output_dir / "estimated_trajectory.ply", trajectory)

    flat_poses = c2w_all.reshape(len(c2w_all), 16)
    pose_rows = np.column_stack((np.arange(len(c2w_all), dtype=np.int32), flat_poses))
    np.savetxt(
        output_dir / "estimated_poses_c2w.txt",
        pose_rows,
        fmt=["%d"] + ["%.9f"] * 16,
        header="frame c2w_00 c2w_01 c2w_02 c2w_03 c2w_10 c2w_11 c2w_12 c2w_13 c2w_20 c2w_21 c2w_22 c2w_23 c2w_30 c2w_31 c2w_32 c2w_33",
    )
    metadata = {
        "scene": scene,
        "source": str(scene_dir),
        "gaussians_total": int(len(keep)),
        "gaussians_exported": int(np.count_nonzero(keep)),
        "semantic_feature_dim": int(features.shape[1]),
        "semantic_decode_mode": semantic_mode,
        "semantic_class_count": class_count,
        "camera_pose_count": int(len(c2w_all)),
        "pose_convention": "camera-to-world matrices reconstructed from cam_unnorm_rots and cam_trans",
        "up_axis": ("x", "y", "z")[up_axis],
        "opacity_threshold": args.opacity_threshold,
        "plot_points": int(len(plot_idx)),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[{scene}] gaussians={len(means)}, poses={len(c2w_all)}, "
        f"semantic={semantic_mode}, output={output_dir}"
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path, help="Directory containing all scene result directories")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--opacity-threshold", type=float, default=0.05)
    parser.add_argument("--max-plot-points", type=int, default=150000)
    parser.add_argument("--up-axis", choices=("auto", "x", "y", "z"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    output_root = (args.output_root or experiment_root.with_name(experiment_root.name + "_visualization")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene_dirs = [path for path in experiment_root.iterdir() if path.is_dir() and (path / "params.npz").exists()]
    order = {name: idx for idx, name in enumerate(SCENE_ORDER)}
    scene_dirs.sort(key=lambda path: order.get(scene_name_from_dir(path), len(order)))
    if not scene_dirs:
        raise SystemExit(f"No scene directories containing params.npz under {experiment_root}")
    summaries = []
    for scene_dir in scene_dirs:
        summaries.append(process_scene(scene_dir, output_root, args))
    (output_root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Completed {len(summaries)} scene(s). Output root: {output_root}")


if __name__ == "__main__":
    main()
