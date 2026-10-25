"""Deterministic one-Gaussian CUDA smoke test for TCBR state mutation."""

import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.tree_conflict_refinement import observe_and_refine


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this integration smoke test")
    device = torch.device("cuda")
    frame_count = 2
    raw = {
        "means3D": torch.tensor([[0.0, 0.0, 1.0]], device=device),
        "rgb_colors": torch.tensor([[0.2, 0.3, 0.4]], device=device),
        "unnorm_rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "logit_opacities": torch.zeros((1, 1), device=device),
        "log_scales": torch.log(torch.tensor([[0.03]], device=device)),
        "semantic": torch.zeros((1, 21), device=device),
        "cam_unnorm_rots": torch.tensor(
            [[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], device=device
        ),
        "cam_trans": torch.zeros((1, 3, frame_count), device=device),
    }
    params = {key: torch.nn.Parameter(value) for key, value in raw.items()}
    optimizer = torch.optim.Adam(
        [{"params": [value], "name": key, "lr": 1e-3} for key, value in params.items()],
        lr=0.0,
    )
    # Initialize Adam state exactly as a completed mapping update would.
    sum(value.sum() * 0.0 for value in params.values()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    variables = {
        "seen": torch.ones(1, dtype=torch.bool, device=device),
        "means2D_gradient_accum": torch.zeros(1, device=device),
        "denom": torch.zeros(1, device=device),
        "max_2D_radius": torch.zeros(1, device=device),
        "timestep": torch.zeros(1, device=device),
    }
    height = width = 21
    labels = torch.zeros((4, height, width), dtype=torch.long, device=device)
    labels[0] = 2
    labels[1] = 1
    labels[2, :, :11] = 3
    labels[2, :, 11:] = 5
    labels[3, :, :11] = 20
    labels[3, :, 11:] = 21
    boundary = torch.zeros((height, width), device=device)
    boundary[:, 10:12] = 1
    current = {
        "semantic_label_gt": labels,
        "depth": torch.ones((1, height, width), device=device),
        "intrinsics": torch.tensor(
            [[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]],
            device=device,
        ),
    }
    config = {
        "tree_widths": (10, 4, 7),
        "min_gaussian_age": 0,
        "min_opacity": 0.05,
        "projection_candidate_cap": 10,
        "boundary_candidate_cap": 10,
        "min_support_radius_px": 3.0,
        "max_support_radius_px": 3.0,
        "depth_absolute_tolerance": 0.025,
        "depth_relative_tolerance": 0.02,
        "min_support_per_path": 2,
        "min_confirmed_views": 1,
        "max_splits_per_gaussian": 1,
        "split_distance_scale": 0.5,
        "min_world_shift": 0.00025,
        "max_world_shift": 0.01,
        "scale_shrink": 0.7,
        "semantic_logit_boost": 1.0,
    }
    before_alpha = torch.sigmoid(params["logit_opacities"])[0, 0].item()
    params, variables, stats = observe_and_refine(
        params, variables, optimizer, current, boundary, 1, config, split_budget=1
    )
    assert stats["path_conflicts"] == 1, stats
    assert stats["splits"] == 1, stats
    assert params["means3D"].shape == (2, 3)
    assert not torch.allclose(params["means3D"][0], params["means3D"][1])
    assert not torch.allclose(params["semantic"][0], params["semantic"][1])
    for key in ("means2D_gradient_accum", "denom", "max_2D_radius", "timestep"):
        assert variables[key].numel() == 2, (key, variables[key].shape)
    child_alpha = torch.sigmoid(params["logit_opacities"][:, 0]).detach().cpu().tolist()
    combined_alpha = 1.0 - (1.0 - child_alpha[0]) * (1.0 - child_alpha[1])
    assert abs(combined_alpha - before_alpha) < 1e-5
    for key in ("means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales", "semantic"):
        group = next(group for group in optimizer.param_groups if group["name"] == key)
        assert group["params"][0].shape[0] == 2
    print({"status": "PASS", "stats": stats})


if __name__ == "__main__":
    main()
