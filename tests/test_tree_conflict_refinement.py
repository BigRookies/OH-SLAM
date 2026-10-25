import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.tree_conflict_refinement import (
    compute_tree_boundary_mask,
    decode_paths,
    encode_paths,
    first_divergent_level,
    initialize_child_semantics,
)


def test_tree_path_codec_keeps_reused_local_indices_distinct():
    paths = torch.tensor([[0, 1, 2], [1, 1, 2], [9, 3, 6]])
    codes = encode_paths(paths, (10, 4, 7))
    assert codes.unique().numel() == paths.shape[0]
    assert torch.equal(decode_paths(codes, (10, 4, 7)), paths)


def test_first_divergence_and_suffix_only_initialization():
    path_a = torch.tensor([[2, 1, 3], [2, 1, 3]])
    path_b = torch.tensor([[2, 1, 5], [2, 3, 0]])
    assert first_divergent_level(path_a, path_b).tolist() == [2, 1]
    original = torch.zeros(2, 21)
    semantic_a, semantic_b = initialize_child_semantics(
        original, path_a, path_b, (10, 4, 7), boost=1.5
    )
    # Shared prefix stays unchanged for the fine-only conflict.
    assert torch.equal(semantic_a[0, :14], original[0, :14])
    assert semantic_a[0, 14 + 3] == 1.5
    assert semantic_a[0, 14 + 5] == -1.5
    assert semantic_b[0, 14 + 5] == 1.5
    # The second path differs at level 1, so level 0 alone stays unchanged.
    assert torch.equal(semantic_a[1, :10], original[1, :10])
    assert not torch.equal(semantic_a[1, 10:], original[1, 10:])


def test_invalid_identical_pair_is_rejected():
    path = torch.tensor([[1, 2, 3]])
    try:
        first_divergent_level(path, path)
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("An identical path pair must not trigger refinement")


def test_tree_boundary_mask_detects_path_transition():
    labels = torch.zeros((4, 9, 11), dtype=torch.long)
    labels[0] = 2
    labels[1] = 1
    labels[2, :, 6:] = 5
    mask = compute_tree_boundary_mask(labels, (10, 4, 7), boundary_width=0)
    assert mask.dtype == torch.bool
    assert mask[:, 5:7].all()
    assert not mask[:, :5].any()
    assert not mask[:, 7:].any()


def test_tree_boundary_mask_excludes_void_pairs():
    labels = torch.zeros((4, 7, 9), dtype=torch.long)
    labels[0] = 2
    labels[1] = 1
    labels[2] = 3
    labels[:3, :, 4:] = -1
    mask = compute_tree_boundary_mask(labels, (10, 4, 7), boundary_width=2)
    assert not mask.any()


if __name__ == "__main__":
    test_tree_path_codec_keeps_reused_local_indices_distinct()
    test_first_divergence_and_suffix_only_initialization()
    test_invalid_identical_pair_is_rejected()
    test_tree_boundary_mask_detects_path_transition()
    test_tree_boundary_mask_excludes_void_pairs()
    print("TCBR CPU tests passed")
