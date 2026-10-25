import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_public_config_uses_the_three_ohslam_modules():
    os.environ.setdefault("SCENE_NUM", "0")
    os.environ.setdefault("SEED", "0")
    from configs.replica.ohslam_semantic_run import config

    assert config["data"]["expected_semantic_channels"] == 21
    refinement = config["mapping"]["tree_conflict_refinement"]
    assert refinement["enabled"] is True
    assert refinement["boundary_source"] == "tree_gt"
    assert refinement["tree_widths"] == (10, 4, 7)


if __name__ == "__main__":
    test_public_config_uses_the_three_ohslam_modules()
    print("OH-SLAM configuration test passed")
