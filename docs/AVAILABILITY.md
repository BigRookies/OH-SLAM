# Code and data availability

Repository: https://github.com/BigRookies/OH-SLAM

The repository contains tracking and mapping code, TCBR refinement, HCLR post-map calibration, the canonical PST hierarchy, configuration files, evaluation utilities, and the eight-scene seed-0 semantic summary tables in `results/paper_seed0/`. Use a specific Git commit to identify the code and numerical tables used in a comparison.

Replica is a third-party dataset: https://github.com/facebookresearch/Replica-Dataset. The rendered RGB-D and semantic sequences follow the preparation linked in the main README and Hier-SLAM. These rendered sequences must be obtained separately under the applicable dataset terms. The original Replica repository and the rendered benchmark inputs are distinct resources.

RGB-D images, semantic-label images, trained checkpoints, full per-frame prediction archives, and raw training logs are not redistributed here. The CSV files contain archived numerical summaries, not new experiments or a complete raw-data archive. They support inspection of the paper's reported semantic comparisons; reproducing the complete runs also requires the external data and GPU environment described in the README.

Project code retains its existing Apache-2.0 license. Third-party code and datasets retain their respective licenses and terms.
