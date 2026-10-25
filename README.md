# OH-SLAM

Research implementation of **OH-SLAM**, a supervised RGB-D semantic Gaussian SLAM system. Inspired by the hierarchical semantic coding in [Hier-SLAM](https://github.com/LeeBY68/Hier-SLAM), OH-SLAM reconstructs a path-encoded semantic tree and combines it with conflict-driven spatial refinement and map-preserving leaf recalibration. The implementation reuses Hier-SLAM infrastructure; upstream attribution and licenses are retained.

OH-SLAM represents Replica semantics with a compact three-level tree and improves the final semantic map through two hierarchy-aware components:

- **PST (path-encoded semantic tree)** encodes 102 output labels with a `10 + 4 + 7 = 21` dimensional path.
- **TCBR** identifies persistent multi-view tree-path conflicts and directionally refines Gaussians near semantic boundaries.
- **HCLR** recalibrates the leaf decoder after mapping while keeping camera poses, geometry, appearance, opacity, and Gaussian semantic features fixed.

The accompanying manuscript is **OH-SLAM with hierarchical semantic boundary refinement and map-preserving leaf recalibration**. The repository contains its source code and numerical comparison tables.

See [Reproducibility and evaluation](docs/REPRODUCIBILITY.md) for the paper protocol, limitations, and the archived eight-scene semantic results.

## Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Run](#run)
- [Evaluation and visualization](#evaluation-and-visualization)
- [Fixed semantic hierarchy](#fixed-semantic-hierarchy)
- [Reproducibility and reported results](docs/REPRODUCIBILITY.md)
- [Code and data availability](docs/AVAILABILITY.md)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Installation

The reference environment uses Ubuntu 20.04, Python 3.10, PyTorch 1.12.1, and CUDA 11.6.

```bash
git clone https://github.com/BigRookies/OH-SLAM.git
cd OH-SLAM

conda env create -f environment.yml
conda activate hierslam

bash recompile_cuda_extension.sh
```

The differentiable Gaussian rasterizer is configured for 21 semantic channels. Recompile the extension after changing the CUDA environment or semantic channel count.

## Dataset

### Replica

OH-SLAM follows the Replica RGB-D and semantic-label preparation used by Hier-SLAM. Download the Replica RGB-D sequences with:

```bash
bash bash_scripts/download_replica.sh
```

Obtain the rendered Replica semantic labels by following the [Hier-SLAM download instructions](https://github.com/LeeBY68/Hier-SLAM#replica), then arrange the dataset as:

```text
data/Replica_semantic/
├── info_semantic_tree.json
├── room0/
│   ├── results/
│   ├── semantic_class/
│   ├── traj.txt
│   └── info_semantic_tree.json
├── room1/
├── room2/
├── office0/
├── office1/
├── office2/
├── office3/
└── office4/
```

This repository provides a canonical PST definition at `data/Replica_semantic/info_semantic_tree.json`. Copy it into every scene directory:

```bash
for scene in room0 room1 room2 office0 office1 office2 office3 office4; do
  cp data/Replica_semantic/info_semantic_tree.json \
     "data/Replica_semantic/${scene}/info_semantic_tree.json"
done
```

RGB-D frames, semantic images, checkpoints, caches, and generated experiment outputs are not included in the repository.

## Run

### Online mapping

Run OH-SLAM on one Replica scene. `SCENE_NUM=0..7` corresponds to `room0`, `room1`, `room2`, and `office0..4`.

```bash
SCENE_NUM=0 SEED=0 PRIMARY_DEVICE=cuda:0 \
python scripts/hierslam.py configs/replica/ohslam_semantic_run.py
```

The default output is written to:

```text
experiments/OHSLAM_REPLICA/room0_ohslam_seed0/
```

### HCLR post-map calibration

After online mapping completes, run HCLR on the saved map:

```bash
python scripts/calibrate_hclr.py \
  --source experiments/OHSLAM_REPLICA/room0_ohslam_seed0 \
  --config-override experiments/OHSLAM_REPLICA/room0_ohslam_seed0/config.py \
  --output experiments/OHSLAM_HCLR/room0_ohslam_hclr_seed0 \
  --seed 0 \
  --device cuda:0
```

HCLR stores the calibrated decoder, its runtime gate, evaluation output, and a completion marker in the requested output directory.

### All Replica scenes

The restart-safe launcher runs mapping and HCLR sequentially for all eight scenes:

```bash
OHSLAM_PYTHON="$(which python)" \
PRIMARY_DEVICE=cuda:0 SEED=0 \
bash bash_scripts/run_replica.sh replica
```

Existing completed mapping and HCLR outputs are skipped. Use `tmux`, `screen`, or a cluster scheduler for long runs.

## Evaluation and visualization

A mapping run performs evaluation when it finishes. To evaluate an existing mapping output again:

```bash
SCENE_NUM=0 SEED=0 PRIMARY_DEVICE=cuda:0 \
python scripts/eval_novel_view.py configs/replica/ohslam_semantic_run.py
```

Export the reconstructed semantic Gaussian map:

```bash
SCENE_NUM=0 SEED=0 PRIMARY_DEVICE=cuda:0 \
python scripts/export_ply_semantic_tree.py configs/replica/ohslam_semantic_run.py
```

Visualize the reconstructed map and estimated camera poses:

```bash
SCENE_NUM=0 SEED=0 PRIMARY_DEVICE=cuda:0 \
python viz_scripts/online_recon_sem_replica.py \
  configs/replica/ohslam_semantic_run.py --flag_semantic
```

Omit `--flag_semantic` to visualize RGB appearance.

## Fixed semantic hierarchy

The PST definition is provided in `data/Replica_semantic/info_semantic_tree.json`, with Replica class names in `replica_classes.json`. The supplied hierarchy is fixed during mapping and calibration. No external LLM service, SegFormer checkpoint, or API key is required by this release.

## Repository structure

```text
configs/replica/ohslam_semantic_run.py   Replica configuration
data/Replica_semantic/                   Canonical PST definition
datasets/gradslam_datasets/              RGB-D and semantic loaders
hierslam-diff-gaussian-rasterization-w-depth/
                                         21-channel rasterizer
scripts/hierslam.py                      Online tracking and mapping
scripts/calibrate_hclr.py                HCLR post-map calibration
utils/tree_conflict_refinement.py        TCBR implementation
utils/hierarchical_leaf_recalibration.py HCLR objectives and runtime gate
viz_scripts/                             Map visualization
```

## Acknowledgements

OH-SLAM is developed from [Hier-SLAM](https://github.com/LeeBY68/Hier-SLAM) and uses components inspired by 3D Gaussian Splatting and SplaTAM. We thank the authors of these projects and the Replica dataset for making their work available.

## Citation

Use [CITATION.cff](CITATION.cff) for repository citation metadata. The manuscript authors are Yanli Liu, Yuyu Huang, Heng Zhang, and Zhimin Chen. A publication DOI will be added if assigned. Please also cite Hier-SLAM:

```bibtex
@inproceedings{li2025hier,
  title     = {Hier-SLAM: Scaling-up Semantics in SLAM with a Hierarchically Categorical Gaussian Splatting},
  author    = {Li, Boying and Cai, Zhixi and Li, Yuan-Fang and Reid, Ian and Rezatofighi, Hamid},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). Third-party components retain their original licenses.
