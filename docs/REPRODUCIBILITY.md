# Reproducibility and evaluation

## Method and implementation

PST denotes the reconstructed path-encoded semantic tree: 102 output labels use a 21-channel path representation with widths 10, 4, and 7. TCBR refines spatial support when depth-consistent complete-path conflicts persist across views. HCLR subsequently optimizes the leaf decoder while map geometry, appearance, Gaussian semantic features, and camera poses remain fixed.

The fixed hierarchy and Replica class metadata are provided in `data/Replica_semantic/`. Standard experiments use this hierarchy directly and require no LLM service or API credentials.

## Run and calibration

Follow the main README for environment installation, the 21-channel rasterizer, dataset preparation, mapping, and HCLR. HCLR's default calibration runs for 128 steps, checks validation every 32 steps, and uses 64 selected training frames and 48 validation frames. Official evaluation frames are excluded from both calibration and validation. Consult `scripts/calibrate_hclr.py` for the exact split and activation gate. No new training is performed by the numerical summaries provided here.

## Archived paper results

`results/paper_seed0/semantic_per_scene.csv` contains all eight Replica scenes for Hier-SLAM, GS3LAM, TGS-SLAM, and OH-SLAM, under the common leaf-label protocol. Values are percentages and preserve the available source precision. `semantic_mean.csv` takes the equal-weight mean of the eight scene values. These are archived full-sequence semantic summaries, not native-label scores or a new benchmark run. SGS-SLAM semantic scores are excluded because label equivalence was not verified; SplaTAM supplies no corresponding semantic output in this comparison.

| Method | Eight-scene mIoU (%) | Eight-scene mBIoU (%) |
|---|---:|---:|
| Hier-SLAM | 73.920 | 72.931 |
| GS3LAM | 84.397 | 83.253 |
| TGS-SLAM | 68.007 | 67.403 |
| OH-SLAM | 82.870 | 81.832 |

Table 3 of the manuscript selects room1, room2, office3, and office4. OH-SLAM ranks first in both metrics for room2 and office3. For room1 and office4, it ranks second in mIoU and first in mBIoU. This scene selection does not imply the highest eight-scene mean; GS3LAM has the higher aggregate semantic scores.

## Metric conventions and limits

The manuscript distinguishes camera-center trajectory RMSE after rigid alignment, valid-depth L1, framewise semantic scores, and full-sequence semantic scores. Some inherited console/file labels use historical conventions: an output named ATE RMSE or Depth RMSE must not be treated as the manuscript metric without checking its definition. The archived semantic CSVs do not certify a fresh end-to-end run of the current release. No unified-hardware runtime or peak-memory claim is made. The 21 semantic channels describe attribute storage, not measured total system memory.

The method uses semantic supervision on Replica. The reported experiments do not establish open-world, label-free, or cross-dataset generalization. Full raw logs, rendered dataset inputs, and checkpoints are not bundled with this repository.
