#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${OHSLAM_PYTHON:-python}"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-0}"
PRIMARY_DEVICE="${PRIMARY_DEVICE:-cuda:0}"
CONFIG="configs/replica/ohslam_semantic_run.py"
MAPPING_ROOT="${ROOT_DIR}/experiments/OHSLAM_${RUN_ID}"
HCLR_ROOT="${ROOT_DIR}/experiments/OHSLAM_HCLR_${RUN_ID}"
SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)

mkdir -p "${MAPPING_ROOT}/logs" "${HCLR_ROOT}"

for index in "${!SCENES[@]}"; do
    scene="${SCENES[$index]}"
    run_name="${scene}_ohslam_seed${SEED}"
    scene_dir="${MAPPING_ROOT}/${run_name}"
    hclr_dir="${HCLR_ROOT}/${scene}_ohslam_hclr_seed${SEED}"

    if [[ ! -f "${scene_dir}/params.npz" ]]; then
        SCENE_NUM="${index}" SEED="${SEED}" PRIMARY_DEVICE="${PRIMARY_DEVICE}" \
            "${PYTHON_BIN}" scripts/hierslam.py "${CONFIG}" --set \
            workdir="${MAPPING_ROOT}" \
            run_name="${run_name}" \
            data.results_dir="${scene_dir}" \
            2>&1 | tee -a "${MAPPING_ROOT}/logs/${scene}.log"
    else
        echo "[OH-SLAM] ${scene}: mapping output already exists"
    fi

    if [[ ! -f "${hclr_dir}/RUN_COMPLETE" ]]; then
        mkdir -p "${hclr_dir}"
        "${PYTHON_BIN}" scripts/calibrate_hclr.py \
            --source "${scene_dir}" \
            --config-override "${scene_dir}/config.py" \
            --output "${hclr_dir}" \
            --steps 128 \
            --candidate-count 160 \
            --selected-count 64 \
            --validation-count 48 \
            --temporal-bins 8 \
            --validate-every 32 \
            --lr 0.0001 \
            --minimum-row-radius 0.05 \
            --maximum-row-radius 0.15 \
            --protect-confidence 0.90 \
            --activation-minimum-pixels 9 \
            --activation-minimum-area-ratio 0.00001 \
            --seed "${SEED}" \
            --device "${PRIMARY_DEVICE}" \
            2>&1 | tee -a "${hclr_dir}/run.log"
    else
        echo "[OH-SLAM] ${scene}: HCLR output already exists"
    fi
done

touch "${HCLR_ROOT}/ALL_COMPLETE"
echo "[OH-SLAM] Completed outputs: ${HCLR_ROOT}"
