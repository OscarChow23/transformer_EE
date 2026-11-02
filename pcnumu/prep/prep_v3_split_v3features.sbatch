#!/bin/bash
#SBATCH --job-name=prep_v3_split
#SBATCH --partition=normalrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data/nova/chow/transformerEE/datasets/mprod6_1/training_v3/derived/prep_v3_split_v3features-%j.out

# CPU-only. Builds the r05 training files: v4-feature columns + IDs/sample/targets,
# hash-based per-event split (test 20% / valid 4% / train 76%, same for every sample),
# events with non-finite feature values dropped, plus a 2% smoke file and a manifest.
set -euo pipefail
BASE=/data/nova/chow/transformerEE
OUT=$BASE/datasets/mprod6_1/training_v3/derived
source /common/users/chow/miniforge3/etc/profile.d/conda.sh
conda activate transformer_ee
for HORN in fhc rhc; do
  python -I $BASE/prep/make_split_csv.py \
    --input $BASE/datasets/mprod6_1/training_v3/dataset_transformer_ee_fd_${HORN}_nonswap_training_v3.csv.xz \
    --config $BASE/configs/pcnumu/$HORN/pcnumu_${HORN}_r05_v3data-allsamples_v3features.json \
    --out-dir $OUT \
    --name dataset_transformer_ee_fd_${HORN}_nonswap_training_v3_split_v3features \
    --seed 20261029
done
