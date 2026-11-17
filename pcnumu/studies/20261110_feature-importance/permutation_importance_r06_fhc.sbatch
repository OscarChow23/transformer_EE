#!/bin/bash
#SBATCH --job-name=permimp_r06_fhc
#SBATCH --partition=normalrun
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/nova/chow/pcnumu-runs/analysis/feature-importance/r06/fhc/slurm-%j.out

# Inference only (no training, so no pre-training gate): permutation importance of the frozen
# r06 FHC model. Oscar submits: sbatch pcnumu/studies/20261110_feature-importance/permutation_importance_r06_fhc.sbatch
set -euo pipefail
CODE_DIR=/data/nova/chow/transformerEE-pcnumu-loop
RUNS=/data/nova/chow/pcnumu-runs
source /common/users/chow/miniforge3/etc/profile.d/conda.sh
conda activate transformer_ee
export PATH=/data/chow/tools/git-env/bin:$PATH POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4
nvidia-smi
python -I "$CODE_DIR/pcnumu/studies/20261110_feature-importance/permutation_importance.py" \
  --model-dir "$RUNS/r06_flat-alpha001/fhc/model_f7bddcadada57cb477800c2be83ba5d4" \
  --out-dir "$RUNS/analysis/feature-importance/r06/fhc" --seeds 5
