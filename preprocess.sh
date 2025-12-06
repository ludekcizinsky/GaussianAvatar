#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate lhm
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/GaussianAvatar

python scripts/gen_pose_map_our_smpl.py --root-save-dir /scratch/izar/cizinsky/thesis/preprocessing/ps_male3_casual/lhm --save-png