#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate lhm
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/GaussianAvatar

