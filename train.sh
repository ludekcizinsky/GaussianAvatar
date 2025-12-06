#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate lhm
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/GaussianAvatar


python train.py -s $gs_data_path/m4c_processed -m output/m4c_processed --train_stage 1