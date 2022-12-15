#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40GB
#SBATCH --mail-type=END
#SBATCH --output=8_128_%j.out
#SBATCH --job-name=8_128

SCRIPT_DIR=${SCRATCH}/cv_project_22/
PLOTS_DIR=${SCRIPT_DIR}plots/8_128/
MODELS_DIR=${SCRIPT_DIR}models/

TRAIN_IMGS=${SCRIPT_DIR}data/fundus_ds/Training_Set/Training_Set/Training/resized_complete/128/
TRAIN_CSV=${SCRIPT_DIR}data/fundus_ds/Training_Set/Training_Set/RFMiD_Training_Labels_w_upsampling_newen.csv
VAL_IMGS=${SCRIPT_DIR}data/fundus_ds/Evaluation_Set/Evaluation_Set/Validation/resized_complete/128/
VAL_CSV=${SCRIPT_DIR}data/fundus_ds/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv

singularity exec --nv \
--bind /scratch \
--overlay ${SCRATCH}/cv_gpu4/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh

python3 ${SCRIPT_DIR}main.py \
--train-ds-sz 9226 --val-ds-sz 640 \
--model-name resent18 \
--lr 0.0001 \
--epochs 100 \
--sgd \
--decay 0.9 \
--momentum 0.9 \
--save-model \
--save-model-dir $MODELS_DIR \
--save-plots \
--save-plot-dir $PLOTS_DIR \
--num-worker 1 \
--train-csv ${TRAIN_CSV} \
--train-imgs-dir ${TRAIN_IMGS} \
--val-csv ${VAL_CSV} \
--val-imgs-dir ${VAL_IMGS} \
--image-size 64 \
--batch-size 64
"
