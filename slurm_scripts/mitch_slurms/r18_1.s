#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --job-name=1
#SBATCH --mail-type=END
#SBATCH --mail-user=${USER}@nyu.edu
#SBATCH --output=1_%j.out
#SBATCH --wrap "sleep infinity"

SCRIPT_DIR=${SCRATCH}/cv_project_22/
PLOTS_DIR=${SCRIPT_DIR}plots/
MODELS_DIR=${SCRIPT_DIR}models/

TRAIN_IMGS=${SCRIPT_DIR}data/fundus_ds/Training_Set/Training_Set/Training/resized_complete/upsampled_64
TRAIN_CSV=${SCRIPT_DIR}data/fundus_ds/Training_Set/Training_Set/RFMiD_Training_Labels_w_upsampling.csv
VAL_IMGS=${SCRIPT_DIR}data/fundus_ds/Evaluation_Set/Evaluation_Set/Validation/resized_complete/64
VAL_CSV=${SCRIPT_DIR}data/fundus_ds/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv

singularity exec --nv \
--bind /scratch \
--overlay ${SCRATCH}/cv_gpu/overlay-10GB-400K.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh

python3 ${SCRIPT_DIR}main.py \
--train-ds-sz 9226 --val-ds-sz 640 \
--epochs 1 \
--batch-size 64 \
--lr 0.0001 \
--decay 1.0 \
--momentum 0.9 \
--image-size 64 \
--model-name resent18 \
--save-model \
--save-model-dir $MODELS_DIR \
--save-plots \
--save-plot-dir $PLOTS_DIR \
--num-worker 2 \
--train-csv ${TRAIN_CSV} \
--train-imgs-dir ${TRAIN_IMGS} \
--val-csv ${VAL_CSV} \
--val-imgs-dir ${VAL_IMGS}
"
