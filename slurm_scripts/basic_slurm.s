#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=${USER}@nyu.edu
#SBATCH --output=test_out_%j.out

SCRIPT_DIR=${SCRATCH}/cv_project_22/
PLOTS_DIR=${SCRIPT_DIR}plots/
MODELS_DIR=${SCRIPT_DIR}models/

TRAIN_IMGS=${SCRIPT_DIR}data/fundus_ds/Training_Set/Training_Set/Training/resized_complete/upsampled_64
TRAIN_CSV=${SCRIPT_DIR}plots/.....
VAL_IMGS=${SCRIPT_DIR}plots/.....
VAL_CSV=${SCRIPT_DIR}plots/.....

singularity exec --nv \
--bind /scratch \
--overlay ${SCRATCH}/cv_gpu/overlay-10GB-400K.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh

python3 ${SCRIPT_DIR}main.py \
--train-ds-sz 50 --val-ds-sz 5 \
--epochs 1 \
--batch-size 20 \
--model-name resent18 \
--save-model \
--save-model-dir $MODELS_DIR \
--save-plots \
--save-plot-dir $PLOTS_DIR \
--root-dir ${SCRIPT_DIR} \
--train-csv ${TRAIN_CSV} \
--train-imgs-dir ${TRAIN_IMGS} \
--val-csv ${VAL_CSV} \
--val-imgs-dir ${VAL_IMGS}
"

















