#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --mail-type=END
#SBATCH --mail-user=mya6510@nyu.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=vgg19s64e50
#SBATCH --output=vgg19_g1_e1_s64_samp_f_%j.out
#SBATCH --wrap "sleep infinity"

SCRIPT_DIR=${SCRATCH}/cv_proj/
PLOTS_DIR=${SCRIPT_DIR}plots/
MODELS_DIR=${SCRIPT_DIR}models/

singularity exec --nv \
--bind /scratch \
--overlay ${SCRATCH}/cv_gpu/overlay-50G-10M.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh

python3 ${SCRIPT_DIR}main.py \
--train-ds-sz 9226 --val-ds-sz 640 \
--epochs 1 \
--batch-size 64 \
--lr 0.00001 \
--momentum 0.9 \
--image-size 64 \
--model-name vgg19 \
--save-model \
--save-model-dir $MODELS_DIR \
--save-plots \
--save-plot-dir $PLOTS_DIR \
--root-dir ${SCRIPT_DIR} \
--num-worker 2
"
