#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8GB
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=EMAIL_GOES_HERE@nyu.edu
#SBATCH --output=test_out_%j.out

SCRIPT_DIR=${SCRATCH}/cv_proj/

singularity exec --nv \
--bind /scratch \
--overlay /scratch/NETID_GOES_HERE/cv_gpu/overlay-10GB-400K.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh
python3 ${SCRIPT_DIR}/test.py

"
