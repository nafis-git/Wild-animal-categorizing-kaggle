#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1,nvme:100
#SBATCH --time=00:15:00
#SBATCH --mem=64G
#SBATCH --account=project_2000859

module purge
module load pytorch/1.3.0
module list

export DATADIR=/scratch/project_2000859/extracted
export TMPDIR=$LOCAL_SCRATCH

set -xv
python3.7 $*
