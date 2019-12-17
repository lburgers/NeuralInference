#!/bin/bash

#SBATCH --mail-user=lukas.burger@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=neural_inf_data
#SBATCH --partition=week
#SBATCH --time=2-

PROCESS=$1

module load Tools/miniconda
source activate py27
python ./build_training_set.py $PROCESS
