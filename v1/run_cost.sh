#!/bin/bash

#SBATCH --mail-user=lukas.burger@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=neural_inf_cost
#SBATCH --partition=day
#SBATCH --time=1-

module load Tools/miniconda
source activate py27
python ./build_cost_matrix.py
