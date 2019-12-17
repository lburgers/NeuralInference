#!/bin/bash

#SBATCH --mail-user=lukas.burger@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --output=training_job.txt
#SBATCH --job-name=neural_inf_training
#SBATCH --partition=day
#SBATCH --time=1-

module load Tools/miniconda
source activate py27
python ./train_network.py
