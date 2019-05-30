#!/bin/bash

declare -a BLOCKS=("1" "2" "3" "4" "5")

for NUM_BLOCKS in "${BLOCKS[@]}"; do
	sbatch slurm_run.sh $NUM_BLOCKS
done
