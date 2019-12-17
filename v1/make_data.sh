#!/bin/bash

COUNTER=50
until [  $COUNTER -lt 0 ]; do
    sbatch run.sh $COUNTER
    let COUNTER-=1
done