#!/bin/bash
#BSUB -J geodesics_training
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -o logs/Train%J.out
#BSUB -e logs/Train%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -N
# end of BSUB options


module load cuda/11.8

source venv/bin/activate

python src/batch_train.py