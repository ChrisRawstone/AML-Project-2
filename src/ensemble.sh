#!/bin/bash
#BSUB -J geodesics_run6
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u lassesofus@gmail.com
#BSUB -o %J.out
#BSUB -e %J.err

module load cuda/11.8

source /zhome/e3/3/139772/Desktop/AML/aml_new/bin/activate

python ensemble_vae.py --mode train --device cuda --experiment-folder ../experiment/run6 --num-decoders 1 --num-t 50 --num-curves 1
python ensemble_vae.py --mode geodesics --device cuda --experiment-folder ../experiment/run6 --num-decoders 1 --num-t 50 --num-curves 1