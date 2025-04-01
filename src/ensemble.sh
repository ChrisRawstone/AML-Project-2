#!/bin/bash
#BSUB -J ensemble_3_v6
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

#python ensemble_vae.py --mode train --device cuda --experiment-folder ../experiment/ensemble_3_v6 --num-decoders 3 --num-t 50 --num-curves 25
python ensemble_vae.py --mode geodesics --device cuda --experiment-folder ../experiment/ensemble_3_v6 --num-decoders 3 --num-t 50 --num-curves 1