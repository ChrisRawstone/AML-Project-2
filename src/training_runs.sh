#!/bin/bash
#BSUB -J geodesics
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R "rusage[mem=5GB]"
# #BSUB -u lassesofus@gmail.com
#BSUB -o %J.out
#BSUB -e %J.err

module load cuda/11.8

# source /zhome/e3/3/139772/Desktop/AML/aml_new/bin/activate
source ../actvenv.sh

NUM_RUNS=1
BASE_DIR="experiment"
mkdir -p "$BASE_DIR"

for i in $(seq 1 $NUM_RUNS); do
    RUN_DIR="$BASE_DIR/run_$(printf "%02d" $i)"
    mkdir -p "$RUN_DIR"

    echo "Run $i: Training model..."
    python ensemble_vae.py --mode train --device cuda --experiment-folder "$RUN_DIR"
    
    echo "Run $i: Computing geodesics..."
    python ensemble_vae.py --mode geodesics --device cuda --experiment-folder "$RUN_DIR"
    
    # Move output files to the run directory with run-specific names.
    for file in latent_geodesics.png linear_interpolation_reconstructions.png optimized_geodesic_reconstructions.png; do
        if [ -f "$file" ]; then
            mv "$file" "$RUN_DIR/${file%.*}_$(printf "%02d" $i).png"
        else
            echo "Warning: $file not found for run $i"
        fi
    done
done

echo "All runs completed. Check the '$BASE_DIR' directory for the plots."
