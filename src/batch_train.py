import subprocess
import os

# Config
training_root = "training_runs"
num_decoders_list = [1, 2, 3]
num_reruns = 10

os.makedirs(training_root, exist_ok=True)

global_seed = 0  # Start from seed 0

for num_decoders in num_decoders_list:
    setting_folder = os.path.join(training_root, f"{num_decoders}_decoder")
    os.makedirs(setting_folder, exist_ok=True)

    for run_id in range(num_reruns):
        run_folder = os.path.join(setting_folder, f"run_{run_id}")
        os.makedirs(run_folder, exist_ok=True)

        print(f"\n--- Training {num_decoders}-decoder model, run {run_id}, seed={global_seed} ---\n")

        subprocess.run([
            "python", "main.py",
            "--mode", "train",
            "--experiment-folder", run_folder,
            "--num-decoders", str(num_decoders),
            "--device", "cuda",  # or "cpu"
            "--epochs-per-decoder", "200",
            "--seed", str(global_seed),
        ])

        global_seed += 1