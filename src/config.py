CONFIG = {
    "general": {
        "device": "cuda",
        "experiment_folder": "experiment",
        "samples": "samples.png",
        "latent_dim": 2,
        "batch_size": 32,
        "num_classes": 3,
        "num_train_data": 2048,
    },
    "train": {
        "epochs_per_decoder": 50,
        "learning_rate": 1e-3
    },
    "sample": {
        "num_samples": 64
    },
    "geodesics": {
        "num_segments": 10,
        "steps": 20,
        "optimizer_type": "lbfgs",  # or "adam"
        "lr": 1e-3,
        "z_start": [2.0, -1.0],
        "z_end": [-1.5, 2.0]
    }
}