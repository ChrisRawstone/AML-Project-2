# src/config.py
# do not remove this comment or the comment above

import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[DEBUG] Using random seed: {seed}")

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
        "num_segments": 50,
        "steps": 20,
        # Lower default LR for safer, more stable geodesic optimization
        "lr": 1e-2,
        "optimizer_type": "adam",  # or "adam"
        # If using LBFGS, the line_search_fn can help with stability
        "line_search": "strong_wolfe",
        "z_start": [1.8, -2.0],
        "z_end": [-1.5, 2.0]
    }
}
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="geodesics",
                        choices=["train", "sample", "eval", "geodesics"],
                        help="Action to perform.")
    parser.add_argument("--experiment-folder", type=str, default=CONFIG["general"]["experiment_folder"],
                        help="Folder to load/save the model.")
    parser.add_argument("--samples", type=str, default=CONFIG["general"]["samples"],
                        help="File to save samples.")
    parser.add_argument("--device", type=str, default=CONFIG["general"]["device"],
                        choices=["cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=CONFIG["general"]["batch_size"],
                        help="Batch size for training.")
    parser.add_argument("--epochs-per-decoder", type=int, default=CONFIG["train"]["epochs_per_decoder"],
                        help="Number of training epochs per decoder.")
    parser.add_argument("--latent-dim", type=int, default=CONFIG["general"]["latent_dim"],
                        help="Dimension of the latent space.")
    parser.add_argument("--num-segments", type=int, default=CONFIG["geodesics"]["num_segments"],
                        help="Number of segments in the geodesic.")
    parser.add_argument("--steps", type=int, default=CONFIG["geodesics"]["steps"],
                        help="Number of outer optimization steps for geodesics.")
    parser.add_argument("--lr", type=float, default=CONFIG["geodesics"]["lr"],
                        help="Learning rate for geodesic optimization.")
    parser.add_argument("--optimizer_type", type=str, default=CONFIG["geodesics"]["optimizer_type"],
                        choices=["lbfgs", "adam"], help="Optimizer type for geodesic computation.")
    parser.add_argument("--line_search", type=str, default=CONFIG["geodesics"]["line_search"],
                        choices=["none", "strong_wolfe"], help="LBFGS line search fn for stability.")
    
    args = parser.parse_args()

    CONFIG["general"]["device"] = args.device
    CONFIG["general"]["experiment_folder"] = args.experiment_folder
    CONFIG["general"]["samples"] = args.samples
    CONFIG["general"]["batch_size"] = args.batch_size
    CONFIG["general"]["latent_dim"] = args.latent_dim
    CONFIG["train"]["epochs_per_decoder"] = args.epochs_per_decoder
    CONFIG["geodesics"]["num_segments"] = args.num_segments
    CONFIG["geodesics"]["steps"] = args.steps
    CONFIG["geodesics"]["lr"] = args.lr
    CONFIG["geodesics"]["optimizer_type"] = args.optimizer_type
    CONFIG["geodesics"]["line_search"] = args.line_search
    CONFIG["mode"] = args.mode

    return args, CONFIG
