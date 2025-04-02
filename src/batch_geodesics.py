#!/usr/bin/env python

import os
import argparse
import torch
import numpy as np
from PIL import Image

# Import exactly the same classes/functions from your `ensemble_vae.py`
# (Make sure `ensemble_vae.py` is in the same directory, or on your Python path)
from src.ensemble_vae import (
    VAE,
    EnsembleVAE,
    GaussianPrior,
    GaussianEncoder,
    GaussianDecoder,
    new_encoder,
    new_decoder,
    compute_geodesic,        # The same geodesic function used by your "geodesics" mode
    compute_curve_length,    # The same function computing distance in observation space                       # If your ensemble_vae.py has "M" as a global variable for latent dim
)

M = 2

def load_trained_model(run_folder, num_decoders, device="cpu"):
    """
    Rebuild a model the same way your ensemble_vae.py does, then load `model.pt` weights.
    """
    model_path = os.path.join(run_folder, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model weights: {model_path}")

    # Reconstruct the same prior/encoder/decoder logic
    

    prior = GaussianPrior(M)

    # If num_decoders > 1, build an EnsembleVAE. Otherwise, a single VAE.
    if num_decoders > 1:
        decoders_list = [GaussianDecoder(new_decoder(M)) for _ in range(num_decoders)]
        encoder_module = GaussianEncoder(new_encoder(M))
        model = EnsembleVAE(prior, decoders_list, encoder_module).to(device)
    else:
        decoder_module = GaussianDecoder(new_decoder(M))
        encoder_module = GaussianEncoder(new_encoder(M))
        model = VAE(prior, decoder_module, encoder_module).to(device)

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def encode_image_pair(model, img1_path, img2_path, device="cpu"):
    """
    Load two images (28x28 MNIST style) from disk, 
    pass them through model.encoder to get z1 and z2.
    """
    def pil_to_tensor(path):
        with Image.open(path).convert("L") as pil_img:
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # shape [1,1,H,W]
            return tensor

    x1 = pil_to_tensor(img1_path).to(device)
    x2 = pil_to_tensor(img2_path).to(device)

    with torch.no_grad():
        q1 = model.encoder(x1)
        q2 = model.encoder(x2)
        z1 = q1.mean.squeeze(0)  # shape [latent_dim]
        z2 = q2.mean.squeeze(0)

    return z1, z2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--training-root", default="training_runs",
                        help="Folder with [1_decoder, 2_decoder, 3_decoder]/run_[i]/model.pt")
    parser.add_argument("--num-decoders-list", nargs="+", type=int, default=[1,2,3],
                        help="Which ensemble sizes to evaluate.")
    parser.add_argument("--num-reruns", type=int, default=10,
                        help="Number of run folders (e.g. run_0..run_9).")
    parser.add_argument("--paired-images-folder", default="data/paired_images",
                        help="Folder with pair_0, pair_1, etc. each containing img1.png & img2.png.")
    parser.add_argument("--image-pairs-file", default="data/fixed_image_pairs.pt",
                        help="Torch file with Nx2 integer pairs (the indices).")
    parser.add_argument("--output-file", default="geodesics_distances.pt",
                        help="Where to store the final distance dictionary.")
    # Optional: how many segments in geodesic
    parser.add_argument("--num-segments", type=int, default=20,
                        help="How many segments to use in compute_geodesic.")
    args = parser.parse_args()

    device = args.device
    pairs_tensor = torch.load(args.image_pairs_file)  # shape [N,2]
    pairs_list = pairs_tensor.tolist()

    # We'll store a nested dictionary:
    # results[num_decoders][run_id] = [ { "latent_dist":..., "obs_geodesic":...}, ... per pair ]
    results = {}

    for nd in args.num_decoders_list:
        results[nd] = {}
        dec_folder = f"{nd}_decoder"
        for run_id in range(args.num_reruns):
            run_folder = os.path.join(args.training_root, dec_folder, f"run_{run_id}")
            print(f"\n--- Loading model for decoders={nd}, run={run_id} ---")
            model = load_trained_model(run_folder, nd, device=device)

            pairwise_distances = []
            for i, (idxA, idxB) in enumerate(pairs_list):
                # Each pair_i folder has img1.png and img2.png
                pair_dir = os.path.join(args.paired_images_folder, f"pair_{i}")
                img1 = os.path.join(pair_dir, "img1.png")
                img2 = os.path.join(pair_dir, "img2.png")

                # Encode => z1, z2
                z1, z2 = encode_image_pair(model, img1, img2, device=device)

                # Latent Euclidean distance
                latent_dist = torch.norm(z2 - z1, p=2).item()

                # For ensembles, pass ensemble=True
                ensemble_mode = (nd > 1)

                # Geodesic in observation space
                init_curve, final_curve = compute_geodesic(
                    model, z1, z2,
                    num_segments=args.num_segments,
                    ensemble=ensemble_mode
                )
                obs_length = compute_curve_length(model, final_curve, ensemble=ensemble_mode)

                pairwise_distances.append({
                    "pair_index": i,
                    "latent_dist": latent_dist,
                    "obs_geodesic": obs_length
                })

            results[nd][run_id] = pairwise_distances

    # Save them all to a single file
    torch.save(results, args.output_file)
    print(f"All done. Saved distances to {args.output_file}")

if __name__ == "__main__":
    main()
