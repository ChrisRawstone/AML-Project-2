import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def compute_cov(distances):
    """
    Compute the coefficient of variation (CoV) over M=10 reruns for a single pair.

    distances: List of M floats for a given (i,j) pair
    """
    distances = np.array(distances)
    mean = np.mean(distances)
    std = np.std(distances)
    return std / mean if mean > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="geodesics_distances.pt",
                        help="Path to the saved distance dictionary.")
    parser.add_argument("--savefig", default="cov_plot.png",
                        help="Where to save the resulting plot.")
    args = parser.parse_args()

    # Load saved results
    results = torch.load(args.input)

    # Final results
    decoder_sizes = sorted(results.keys())
    cov_latent = []
    cov_geodesic = []

    for num_decoders in decoder_sizes:
        print(f"Processing {num_decoders} decoder(s)...")

        # Transpose the results so we group by pair_index
        # Each entry: list of 10 distance dicts per decoder setting
        runs = results[num_decoders]  # dict: run_id -> list of distances

        num_pairs = len(next(iter(runs.values())))  # e.g., 10 pairs
        latent_covs = []
        geodesic_covs = []

        for pair_idx in range(num_pairs):
            latent_dists = [runs[run_id][pair_idx]["latent_dist"] for run_id in runs]
            geodesic_dists = [runs[run_id][pair_idx]["obs_geodesic"] for run_id in runs]

            cov_lat = compute_cov(latent_dists)
            cov_geo = compute_cov(geodesic_dists)

            latent_covs.append(cov_lat)
            geodesic_covs.append(cov_geo)

        # Average across all pairs
        avg_cov_latent = np.mean(latent_covs)
        avg_cov_geodesic = np.mean(geodesic_covs)

        cov_latent.append(avg_cov_latent)
        cov_geodesic.append(avg_cov_geodesic)

        print(f"Avg CoV (latent):   {avg_cov_latent:.4f}")
        print(f"Avg CoV (geodesic): {avg_cov_geodesic:.4f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(decoder_sizes, cov_latent, marker='o', linestyle='-', label="Latent Euclidean CoV")
    plt.plot(decoder_sizes, cov_geodesic, marker='o', linestyle='-', label="Geodesic CoV")
    plt.xlabel("Number of Decoders in Ensemble")
    plt.ylabel("Average Coefficient of Variation (CoV)")
    plt.title("Reliability of Distances vs Number of Decoders")
    plt.xticks(decoder_sizes)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.savefig, dpi=300)
    plt.show()
    print(f"\nPlot saved to {args.savefig}")


if __name__ == "__main__":
    main()
