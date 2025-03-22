import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse

from model import GaussianPrior, GaussianEncoder, GaussianDecoder, VAE, new_encoder, new_decoder
from train import train
from geodesics import compute_geodesic_pullback_lbfgs
from plot_utils import (
    plot_energy_curve,
    plot_geodesic_comparison,
    plot_latent_space
)

def subsample(data, targets, num_data, num_classes):
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).float()/255.0
    new_targets = targets[idx][:num_data]
    return torch.utils.data.TensorDataset(new_data, new_targets)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="geodesics",
                        choices=["train", "sample", "eval", "geodesics"],
                        help="Action to perform.")
    parser.add_argument("--experiment-folder", type=str, default="experiment",
                        help="Folder to load/save the model.")
    parser.add_argument("--samples", type=str, default="samples.png",
                        help="File to save samples.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"], help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs-per-decoder", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--num-segments", type=int, default=5,
                        help="Number of segments in geodesic.")
    parser.add_argument("--steps", type=int, default=10,
                        help="Outer LBFGS iterations.")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Prepare Data
    num_train_data = 2048
    num_classes = 3
    train_raw = datasets.MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
    test_raw  = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
    train_data = subsample(train_raw.data, train_raw.targets, num_train_data, num_classes)
    test_data  = subsample(test_raw.data,  test_raw.targets,  num_train_data, num_classes)
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Build model
    M = args.latent_dim
    prior = GaussianPrior(M)
    enc_net = new_encoder(M)
    dec_net = new_decoder(M)
    encoder = GaussianEncoder(enc_net)
    decoder = GaussianDecoder(dec_net)
    model   = VAE(prior, decoder, encoder).to(device)

    if args.mode == "train":
        os.makedirs(args.experiment_folder, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
        torch.save(model.state_dict(), f"{args.experiment_folder}/model.pt")

    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu()
            save_image(samples.view(64,1,28,28), args.samples)
            data, _ = next(iter(mnist_test_loader))
            data = data.to(device)
            recon_mean = model.decoder(model.encoder(data).mean).mean
            out = torch.cat([data.cpu(), recon_mean.cpu()], dim=0)
            save_image(out, "recon_means.png", nrow=data.size(0))
            print("Saved samples and reconstructions.")

    elif args.mode == "eval":
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, _ in mnist_test_loader:
                x = x.to(device)
                elbos.append(model.elbo(x))
        mean_elbo = torch.stack(elbos).mean()
        print("Mean test ELBO =", mean_elbo.item())

    elif args.mode == "geodesics":
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        model.eval()

        # Get a batch from the test set.
        data_batch, labels_batch = next(iter(mnist_test_loader))
        data_batch = data_batch.to(device)
        z_enc = model.encoder(data_batch).mean  # shape (batch_size, M)

        # Choose endpoints; here, for example, pick points near (0,0) and (-2,6).
        target_start = torch.tensor([0.0, 0.0], device=device)
        target_end   = torch.tensor([-2.0, 6.0], device=device)
        dstart = torch.norm(z_enc - target_start, dim=1)
        dend   = torch.norm(z_enc - target_end,   dim=1)
        z_start = z_enc[dstart.argmin()]
        z_end   = z_enc[dend.argmin()]

        print("z_start:", z_start)
        print("z_end:", z_end)
        print("Distance between endpoints:", (z_start - z_end).norm().item())

        # Compute geodesic with LBFGS (pull-back metric) optimizing only interior points.
        z_opt, energy_hist, z_init = compute_geodesic_pullback_lbfgs(
            model,
            z_start, 
            z_end,
            num_segments=args.num_segments,
            lr=1e-2,
            outer_steps=args.steps
        )
        print("LBFGS pull-back geodesic optimization done.")
        print("Final energy:", energy_hist[-1])

        # Decode geodesic for visualization.
        with torch.no_grad():
            imgs_curve = model.decoder(z_opt).mean  # shape: (S+1, 1, 28, 28)
        save_image(imgs_curve, "geodesic_path_pullback_lbfgs.png", nrow=imgs_curve.size(0))
        print("Saved geodesic images as 'geodesic_path_pullback_lbfgs.png'")

        # Gather latent representations from the test set.
        all_latents = []
        all_labels  = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                all_latents.append(model.encoder(x).mean)
                all_labels.append(y)
        all_latents = torch.cat(all_latents, dim=0)
        all_labels  = torch.cat(all_labels, dim=0)

        # Plot comparison: initial (linear) vs. optimized geodesic.
        plot_geodesic_comparison(
            all_latents, all_labels,
            z_init, z_opt, 
            z_start, z_end,
            save_path="latent_space_geodesic_pullback_lbfgs.png"
        )
        print("Saved latent space geodesic plot: latent_space_geodesic_pullback_lbfgs.png")

        # Plot energy curve.
        plot_energy_curve(energy_hist, save_path="energy_curve_pullback_lbfgs.png")
        print("Saved energy curve plot: energy_curve_pullback_lbfgs.png")

if __name__ == "__main__":
    main()