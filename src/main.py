# src/main.py
# do not remove this comment or the comment above

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
    new_data = data[idx][:num_data].unsqueeze(1).float() / 255.0
    new_targets = targets[idx][:num_data]
    return torch.utils.data.TensorDataset(new_data, new_targets)

def main():
    from config import get_config, set_seed
    args, config = get_config()
    set_seed()

    device = torch.device(config["general"]["device"])
    
    # Prepare Data
    num_train_data = 2048
    num_classes = 3
    train_raw = datasets.MNIST("data/", train=True, download=True, transform=transforms.ToTensor())
    test_raw  = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
    train_data = subsample(train_raw.data, train_raw.targets, num_train_data, num_classes)
    test_data  = subsample(test_raw.data, test_raw.targets, num_train_data, num_classes)
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Build model
    M = args.latent_dim
    prior = GaussianPrior(M)
    enc_net = new_encoder(M)
    dec_net = new_decoder(M)
    encoder = GaussianEncoder(enc_net)
    decoder = GaussianDecoder(dec_net)
    model = VAE(prior, decoder, encoder).to(device)

    if config["mode"] == "train":
        os.makedirs(args.experiment_folder, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
        torch.save(model.state_dict(), f"{args.experiment_folder}/model.pt")

    elif config["mode"] == "sample":
        model.load_state_dict(torch.load(f"{config['general']['experiment_folder']}/model.pt"))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)
            data, _ = next(iter(mnist_test_loader))
            data = data.to(device)
            recon_mean = model.decoder(model.encoder(data).mean).mean
            out = torch.cat([data.cpu(), recon_mean.cpu()], dim=0)
            save_image(out, "recon_means.png", nrow=data.size(0))
            print("Saved samples and reconstructions.")

    elif config["mode"] == "eval":
        model.load_state_dict(torch.load(f"{config['general']['experiment_folder']}/model.pt"))
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, _ in mnist_test_loader:
                x = x.to(device)
                elbos.append(model.elbo(x))
        mean_elbo = torch.stack(elbos).mean()
        print("Mean test ELBO =", mean_elbo.item())

    elif config["mode"] == "geodesics":
        model.load_state_dict(torch.load(f"{config['general']['experiment_folder']}/model.pt", weights_only=True))
        model.eval()

        # Directly choose endpoints from config.
        z_start = torch.tensor(config["geodesics"]["z_start"], device=device)
        z_end   = torch.tensor(config["geodesics"]["z_end"], device=device)
        
        print("Chosen endpoints:")
        print("z_start:", z_start)
        print("z_end:", z_end)
        print("Distance between endpoints:", (z_start - z_end).norm().item())

        # Compute the pull-back geodesic.
        z_opt, energy_hist, z_initial, final_energy = compute_geodesic_pullback_lbfgs(
            model,
            z_start,
            z_end,
            num_segments=args.num_segments,
            lr=args.lr,                 # from config or pass a small LR
            outer_steps=args.steps,     # e.g. 5 or 20
            optimizer_type=args.optimizer_type,
            line_search_fn=config["geodesics"]["line_search"],
            device=device,
            debug=False  # enable extra debug prints
        )
        print("Pull-back geodesic optimization done.")
        print(f"Final Energy (computed): {final_energy:.6f}")

        # Decode the geodesic for visualization.
        with torch.no_grad():
            imgs_curve = model.decoder(z_opt).mean  # shape: (S+1, 1, 28, 28)
        save_image(imgs_curve, "geodesic_path_pullback_lbfgs.png", nrow=imgs_curve.size(0))
        print("Saved geodesic images as 'geodesic_path_pullback_lbfgs.png'")

        # Gather latent representations from the test set (for plotting).
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
            z_initial, z_opt, 
            z_start, z_end,
            save_path="latent_space_geodesic_pullback_lbfgs.png"
        )
        print("Saved latent space geodesic plot: latent_space_geodesic_pullback_lbfgs.png")
        # Optionally, you can also plot the energy curve using plot_energy_curve.
        # plot_energy_curve(energy_hist, save_path="energy_curve_pullback_lbfgs.png")
        print("Saved energy curve plot: energy_curve_pullback_lbfgs.png")

if __name__ == "__main__":
    main()
