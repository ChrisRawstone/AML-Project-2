"""
Script for computing coefficient of variance of multiple ensemble VAEs over retrainings.
"""

import os
import torch
from tqdm import tqdm
from torchvision import datasets, transforms

from ensemble_vae import (
    EnsembleVAE, 
    GaussianPrior, 
    GaussianDecoder, 
    GaussianEncoder,
    new_decoder,
    new_encoder,
    compute_geodesic,
    compute_curve_length,
    subsample,
    )
from config import device, args #! imports all args, can be overriden in terminal
# # define a global variable for device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Matrix for storing distances between pairs i,j
as a result of which model. 
There will be 3 of these, one for each decoder.

tensor(0, M, N) : 0th decoder, M models, N pairs
pair    1,2 2,3 5,6 1,9 ... N
mod
1       5.6 4.5 .   .   .   .
2       3.4 3.4  
3       .       .
4       .           .
...     .               .    
M       .                   .    
"""
def CoefficientOfVariance(distance_mat): #TODO: make input be for each decoder?
    M, N = distance_mat.shape #M=10, N=>10
    CoV = torch.zeros(N)
    for n in range(N):
        CoV[n] = torch.std(distance_mat[:, n]) / torch.mean(distance_mat[:, n])
    return CoV


# Euclidean distance doenst depend on decoders so its always the same
def euclidean_curve_length(z0, z1):
    return torch.norm(z0 - z1, 2)

def create_model(args):
    decoders = [GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]
    model = EnsembleVAE(
        GaussianPrior(M),
        decoders,
        GaussianEncoder(new_encoder()),
    ).to(device)
    # see lines 880 in ensemble_vae.py
    return model

def encode_test_data(model, mnist_test_loader):
    model.eval()
    # Encode test data to get latent representations
    all_latents = []
    all_labels = []
    for x, y in mnist_test_loader:
        x = x.to(device)
        with torch.no_grad():
            q = model.encoder(x)
            # Use mean of q as latent representation
            latent = q.mean
        all_latents.append(latent)
        all_labels.append(y)
    all_labels = torch.cat(all_labels, dim=0).cpu()
    all_latents = torch.cat(all_latents, dim=0).cpu()

    return all_latents, all_labels



if __name__ == "__main__":

    num_train_data = 2048
    num_classes = 3
    test_tensors = datasets.MNIST(
            "data/",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    test_data = subsample(
            test_tensors.data, test_tensors.targets, num_train_data, num_classes
        )
    mnist_test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False
        )


    ensemble = True
    num_decoders = 3
    M = 10
    N_pairs = 10
    # init matrix for storing pairwise distances (for both geodesic and euclidean)
    mat_dist_geo = torch.zeros(num_decoders, M, N_pairs)
    mat_dist_euclid = torch.zeros(num_decoders, M, N_pairs)

    exp_dir = args.experiment_folder
    for decoder in range(num_decoders):
        dec_dir = "dec" + str(decoder)
        for m in range(M):
            run_dir = "run" + str(m)
            model_path = os.path.join(exp_dir, dec_dir, run_dir, "model.pth")
            
            model = create_model(args)
            model.load_state_dict(torch.load(model_path))
            all_latents, all_labels = encode_test_data(model, mnist_test_loader)

            # latent_pairs = []

            # Index for class 0 and class 1 in the test set
            class_0_idx = all_labels == 0
            class_1_idx = all_labels == 1

            # Choose 1 pair of latent codes from each class
            indices = [(class_0_idx.nonzero(as_tuple=True)[0][0], 
                        class_1_idx.nonzero(as_tuple=True)[0][0])]

            geo_lengths = torch.zeros(N_pairs)
            euclid_lengths = torch.zeros(N_pairs)
            # for (i, j) in latent_pairs:
            for p_ij, pair in tqdm(enumerate(indices)):
                z0, z1 = all_latents[pair[0]].to(device), all_latents[pair[1]].to(device)
                # latent_pairs.append((z0.cpu().numpy(), z1.cpu().numpy())) #TODO is this at all needed?
                
                # ensemble variable decides True or False
                initial_curve, final_curve = compute_geodesic(model, z0, z1, ensemble=ensemble)
                geo_lengths[p_ij] = compute_curve_length(model, final_curve.to(device), ensemble=ensemble)
                euclid_lengths[p_ij] = euclidean_curve_length(z0, z1)

            mat_dist_geo[decoder, m, :] = geo_lengths
            mat_dist_euclid[decoder, m, :] = euclid_lengths

        # Compute Coefficient of Variance for each decoder
        CoV_geo = CoefficientOfVariance(mat_dist_geo[decoder, :, :])
        CoV_euclid = CoefficientOfVariance(mat_dist_euclid[decoder, :, :])
        print(CoV_geo)
        print(CoV_euclid)

    #TODO: save CoVs together and plot them
    
