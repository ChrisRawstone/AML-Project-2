"""
Global script to import into all files for shared usage of variables.
"""

import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="sample",
    choices=["train", "sample", "eval", "geodesics"],
    help="what to do when running the script (default: %(default)s)",
)
parser.add_argument(
    "--experiment-folder",
    type=str,
    default="experiment",
    help="folder to save and load experiment results in (default: %(default)s)",
)
parser.add_argument(
    "--samples",
    type=str,
    default="samples.png",
    help="file to save samples in (default: %(default)s)",
)
# parser.add_argument(
#     "--device",
#     type=str,
#     default="cpu",
#     choices=["cpu", "cuda", "mps"],
#     help="torch device (default: %(default)s)",
# )
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    metavar="B",
    help="batch size for training (default: %(default)s)",
)
parser.add_argument(
    "--epochs-per-decoder",
    type=int,
    default=200,
    metavar="N",
    help="number of training epochs per each decoder (default: %(default)s)",
)
parser.add_argument(
    "--latent-dim",
    type=int,
    default=2,
    metavar="N",
    help="dimension of latent variable (default: %(default)s)",
)
parser.add_argument(
    "--num-decoders",
    type=int,
    default=1,
    metavar="N",
    help="number of decoders in the ensemble (default: %(default)s)",
)
parser.add_argument(
    "--num-reruns",
    type=int,
    default=10,
    metavar="N",
    help="number of reruns (default: %(default)s)",
)
parser.add_argument(
    "--num-curves",
    type=int,
    default=25,
    metavar="N",
    help="number of geodesics to plot (default: %(default)s)",
)
parser.add_argument(
    "--num-t",  # number of points along the curve
    type=int,
    default=20,
    metavar="N",
    help="number of points along the curve (default: %(default)s)",
)

args = parser.parse_args()
print("# Options")
for key, value in sorted(vars(args).items()):
    print(key, "=", value)