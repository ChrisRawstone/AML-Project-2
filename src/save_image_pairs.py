# save_image_pairs.py
# do not remove this comment or the comment above

import torch
from torchvision import datasets, transforms
from collections import defaultdict
from PIL import Image
import os
import random

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Create output folder
output_dir = "data/paired_images"
os.makedirs(output_dir, exist_ok=True)

# Load MNIST test dataset
full_dataset = datasets.MNIST(
    "data/", train=False, download=True, transform=transforms.ToTensor()
)
data, targets = full_dataset.data, full_dataset.targets

# Choose 3 classes to work with
selected_classes = [0, 1, 2]

# Filter to selected classes
filtered_indices = [i for i, label in enumerate(targets) if label in selected_classes]
filtered_data = data[filtered_indices]
filtered_targets = targets[filtered_indices]

# Downsample to 1000 total samples
selected_indices = torch.randperm(len(filtered_data))[:1000]
data_1000 = filtered_data[selected_indices]
targets_1000 = filtered_targets[selected_indices]

# Group indices by class
class_to_indices = defaultdict(list)
for idx, label in enumerate(targets_1000):
    class_to_indices[label.item()].append(idx)

# Define unique class combinations (both directions)
base_combinations = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]

# Fill in remaining pairs to reach 10
extra_combinations = [(0, 0), (1, 1), (2, 2), (0, 2)]  # arbitrary but valid extras

all_combinations = base_combinations + extra_combinations
assert len(all_combinations) == 10

pairs = []

# For each class combination, randomly sample one index from each class
for i, (class_a, class_b) in enumerate(all_combinations):
    indices_a = class_to_indices[class_a]
    indices_b = class_to_indices[class_b]

    if len(indices_a) == 0 or len(indices_b) == 0:
        continue  # skip if one class is empty (shouldn't happen with 1000 samples)

    idx_a = random.choice(indices_a)
    idx_b = random.choice(indices_b)
    pairs.append((idx_a, idx_b))

    # Save image pair
    pair_dir = os.path.join(output_dir, f"pair_{i}")
    os.makedirs(pair_dir, exist_ok=True)

    img1 = Image.fromarray(data_1000[idx_a].numpy(), mode="L")
    img2 = Image.fromarray(data_1000[idx_b].numpy(), mode="L")

    img1.save(os.path.join(pair_dir, "img1.png"))
    img2.save(os.path.join(pair_dir, "img2.png"))

# Save pairs as tensor
pairs_tensor = torch.tensor(pairs)
torch.save(pairs_tensor, "data/fixed_image_pairs.pt")

print(f"Saved {len(pairs)} image pairs to 'fixed_image_pairs.pt' and folder '{output_dir}/'")
