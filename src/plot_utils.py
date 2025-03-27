# src/plot_utils.py
# do not remove this comment or the comment above


import matplotlib.pyplot as plt
import numpy as np

def plot_energy_curve(energy_history, save_path=None):
    plt.figure(figsize=(8,6))
    plt.plot(energy_history, linewidth=2)
    plt.xlabel('Optimization Step')
    plt.ylabel('Energy')
    plt.title('Energy Curve During Geodesic Optimization')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_latent_space(latents, labels, save_path=None):
    if hasattr(latents, 'detach'):
        latents = latents.detach().cpu().numpy()
    if hasattr(labels, 'detach'):
        labels = labels.detach().cpu().numpy()
    num_classes = len(np.unique(labels))
    cmap = plt.get_cmap('viridis', num_classes)
    
    plt.figure(figsize=(8,6))
    plt.scatter(latents[:,0], latents[:,1], c=labels, cmap=cmap, edgecolor='k', s=40, alpha=0.7)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space')
    plt.grid(True)
    
    handles = []
    for i in range(num_classes):
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color=cmap(i), label=f"Class {i}"))
    plt.legend(handles=handles, title="Labels", loc="best")
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_geodesic_comparison(latents, labels, z_initial, z_optimized, z_start, z_end, save_path=None):
    if hasattr(latents, 'detach'):
        latents_np = latents.detach().cpu().numpy()
    else:
        latents_np = latents
    if hasattr(labels, 'detach'):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = labels

    cmap = plt.get_cmap('viridis', len(np.unique(labels_np)))
    
    plt.figure(figsize=(8,6))
    plt.scatter(latents_np[:,0], latents_np[:,1], c=labels_np, cmap=cmap, edgecolor='k', s=40, alpha=0.7)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Initial vs Optimized Geodesic")
    plt.grid(True)

    z_initial_np = z_initial.detach().cpu().numpy() if hasattr(z_initial, 'detach') else z_initial
    plt.plot(z_initial_np[:,0], z_initial_np[:,1], 'o--', color='magenta', linewidth=2, markersize=8, label="Initial Geodesic")
    
    z_opt_np = z_optimized.detach().cpu().numpy() if hasattr(z_optimized, 'detach') else z_optimized
    plt.plot(z_opt_np[:,0], z_opt_np[:,1], 'o-', color='blue', linewidth=2, markersize=8, label="Optimized Geodesic")
    
    z_start_np = z_start.detach().cpu().numpy() if hasattr(z_start, 'detach') else z_start
    z_end_np   = z_end.detach().cpu().numpy() if hasattr(z_end, 'detach') else z_end
    plt.scatter(z_start_np[0], z_start_np[1], color='green', s=120, marker='s', label='Start')
    plt.scatter(z_end_np[0], z_end_np[1], color='red', s=120, marker='s', label='End')
    
    plt.legend(loc="best")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()