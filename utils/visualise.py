import matplotlib.pyplot as plt
import torch

def show_reconstruction(input_img, recon_img, target_img=None):
    fig, axes = plt.subplots(1, 3 if target_img is not None else 2, figsize=(10, 4))

    axes[0].imshow(input_img.squeeze().cpu(), cmap="gray")
    axes[0].set_title("Input")

    axes[1].imshow(recon_img.squeeze().detach().cpu(), cmap="gray")
    axes[1].set_title("Reconstruction")

    if target_img is not None:
        axes[2].imshow(target_img.squeeze().cpu(), cmap="gray")
        axes[2].set_title("Ground Truth")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

