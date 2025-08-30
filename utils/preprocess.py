# utils/preprocess.py

import numpy as np

def normalize(volume):
    """
    Normalize MRI volume to zero mean, unit variance.
    """
    mean = np.mean(volume)
    std = np.std(volume)
    if std == 0:
        return volume - mean
    return (volume - mean) / std


def fft2c(img):
    """
    Centered 2D FFT (convert image -> k-space).
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))


def ifft2c(kspace):
    """
    Centered 2D inverse FFT (convert k-space -> image).
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))


def extract_patches(volume, patch_size=64, stride=64):
    """
    Extract non-overlapping 2D patches from a 3D volume (H, W, D).
    Returns: numpy array of patches (N, patch_size, patch_size).
    """
    h, w, d = volume.shape
    patches = []

    for z in range(d):
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch = volume[i:i+patch_size, j:j+patch_size, z]
                patches.append(patch)

    return np.array(patches)


def preprocess_mri(volume, patch_size=64):
    """
    Full preprocessing pipeline:
      1. Normalize MRI
      2. Convert to k-space
      3. Return reconstructed volume (magnitude only)
    """
    # Normalize
    norm_volume = normalize(volume)

    # Convert each slice to k-space
    h, w, d = norm_volume.shape
    recon_volume = np.zeros_like(norm_volume, dtype=np.float32)

    for z in range(d):
        kspace = fft2c(norm_volume[:, :, z])
        recon = np.abs(ifft2c(kspace))  # magnitude reconstruction
        recon_volume[:, :, z] = recon

    return recon_volume
