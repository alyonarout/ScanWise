# datasets/oasis_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib   # for reading NIfTI MRI scans
from utils.preprocess import preprocess_mri  # custom preprocessing

class OASISDataset(Dataset):
    """
    OASIS MRI Dataset Loader

    Loads 3D MRI scans from NIfTI (.nii / .nii.gz),
    applies preprocessing (normalization, k-space, patching),
    and returns patches as PyTorch tensors.
    """

    def __init__(self, root_dir, transform=None, patch_size=64, mode="train"):
        """
        Args:
            root_dir (str): Path to dataset folder (e.g., data/oasis/)
            transform (callable, optional): Optional transform on a sample
            patch_size (int): size of square patch to extract
            mode (str): "train", "val", or "test"
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size
        self.mode = mode

        # collect all nifti files
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                      if f.endswith(".nii") or f.endswith(".nii.gz")]

        if len(self.files) == 0:
            raise RuntimeError(f"No NIfTI files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        nifti_path = self.files[idx]

        # load MRI volume
        volume = nib.load(nifti_path).get_fdata().astype(np.float32)

        # preprocess (normalize + convert to k-space + patchify)
        volume = preprocess_mri(volume, patch_size=self.patch_size)

        # pick a random patch (for training)
        if self.mode == "train":
            patch = self._get_random_patch(volume)
        else:
            patch = volume  # return full preprocessed volume in val/test

        # convert to torch tensor
        patch = torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)

        if self.transform:
            patch = self.transform(patch)

        sample = {
            "image": patch,
            "path": nifti_path
        }
        return sample

    def _get_random_patch(self, volume):
        """Extract a random 2D patch from a 3D volume."""
        h, w, d = volume.shape
        ph, pw = self.patch_size, self.patch_size

        # pick random slice & location
        z = np.random.randint(0, d)
        x = np.random.randint(0, h - ph)
        y = np.random.randint(0, w - pw)

        return volume[x:x+ph, y:y+pw, z]


if __name__ == "__main__":
    # quick test
    dataset = OASISDataset(root_dir="data/oasis", patch_size=64)
    sample = dataset[0]
    print("Sample shape:", sample["image"].shape)

