import os
import nibabel as nib
import torch
import argparse

def preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(".nii.gz"):
            img = nib.load(os.path.join(input_dir, fname)).get_fdata()
            # Normalize
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # Save slices
            for i in range(img.shape[2]):
                slice_ = torch.tensor(img[:, :, i], dtype=torch.float32).unsqueeze(0)
                torch.save(slice_, os.path.join(output_dir, f"{fname}_{i}.pt"))
    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)

