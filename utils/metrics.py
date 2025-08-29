import torch
import torch.nn.functional as F
import math
from skimage.metrics import structural_similarity as ssim

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())

def ssim_score(pred, target):
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    return ssim(pred_np, target_np, data_range=1.0)

