import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

def compute_mse(clean, denoised):
    """Computes Mean Squared Error across all pixels."""
    return float(np.mean((clean - denoised) ** 2))

def compute_mae(clean, denoised):
    """Computes Mean Absolute Error across all pixels."""
    return float(np.mean(np.abs(clean - denoised)))

def compute_psnr(clean, denoised, max_val=1.0):
    """Computes Peak Signal-to-Noise Ratio (dB)."""
    mse = compute_mse(clean, denoised)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10((max_val ** 2) / mse))

def compute_ssim(clean, denoised):
    """
    Computes average Structural Similarity Index (SSIM) over a batch of images.
    Input images can be (N, 784) or (N, 28, 28) or (N, 28, 28, 1).
    """
    clean_imgs = clean.reshape(-1, 28, 28)
    denoised_imgs = denoised.reshape(-1, 28, 28)
    
    total_ssim = 0.0
    num_samples = len(clean_imgs)
    for i in range(num_samples):
        score = ssim_fn(clean_imgs[i], denoised_imgs[i], data_range=1.0)
        total_ssim += score
    return float(total_ssim / num_samples)

def evaluate_predictions(clean, denoised):
    """
    Computes all 4 evaluation metrics defined in Assignment 4.
    Returns:
        dict: {'MSE': ..., 'MAE': ..., 'PSNR': ..., 'SSIM': ...}
    """
    return {
        'MSE': compute_mse(clean, denoised),
        'MAE': compute_mae(clean, denoised),
        'PSNR': compute_psnr(clean, denoised),
        'SSIM': compute_ssim(clean, denoised)
    }
