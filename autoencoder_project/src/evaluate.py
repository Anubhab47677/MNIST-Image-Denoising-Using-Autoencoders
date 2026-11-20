import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_loader import prepare_data, add_gaussian_noise, load_mnist_data
from src.metrics import evaluate_predictions
from src.models import (
    build_undercomplete_ae,
    build_sparse_ae,
    build_denoising_ae,
    build_conv_ae,
    build_contractive_ae
)

def load_or_rebuild_model(model_name, save_dir='saved_models'):
    """Attempts to load saved model from disk or rebuilds it."""
    file_path = os.path.join(save_dir, f"{model_name.lower()}_model.keras")
    if os.path.exists(file_path):
        try:
            return tf.keras.models.load_model(file_path, compile=False)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
    
    # Rebuild fallback
    builders = {
        'Undercomplete_AE': build_undercomplete_ae,
        'Sparse_AE': build_sparse_ae,
        'Denoising_AE': build_denoising_ae,
        'Convolutional_AE': build_conv_ae,
        'Contractive_AE': build_contractive_ae
    }
    return builders[model_name]()

def run_evaluation_suite(models_dict=None, save_dir='saved_models', output_dir='reports'):
    """
    Evaluates all five autoencoders on the full test set and produces comparison tables and figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = prepare_data(noise_factor=0.3)
    dense_data = data['dense']
    conv_data = data['conv']
    
    model_names = ['Undercomplete_AE', 'Sparse_AE', 'Denoising_AE', 'Convolutional_AE', 'Contractive_AE']
    
    if models_dict is None:
        models_dict = {}
        for name in model_names:
            models_dict[name] = load_or_rebuild_model(name, save_dir=save_dir)
            
    results = []
    reconstructions = {}
    
    for name in model_names:
        model = models_dict[name]
        is_conv = 'Convolutional' in name
        
        test_noisy = conv_data['test_noisy'] if is_conv else dense_data['test_noisy']
        test_clean = conv_data['test_clean'] if is_conv else dense_data['test_clean']
        
        preds = model.predict(test_noisy, batch_size=256, verbose=0)
        reconstructions[name] = preds
        
        clean_eval = test_clean.reshape(-1, 28, 28)
        preds_eval = preds.reshape(-1, 28, 28)
        
        metrics = evaluate_predictions(clean_eval, preds_eval)
        
        param_count = model.count_params()
        results.append({
            'Model': name,
            'Parameters': param_count,
            'MSE': metrics['MSE'],
            'MAE': metrics['MAE'],
            'PSNR (dB)': metrics['PSNR'],
            'SSIM': metrics['SSIM']
        })
        
    df_results = pd.DataFrame(results)
    table_path = os.path.join(output_dir, 'final_metrics.csv')
    df_results.to_csv(table_path, index=False)
    print("\n--- Model Evaluation on Test Set ---")
    print(df_results.to_string(index=False))
    print(f"\nSaved metrics to {table_path}")
    
    # Plot reconstruction gallery of first 10 digits
    plot_reconstruction_gallery(
        clean_images=dense_data['test_clean'][:10].reshape(10, 28, 28),
        noisy_images=dense_data['test_noisy'][:10].reshape(10, 28, 28),
        model_predictions={k: v[:10].reshape(10, 28, 28) for k, v in reconstructions.items()},
        save_path=os.path.join(output_dir, 'reconstruction_gallery.png')
    )
    
    return df_results, reconstructions

def plot_reconstruction_gallery(clean_images, noisy_images, model_predictions, save_path):
    """
    Plots Original vs Noisy vs Model Reconstructions for 10 digits.
    """
    num_samples = len(clean_images)
    num_models = len(model_predictions)
    total_rows = 2 + num_models
    
    fig, axes = plt.subplots(total_rows, num_samples, figsize=(num_samples * 1.5, total_rows * 1.5))
    
    # Row 0: Original Clean
    for i in range(num_samples):
        axes[0, i].imshow(clean_images[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Clean Original', loc='left', fontsize=10, fontweight='bold')
            
    # Row 1: Noisy Input
    for i in range(num_samples):
        axes[1, i].imshow(noisy_images[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noisy Input (σ=0.3)', loc='left', fontsize=10, fontweight='bold')
            
    # Subsequent rows: Model reconstructions
    for r_idx, (m_name, preds) in enumerate(model_predictions.items(), start=2):
        for i in range(num_samples):
            axes[r_idx, i].imshow(preds[i], cmap='gray')
            axes[r_idx, i].axis('off')
            if i == 0:
                axes[r_idx, i].set_title(m_name, loc='left', fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved gallery comparison to {save_path}")

def run_noise_robustness_experiment(models_dict, noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5], num_test=1000, output_dir='reports'):
    """
    Tests trained models against varied test noise levels without retraining.
    """
    os.makedirs(output_dir, exist_ok=True)
    (x_train, _), (x_test, _) = load_mnist_data()
    clean_subset = x_test[:num_test]
    
    rows = []
    for sigma in noise_levels:
        noisy_subset = add_gaussian_noise(clean_subset, noise_factor=sigma, seed=int(sigma*100))
        noisy_flat = noisy_subset.reshape(-1, 784)
        noisy_conv = np.expand_dims(noisy_subset, axis=-1)
        
        for name, model in models_dict.items():
            is_conv = 'Convolutional' in name
            inp = noisy_conv if is_conv else noisy_flat
            preds = model.predict(inp, batch_size=256, verbose=0).reshape(-1, 28, 28)
            
            metrics = evaluate_predictions(clean_subset, preds)
            rows.append({
                'Sigma': sigma,
                'Model': name,
                'MSE': metrics['MSE'],
                'MAE': metrics['MAE'],
                'PSNR': metrics['PSNR'],
                'SSIM': metrics['SSIM']
            })
            
    df_noise = pd.DataFrame(rows)
    save_path = os.path.join(output_dir, 'noise_robustness_results.csv')
    df_noise.to_csv(save_path, index=False)
    print(f"Saved noise robustness experiment results to {save_path}")
    return df_noise

def run_latent_dimension_experiment(latent_dims=[8, 16, 32, 64, 128], epochs=15, batch_size=128, output_dir='reports'):
    """
    Tests varying bottleneck dimensions on undercomplete autoencoders.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = prepare_data(noise_factor=0.3)
    dense_data = data['dense']
    
    results = []
    for k in latent_dims:
        print(f"Training latent dimension k = {k}...")
        model = build_undercomplete_ae(latent_dim=k)
        model.fit(
            dense_data['train_noisy'], dense_data['train_clean'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(dense_data['val_noisy'], dense_data['val_clean']),
            verbose=0
        )
        preds = model.predict(dense_data['test_noisy'], batch_size=256, verbose=0)
        clean = dense_data['test_clean'].reshape(-1, 28, 28)
        denoised = preds.reshape(-1, 28, 28)
        
        metrics = evaluate_predictions(clean, denoised)
        results.append({
            'Latent_Dim': k,
            'Parameters': model.count_params(),
            'MSE': metrics['MSE'],
            'MAE': metrics['MAE'],
            'PSNR': metrics['PSNR'],
            'SSIM': metrics['SSIM']
        })
        
    df_latent = pd.DataFrame(results)
    save_path = os.path.join(output_dir, 'latent_dimension_results.csv')
    df_latent.to_csv(save_path, index=False)
    print(f"Saved latent dimension experiment results to {save_path}")
    return df_latent

if __name__ == '__main__':
    run_evaluation_suite()
