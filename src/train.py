import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_loader import prepare_data
from src.models import (
    build_undercomplete_ae,
    build_sparse_ae,
    build_denoising_ae,
    build_conv_ae,
    build_contractive_ae
)

def train_single_model(model_name, model, x_train, y_train, x_val, y_val, epochs=15, batch_size=128, save_dir='saved_models'):
    """
    Trains a model, records execution time and history, and saves the trained model.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n==========================================")
    print(f"  Training {model_name} ({epochs} epochs, batch {batch_size})")
    print(f"==========================================")
    
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_val, y_val),
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Finished {model_name} in {training_time:.2f} seconds.")
    
    # Save model weights/architecture
    save_path = os.path.join(save_dir, f"{model_name.lower()}_model.keras")
    try:
        model.save(save_path)
        print(f"Saved model to: {save_path}")
    except Exception as e:
        print(f"Warning: Could not save as .keras format directly: {e}. Saving weights instead.")
        weights_path = os.path.join(save_dir, f"{model_name.lower()}_weights.weights.h5")
        model.save_weights(weights_path)
        print(f"Saved weights to: {weights_path}")
        
    return {
        'model_name': model_name,
        'training_time': training_time,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }

def train_all_models(epochs=15, batch_size=128, save_dir='saved_models', noise_factor=0.3):
    """
    Runs full pipeline to train all 5 autoencoders under identical protocol.
    """
    data = prepare_data(noise_factor=noise_factor)
    dense_data = data['dense']
    conv_data = data['conv']
    
    models_to_train = [
        ("Undercomplete_AE", build_undercomplete_ae(), dense_data['train_noisy'], dense_data['train_clean'], dense_data['val_noisy'], dense_data['val_clean']),
        ("Sparse_AE", build_sparse_ae(), dense_data['train_noisy'], dense_data['train_clean'], dense_data['val_noisy'], dense_data['val_clean']),
        ("Denoising_AE", build_denoising_ae(), dense_data['train_noisy'], dense_data['train_clean'], dense_data['val_noisy'], dense_data['val_clean']),
        ("Convolutional_AE", build_conv_ae(), conv_data['train_noisy'], conv_data['train_clean'], conv_data['val_noisy'], conv_data['val_clean']),
        ("Contractive_AE", build_contractive_ae(), dense_data['train_noisy'], dense_data['train_clean'], dense_data['val_noisy'], dense_data['val_clean']),
    ]
    
    training_results = {}
    for name, model, x_tr, y_tr, x_v, y_v in models_to_train:
        res = train_single_model(name, model, x_tr, y_tr, x_v, y_v, epochs=epochs, batch_size=batch_size, save_dir=save_dir)
        training_results[name] = res
        
    # Save training log
    log_path = os.path.join(save_dir, 'training_summary.json')
    with open(log_path, 'w') as f:
        json.dump(training_results, f, indent=4)
    print(f"\nTraining summary saved to {log_path}")
    return training_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Autoencoders for MNIST Image Denoising")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--noise', type=float, default=0.3, help="Training noise factor")
    parser.add_argument('--save_dir', type=str, default='saved_models', help="Directory to save models")
    args = parser.parse_args()
    
    train_all_models(epochs=args.epochs, batch_size=args.batch_size, save_dir=args.save_dir, noise_factor=args.noise)
