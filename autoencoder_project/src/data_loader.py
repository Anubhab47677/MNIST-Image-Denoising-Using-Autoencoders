import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    """
    Loads MNIST dataset and normalizes pixel values to [0.0, 1.0].
    Returns:
        (x_train, y_train), (x_test, y_test) as float32 numpy arrays.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

def add_gaussian_noise(images, noise_factor=0.3, seed=None):
    """
    Adds additive zero-mean Gaussian noise with scale `noise_factor`
    and clips the resulting pixel values strictly to [0.0, 1.0].
    """
    if seed is not None:
        np.random.seed(seed)
    noisy_images = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = np.clip(noisy_images, 0.0, 1.0)
    return noisy_images.astype('float32')

def prepare_data(noise_factor=0.3, val_split=0.1, seed=42):
    """
    Prepares training, validation, and test datasets in both flattened (784)
    and spatial (28, 28, 1) formats.
    """
    (x_train_clean, y_train), (x_test_clean, y_test) = load_mnist_data()
    
    # Validation split from training partition (last val_split portion)
    num_train = int(len(x_train_clean) * (1 - val_split))
    
    x_val_clean = x_train_clean[num_train:]
    x_train_clean = x_train_clean[:num_train]
    
    # Generate noisy counterparts
    x_train_noisy = add_gaussian_noise(x_train_clean, noise_factor, seed=seed)
    x_val_noisy = add_gaussian_noise(x_val_clean, noise_factor, seed=seed + 1 if seed else None)
    x_test_noisy = add_gaussian_noise(x_test_clean, noise_factor, seed=seed + 2 if seed else None)
    
    # Dense layouts (N, 784)
    x_train_clean_flat = x_train_clean.reshape(-1, 784)
    x_train_noisy_flat = x_train_noisy.reshape(-1, 784)
    x_val_clean_flat = x_val_clean.reshape(-1, 784)
    x_val_noisy_flat = x_val_noisy.reshape(-1, 784)
    x_test_clean_flat = x_test_clean.reshape(-1, 784)
    x_test_noisy_flat = x_test_noisy.reshape(-1, 784)
    
    # Convolutional layouts (N, 28, 28, 1)
    x_train_clean_conv = np.expand_dims(x_train_clean, axis=-1)
    x_train_noisy_conv = np.expand_dims(x_train_noisy, axis=-1)
    x_val_clean_conv = np.expand_dims(x_val_clean, axis=-1)
    x_val_noisy_conv = np.expand_dims(x_val_noisy, axis=-1)
    x_test_clean_conv = np.expand_dims(x_test_clean, axis=-1)
    x_test_noisy_conv = np.expand_dims(x_test_noisy, axis=-1)
    
    return {
        'dense': {
            'train_clean': x_train_clean_flat,
            'train_noisy': x_train_noisy_flat,
            'val_clean': x_val_clean_flat,
            'val_noisy': x_val_noisy_flat,
            'test_clean': x_test_clean_flat,
            'test_noisy': x_test_noisy_flat,
        },
        'conv': {
            'train_clean': x_train_clean_conv,
            'train_noisy': x_train_noisy_conv,
            'val_clean': x_val_clean_conv,
            'val_noisy': x_val_noisy_conv,
            'test_clean': x_test_clean_conv,
            'test_noisy': x_test_noisy_conv,
        },
        'labels': {
            'train': y_train[:num_train],
            'val': y_train[num_train:],
            'test': y_test
        }
    }
