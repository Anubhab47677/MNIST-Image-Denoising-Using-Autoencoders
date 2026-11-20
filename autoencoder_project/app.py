import os
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import keras

from src.data_loader import load_mnist_data, add_gaussian_noise
from src.metrics import evaluate_predictions
from src.models import (
    build_conv_ae,
    build_denoising_ae,
    build_undercomplete_ae,
    build_contractive_ae,
    build_sparse_ae,
    ContractiveAutoencoder
)

st.set_page_config(
    page_title="MNIST Image Denoising Autoencoder Studio",
    page_icon="🔬",
    layout="wide"
)

# Title and Description
st.title("🔬 MNIST Image Denoising Using Autoencoders")
st.markdown("""
**Lab Assignment 04 Deployment Application**  
An interactive deep learning application to clean, denoise, and reconstruct handwritten digits using trained Autoencoders.
""")

POSSIBLE_MODEL_DIRS = [
    os.path.join(os.path.dirname(__file__), 'saved_models'),
    os.path.join(os.getcwd(), 'saved_models'),
    r"C:\Users\lenovo\OneDrive\Desktop\Autoencoder\autoencoder_project\saved_models",
    r"C:\Users\lenovo\OneDrive\Desktop\Autoencoder\saved_models"
]

CUSTOM_OBJECTS = {
    'ContractiveAutoencoder': ContractiveAutoencoder,
    'src.models>ContractiveAutoencoder': ContractiveAutoencoder
}

def find_model_path(model_choice):
    filename = f"{model_choice.lower()}_model.keras"
    for d in POSSIBLE_MODEL_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

@st.cache_resource
def get_dataset():
    (_, _), (x_test, y_test) = load_mnist_data()
    return x_test, y_test

@st.cache_resource
def get_model(model_choice):
    path = find_model_path(model_choice)
    if path:
        try:
            m = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)
            if 'Contractive' in model_choice:
                _ = m(tf.zeros((1, 784)))
            return m, True, path
        except Exception as e:
            if model_choice == 'Contractive_AE':
                try:
                    c_model = build_contractive_ae()
                    c_model.load_weights(path)
                    _ = c_model(tf.zeros((1, 784)))
                    return c_model, True, path
                except Exception as inner_e:
                    st.sidebar.error(f"Error loading {path}: {inner_e}")
            else:
                st.sidebar.error(f"Error loading {path}: {e}")
            
    # Fallback to freshly initialized model
    builders = {
        'Convolutional_AE': build_conv_ae,
        'Denoising_AE': build_denoising_ae,
        'Undercomplete_AE': build_undercomplete_ae,
        'Contractive_AE': build_contractive_ae,
        'Sparse_AE': build_sparse_ae
    }
    m = builders[model_choice]()
    if 'Contractive' in model_choice:
        _ = m(tf.zeros((1, 784)))
    return m, False, None

# Sidebar Controls
st.sidebar.header("⚙️ Model & Input Settings")

model_choice = st.sidebar.selectbox(
    "Select Autoencoder Architecture:",
    [
        "Convolutional_AE",
        "Denoising_AE",
        "Undercomplete_AE",
        "Contractive_AE",
        "Sparse_AE"
    ],
    index=0,
    help="Convolutional AE is the best-performing model based on the experimental results."
)

model, is_trained, loaded_from = get_model(model_choice)
is_conv = (model_choice == "Convolutional_AE")

if is_trained:
    st.sidebar.success(f"✅ Loaded Trained Model ({os.path.basename(loaded_from)})")
else:
    st.sidebar.warning(f"⚠️ Untrained Model Active for {model_choice}!")

input_source = st.sidebar.radio(
    "Input Image Source:",
    ["Upload Custom Image (PNG/JPG)", "Pick from MNIST Test Set"]
)

x_test, y_test = get_dataset()

clean_img = None
noisy_img = None
is_inverted = False

if input_source == "Pick from MNIST Test Set":
    digit_filter = st.sidebar.selectbox("Filter by digit label:", list(range(10)))
    matching_indices = np.where(y_test == digit_filter)[0]
    sample_index = st.sidebar.slider("Select sample index:", 0, min(100, len(matching_indices) - 1), 0)
    actual_idx = matching_indices[sample_index]
    
    clean_img = x_test[actual_idx]
    noise_sigma = st.sidebar.slider("Gaussian Noise Level (σ):", 0.0, 1.0, 0.3, 0.05)
    noisy_img = add_gaussian_noise(clean_img, noise_factor=noise_sigma, seed=42)

else:
    uploaded_file = st.sidebar.file_uploader("Upload a handwritten digit image:", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert('L').resize((28, 28))
        raw_arr = np.array(pil_img, dtype='float32') / 255.0
        
        # Robust inset corner check
        inset_corners = [raw_arr[2, 2], raw_arr[2, -3], raw_arr[-3, 2], raw_arr[-3, -3]]
        suggest_invert = (np.mean(raw_arr) > 0.5) and (np.mean(inset_corners) > 0.5)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🖼️ Preprocessing Options")
        invert_colors = st.sidebar.checkbox(
            "Invert Colors (Dark ink on light paper ➔ White on black)",
            value=suggest_invert,
            help="Uncheck if your image is already white digit on a black background."
        )
        
        if invert_colors:
            noisy_img = 1.0 - raw_arr
            is_inverted = True
        else:
            noisy_img = raw_arr
            
        add_noise = st.sidebar.checkbox("Add synthetic Gaussian noise", value=False)
        if add_noise:
            sigma = st.sidebar.slider("Synthetic Noise Level (σ):", 0.0, 1.0, 0.2, 0.05)
            noisy_img = add_gaussian_noise(noisy_img, noise_factor=sigma)
    else:
        st.info("👆 Please upload an image from the sidebar or choose 'Pick from MNIST Test Set' to begin.")

if noisy_img is not None:
    # Model inference
    if is_conv:
        input_tensor = np.expand_dims(noisy_img, axis=(0, -1))
    else:
        input_tensor = noisy_img.reshape(1, 784)
        
    prediction = model.predict(input_tensor, verbose=0)
    denoised_img = prediction.reshape(28, 28)

    # Visual Comparison Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Original / Uploaded")
        if clean_img is not None:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            ax1.imshow(clean_img, cmap='gray', vmin=0, vmax=1)
            ax1.axis('off')
            st.pyplot(fig1)
        else:
            fig0, ax0 = plt.subplots(figsize=(3, 3))
            ax0.imshow(raw_arr, cmap='gray', vmin=0, vmax=1)
            ax0.axis('off')
            st.pyplot(fig0)
            st.caption("Raw Uploaded Image")
            
    with col2:
        st.subheader("2. Model Input (MNIST Space)")
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
        ax2.axis('off')
        st.pyplot(fig2)
        st.caption("Fed into Autoencoder")
        
    with col3:
        st.subheader("3. Denoised Reconstruction")
        fig3, ax3 = plt.subplots(figsize=(3, 3))
        ax3.imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
        ax3.axis('off')
        st.pyplot(fig3)
        st.caption("Reconstructed Output")
        
    # Metrics if reference clean image available
    if clean_img is not None:
        metrics = evaluate_predictions(clean_img.reshape(1, 28, 28), denoised_img.reshape(1, 28, 28))
        input_metrics = evaluate_predictions(clean_img.reshape(1, 28, 28), noisy_img.reshape(1, 28, 28))
        
        st.markdown("### 📊 Reconstruction Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SSIM", f"{metrics['SSIM']:.4f}", delta=f"{metrics['SSIM'] - input_metrics['SSIM']:+.4f}")
        m2.metric("PSNR", f"{metrics['PSNR']:.2f} dB", delta=f"{metrics['PSNR'] - input_metrics['PSNR']:+.2f} dB")
        m3.metric("MSE", f"{metrics['MSE']:.5f}", delta=f"{metrics['MSE'] - input_metrics['MSE']:.5f}", delta_color="inverse")
        m4.metric("MAE", f"{metrics['MAE']:.5f}", delta=f"{metrics['MAE'] - input_metrics['MAE']:.5f}", delta_color="inverse")

    # Download Button
    buf = io.BytesIO()
    denoised_uint8 = (np.clip(denoised_img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(denoised_uint8).save(buf, format="PNG")
    st.download_button(
        label="📥 Download Denoised Image (PNG)",
        data=buf.getvalue(),
        file_name="denoised_digit.png",
        mime="image/png"
    )

with st.expander("ℹ️ Architecture & Experimental Comparison Details"):
    try:
        p_count = model.count_params()
        p_display = f"{p_count:,}"
    except Exception:
        p_display = "N/A"
        
    st.markdown(f"""
    - **Current Model**: `{model_choice}`
    - **Status**: `{'Trained model loaded' if is_trained else 'Untrained / Initialized model'}`
    - **Parameters**: `{p_display}`
    - **Note on Contrast**: Standard MNIST requires **white digit strokes on a black background**.
    """)
