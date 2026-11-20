<div align="center">

# MNIST Image Denoising Using Autoencoders

### A Comparative Deep Learning Study of Five Autoencoder Architectures

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MNIST-2E8B57)](http://yann.lecun.com/exdb/mnist/)

**An end-to-end image denoising project that trains, evaluates, and compares five autoencoder architectures on Gaussian-noise-corrupted MNIST digits, with an interactive Streamlit application for experimentation.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Problem Statement](#problem-statement)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Autoencoder Architectures](#autoencoder-architectures)
- [Experimental Setup](#experimental-setup)
- [Results and Model Comparison](#results-and-model-comparison)
- [Additional Experiments](#additional-experiments)
- [Interactive Web Application](#interactive-web-application)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Saved Models](#saved-models)
- [Implementation Notes](#implementation-notes)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

## Overview

Image denoising is a fundamental image-restoration task in which a model learns to recover a clean image from a corrupted observation. This project explores **deep autoencoders** as data-driven denoising models using the MNIST handwritten-digit dataset.

Five architectures are implemented under a common denoising workflow:

1. Undercomplete Autoencoder
2. Sparse Autoencoder
3. Denoising Autoencoder
4. Convolutional Autoencoder
5. Contractive Autoencoder

The models are trained to map noisy digit images to their corresponding clean targets. Their reconstructions are compared quantitatively using Mean Squared Error (MSE), Mean Absolute Error (MAE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).

The repository includes a modular Python implementation, an executable Jupyter Notebook containing the experimental workflow, saved Keras models, training histories, and a Streamlit interface for interactive inference.

## Highlights

- Five distinct autoencoder designs implemented with TensorFlow/Keras.
- MNIST image preprocessing and additive Gaussian noise generation.
- Paired noisy-input and clean-target training.
- Dense and convolutional input pipelines.
- Custom contractive autoencoder training step and regularization term.
- Model benchmarking using four image-quality metrics.
- Noise-level robustness experiment across multiple Gaussian noise levels.
- Latent-dimension experiment for undercomplete autoencoders.
- Saved `.keras` models and training-history JSON.
- Streamlit interface for MNIST examples and user-uploaded digit images.
- Downloadable denoised PNG output.

## Problem Statement

Handwritten digit images can be degraded by acquisition noise, scanning artifacts, or other unwanted variations. A denoising model should preserve the meaningful digit structure while suppressing noise.

The objective of this project is to investigate how different autoencoder constraints and architectures affect reconstruction quality. Each model receives a noisy image and learns to reconstruct its clean counterpart:

\[
\text{Noisy Image} \xrightarrow{\text{Encoder}} \text{Latent Representation}
\xrightarrow{\text{Decoder}} \text{Reconstructed Image}
\]

For a clean target image \(x\), a noisy observation \(\tilde{x}\), and a model \(f_\theta\), the reconstruction objective can be represented generally as:

\[
\hat{x}=f_\theta(\tilde{x})
\]

The model is optimized so that \(\hat{x}\) is as close as possible to \(x\), with architecture-specific regularization where applicable.

## Dataset and Preprocessing

### MNIST

The project uses the **MNIST handwritten-digit dataset**, which contains grayscale images of handwritten digits from 0 to 9.

| Property | Description |
|---|---|
| Dataset | MNIST |
| Training images | 60,000 |
| Test images | 10,000 |
| Image dimensions | 28 × 28 pixels |
| Channels | 1 (grayscale) |
| Pixel range after normalization | [0, 1] |
| Training/validation split | 54,000 / 6,000 |

The validation set is created from the original 60,000-image training partition. The original 10,000-image test partition is retained for final evaluation.

### Preprocessing pipeline

1. Load MNIST using `tensorflow.keras.datasets.mnist`.
2. Convert image arrays to `float32`.
3. Normalize pixel intensities from `[0, 255]` to `[0, 1]`.
4. Split the training partition into training and validation subsets.
5. Generate noisy copies using additive, zero-mean Gaussian noise.
6. Clip noisy pixel values to the valid `[0, 1]` range.
7. Prepare flattened vectors for dense models and image tensors for the convolutional model.

Gaussian noise is generated as:

\[
\tilde{x} = \operatorname{clip}(x + \sigma \epsilon, 0, 1),
\qquad \epsilon \sim \mathcal{N}(0,1)
\]

The primary training configuration uses a noise factor of **σ = 0.3**.

### Input formats

| Model family | Input shape |
|---|---|
| Dense autoencoders | `(N, 784)` |
| Convolutional autoencoder | `(N, 28, 28, 1)` |

The target is always the corresponding clean image in the matching format.

## Autoencoder Architectures

### 1. Undercomplete Autoencoder

The undercomplete autoencoder learns a compact representation by forcing the input through a lower-dimensional latent bottleneck.

**Dense architecture:**

`784 → 256 → 128 → 32 → 128 → 256 → 784`

- ReLU activations in hidden layers.
- Sigmoid output layer.
- Latent dimension: 32.
- Reconstruction loss: Mean Squared Error (MSE).

**Purpose:** Establishes a compact-representation baseline and studies how dimensionality reduction affects denoising.

### 2. Sparse Autoencoder

The sparse autoencoder adds an L1 activity regularization term to the latent representation. This encourages only a subset of latent activations to become strongly active for a given input.

**Architecture:**

`784 → 256 → 128 → 32 → 128 → 256 → 784`

- Latent dimension: 32.
- L1 activity regularization: \(10^{-5}\).
- ReLU hidden activations and sigmoid output.
- Reconstruction loss: MSE, with the latent activity regularization added by Keras.

**Purpose:** Investigates whether sparse feature activation improves the quality and structure of learned representations.

### 3. Denoising Autoencoder

The denoising autoencoder is trained using corrupted inputs and clean reconstruction targets. Its latent layer has a larger capacity than the undercomplete and sparse models.

**Architecture:**

`784 → 256 → 128 → 64 → 128 → 256 → 784`

- Latent dimension: 64.
- ReLU hidden activations.
- Sigmoid output layer.
- Reconstruction loss: MSE.

**Purpose:** Learns to remove corruption by reconstructing the clean target from the noisy input.

### 4. Convolutional Autoencoder

The convolutional autoencoder preserves the spatial structure of images through convolution, pooling, and upsampling operations.

**Encoder:**

- Conv2D: 32 filters, 3 × 3, ReLU.
- MaxPooling2D: 2 × 2.
- Conv2D: 64 filters, 3 × 3, ReLU.
- MaxPooling2D: 2 × 2.
- Conv2D latent feature map: 64 filters, 3 × 3, ReLU.

**Decoder:**

- UpSampling2D: 2 × 2.
- Conv2D: 32 filters, 3 × 3, ReLU.
- UpSampling2D: 2 × 2.
- Conv2D output: 1 filter, 3 × 3, sigmoid.

- Input/output shape: 28 × 28 × 1.
- Reconstruction loss: Binary Cross-Entropy (BCE).
- Parameter count: 74,497.

**Purpose:** Uses local spatial features and shared convolutional filters to reconstruct digit structure. In the recorded experiment, this architecture achieved the strongest test-set reconstruction metrics among the five implemented models.

### 5. Contractive Autoencoder

The contractive autoencoder uses a custom Keras model and training step. It combines reconstruction error with a contractive regularization term based on the latent activations.

**Architecture:**

`784 → 256 → 32 → 128 → 784`

- Latent dimension: 32.
- Sigmoid latent activation.
- ReLU intermediate decoder activation.
- Sigmoid output activation.
- Reconstruction loss: MSE.
- Contractive coefficient: \(\lambda = 10^{-4}\).

The implementation computes the regularization term as:

\[
\mathcal{L}_{c}
=
\operatorname{mean}\left((h \odot (1-h))^2\right)
\]

and combines it with reconstruction loss:

\[
\mathcal{L}_{total}
=
\mathcal{L}_{reconstruction}
+
\lambda \mathcal{L}_{c}
\]

**Purpose:** Encourages a representation that is less sensitive to small changes in the input.

## Experimental Setup

The main notebook experiment uses the following configuration:

| Parameter | Setting |
|---|---|
| Dataset | MNIST |
| Training images | 54,000 |
| Validation images | 6,000 |
| Test images | 10,000 |
| Image size | 28 × 28 |
| Noise distribution | Zero-mean Gaussian |
| Training noise factor | 0.3 |
| Optimizer | Adam |
| Learning rate | 0.001 (model defaults) |
| Epochs | 15 |
| Batch size | 128 |
| Validation split | 10% of original training set |
| Dense input shape | 784 |
| Convolutional input shape | 28 × 28 × 1 |

The dense autoencoders use MSE reconstruction loss, except for the sparse model which additionally uses L1 activity regularization. The convolutional model uses BCE. The contractive model uses its custom combined loss.

## Results and Model Comparison

The following results are taken from the recorded quantitative comparison in `Image_Denoising.ipynb`. They represent the notebook's saved experiment output and are not a guarantee that a fresh training run will reproduce identical values.

### Test-set performance

| Model | Parameters | MSE ↓ | MAE ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|
| Undercomplete AE | 476,720 | 0.012938 | 0.039645 | 18.8814 | 0.8480 |
| Sparse AE | 476,720 | 0.027640 | 0.070623 | 15.5845 | 0.6817 |
| Denoising AE | 484,944 | 0.010467 | 0.035024 | 19.8020 | 0.8674 |
| **Convolutional AE** | **74,497** | **0.005041** | **0.022482** | **22.9745** | **0.9378** |
| Contractive AE | 314,544 | 0.018531 | 0.054573 | 17.3211 | 0.7657 |

**Best overall model in the recorded comparison: Convolutional Autoencoder.**

It achieved the lowest MSE and MAE and the highest PSNR and SSIM among the five models. It also used fewer trainable parameters than the dense architectures in this comparison.

### Interpretation

- **Convolutional AE:** Best recorded reconstruction quality across all four metrics.
- **Denoising AE:** Second-best overall metric profile in the recorded comparison.
- **Undercomplete AE:** Provides a useful compact latent-representation baseline.
- **Sparse AE:** Had the weakest scores in this particular setup, indicating that the chosen sparsity constraint did not improve reconstruction quality under the recorded configuration.
- **Contractive AE:** Adds a custom regularization objective, but its recorded reconstruction metrics are below the undercomplete and denoising models.

The result illustrates that preserving spatial information can be valuable for image reconstruction tasks. It does not establish that convolutional autoencoders will outperform dense architectures on every dataset or noise distribution.

## Additional Experiments

### 1. Noise-level robustness

The notebook evaluates the trained models at multiple Gaussian noise levels without retraining:

\[
\sigma \in \{0.1,\ 0.2,\ 0.3,\ 0.4,\ 0.5\}
\]

This experiment studies how reconstruction quality changes as the input corruption becomes stronger. The notebook includes a comparative visualization of the noise-level experiment.

### 2. Latent-dimension experiment

The notebook also trains undercomplete autoencoders with different bottleneck dimensions:

\[
k \in \{8,\ 16,\ 32,\ 64,\ 128\}
\]

Recorded results:

| Latent dimension | Parameters | MSE ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---:|---:|---:|---:|---:|
| 8 | 203,800 | 0.025793 | 15.8851 | 0.7057 |
| 16 | 205,856 | 0.018334 | 17.3674 | 0.7852 |
| 32 | 209,968 | 0.013787 | 18.6054 | 0.8236 |
| 64 | 218,192 | 0.010701 | 19.7059 | 0.8475 |
| 128 | 234,640 | 0.009455 | 20.2433 | 0.8521 |

In this experiment, increasing the latent dimension improved the recorded reconstruction metrics, while also increasing the number of parameters. The experiment highlights the trade-off between representation capacity and compression.

> **Experimental scope:** The latent-dimension experiment is a separate notebook experiment. Its parameter counts should not be confused with the five final model counts in the main model-comparison table.

## Interactive Web Application

The project includes a Streamlit application (`app.py`) for trying the trained autoencoders without manually running inference code.

### Application capabilities

- Select a model architecture from the sidebar.
- Choose a sample from the MNIST test set.
- Upload a handwritten digit image in PNG, JPG, or JPEG format.
- Resize uploaded images to the MNIST input dimensions.
- Optionally invert image colors to match the MNIST convention (white digit strokes on a black background).
- Optionally add synthetic Gaussian noise.
- View the original/uploaded image, model input, and denoised reconstruction.
- For MNIST test samples, compare input and reconstructed image quality using SSIM, PSNR, MSE, and MAE.
- Download the reconstructed image as a PNG file.
- Inspect the selected model's parameter count and loading status.

### Image convention

MNIST images use **white digit strokes on a black background**. Uploaded images with dark ink on a light background may need color inversion. The application provides an inversion option and a basic automatic suggestion based on image intensity.

### Run the application

```bash
streamlit run app.py
```

The application will print a local URL, usually:

```text
http://localhost:8501
```

Open that address in your browser.

> **Important:** The application code has a fallback path that can initialize a fresh model if a saved model cannot be loaded. An initialized fallback model is not trained and should not be treated as a valid denoising result. Check the model status shown in the application before interpreting its output.

## Project Structure

```text
autoencoder_project/
│
├── app.py
├── Image_Denoising.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
│
└── saved_models/
    ├── undercomplete_ae_model.keras
    ├── sparse_ae_model.keras
    ├── denoising_ae_model.keras
    ├── convolutional_ae_model.keras
    ├── contractive_ae_model.keras
    └── training_summary.json
```

### Core modules

| File | Responsibility |
|---|---|
| `src/data_loader.py` | Loads MNIST, normalizes images, injects Gaussian noise, creates train/validation/test data, and prepares dense/convolutional tensors. |
| `src/models.py` | Defines the five autoencoder architectures, including the custom contractive model. |
| `src/train.py` | Trains the five models, records training duration and history, and saves models. |
| `src/evaluate.py` | Runs model evaluation, creates comparison outputs, and performs noise and latent-dimension experiments. |
| `src/metrics.py` | Computes MSE, MAE, PSNR, and SSIM. |
| `app.py` | Provides the interactive Streamlit inference interface. |
| `Image_Denoising.ipynb` | Presents the end-to-end notebook workflow, visualizations, and recorded experiments. |
| `saved_models/` | Contains trained Keras models and training-history metadata. |

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep learning | TensorFlow, Keras |
| Numerical computing | NumPy |
| Data analysis | Pandas |
| Visualization | Matplotlib |
| Image-quality evaluation | scikit-image |
| Image handling | Pillow |
| Interactive interface | Streamlit |
| Experiment environment | Jupyter Notebook / Google Colab |

## Installation and Setup

### Prerequisites

- Python 3.10 or later is recommended.
- `pip` package manager.
- Git (optional, for cloning the repository).
- A GPU is optional; CPU execution is possible but may take longer.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd autoencoder_project
```

Replace the placeholder with the actual GitHub repository URL.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current `requirements.txt` includes:

```text
tensorflow>=2.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-image>=0.21.0
pandas>=2.0.0
streamlit>=1.30.0
Pillow>=9.5.0
```

For notebook execution, ensure Jupyter is available in your environment:

```bash
pip install notebook
```

### 4. Verify the installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

If a compatible GPU and TensorFlow GPU runtime are configured, TensorFlow can use the available accelerator. Otherwise, the project can run on CPU.

## Usage

Run commands from the `autoencoder_project` root directory.

### Option A: Run the complete notebook

Open the notebook in Jupyter:

```bash
jupyter notebook Image_Denoising.ipynb
```

Alternatively, open `Image_Denoising.ipynb` in VS Code or upload it to Google Colab.

Run the cells in order to execute preprocessing, training, visualizations, model evaluation, and the additional experiments.

### Option B: Train all five models from the command line

```bash
python -m src.train --epochs 15 --batch_size 128 --noise 0.3
```

Optional arguments:

| Argument | Default | Description |
|---|---:|---|
| `--epochs` | 15 | Number of training epochs |
| `--batch_size` | 128 | Training batch size |
| `--noise` | 0.3 | Gaussian noise factor |
| `--save_dir` | `saved_models` | Directory in which models and training history are saved |

Example with a different noise level:

```bash
python -m src.train --epochs 20 --batch_size 128 --noise 0.4
```

Training saves the model files and `training_summary.json` in the selected save directory.

### Option C: Evaluate the saved models

```bash
python -m src.evaluate
```

The evaluation module loads saved models where possible, evaluates reconstructions, and writes generated evaluation outputs to a `reports/` directory.

### Option D: Launch the Streamlit application

```bash
streamlit run app.py
```

Use the browser interface to select a model, provide an image, inspect its reconstruction, and download the output.

## Evaluation Metrics

The project uses four metrics to evaluate the similarity between clean reference images and reconstructed images.

| Metric | Meaning | Preferred direction |
|---|---|---|
| MSE | Average squared pixel-level reconstruction error | Lower |
| MAE | Average absolute pixel-level reconstruction error | Lower |
| PSNR | Reconstruction quality expressed as a signal-to-noise ratio in decibels | Higher |
| SSIM | Structural similarity between the reference and reconstructed images | Higher |

### Mean Squared Error (MSE)

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
\]

MSE penalizes larger pixel-level reconstruction errors more strongly.

### Mean Absolute Error (MAE)

\[
MAE = \frac{1}{N}\sum_{i=1}^{N}|x_i-\hat{x}_i|
\]

MAE measures the average absolute difference between corresponding pixels.

### Peak Signal-to-Noise Ratio (PSNR)

\[
PSNR = 10\log_{10}\left(\frac{MAX^2}{MSE}\right)
\]

For normalized images, the implementation uses \(MAX=1\). Higher PSNR indicates lower reconstruction error.

### Structural Similarity Index (SSIM)

SSIM compares luminance, contrast, and structural information between two images. The implementation uses `skimage.metrics.structural_similarity` with `data_range=1.0` and reports the average score across the image batch.

Metric implementations are available in `src/metrics.py`.

## Saved Models

Five trained Keras models are included in `saved_models/`:

| File | Architecture |
|---|---|
| `undercomplete_ae_model.keras` | Undercomplete Autoencoder |
| `sparse_ae_model.keras` | Sparse Autoencoder |
| `denoising_ae_model.keras` | Denoising Autoencoder |
| `convolutional_ae_model.keras` | Convolutional Autoencoder |
| `contractive_ae_model.keras` | Contractive Autoencoder |

`training_summary.json` stores each model's training duration and recorded epoch-level history.

The model files can be loaded through TensorFlow/Keras. For example:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "saved_models/convolutional_ae_model.keras",
    compile=False
)
```

The contractive model is a custom Keras subclass registered by the `src.models` module. Import that module before loading the contractive model in a fresh Python process:

```python
import tensorflow as tf
import src.models

model = tf.keras.models.load_model(
    "saved_models/contractive_ae_model.keras",
    compile=False
)
```

## Implementation Notes

- The notebook's main comparison evaluates all five models on the 10,000-image MNIST test set.
- The main comparison table is taken from the notebook's recorded output. Training times and metrics can vary with hardware, runtime, package versions, and random initialization.
- The command-line training pipeline and notebook are separate execution paths. Their timings and results should not be assumed to be identical.
- The convolutional model uses BCE while the dense models use MSE-based reconstruction objectives. The reported comparison therefore reflects the configurations implemented in this project, not a controlled comparison with an identical loss function.
- The application expects the standard MNIST digit convention: white foreground strokes on a black background.
- The uploaded-image workflow resizes images to 28 × 28. Arbitrary photographs or complex scenes are outside the intended scope of these MNIST-focused models.
- The `reports/` directory is generated by evaluation scripts and is not part of the supplied project archive.

## Limitations and Future Improvements

Potential extensions to this project include:

1. **Broader datasets:** Evaluate on Fashion-MNIST, EMNIST, or real-world noisy images.
2. **Additional noise models:** Test salt-and-pepper, speckle, Poisson, and mixed noise.
3. **Architecture improvements:** Explore deeper convolutional autoencoders, residual blocks, U-Net-style skip connections, and variational autoencoders.
4. **Hyperparameter optimization:** Systematically tune learning rates, batch sizes, latent dimensions, and regularization strengths.
5. **Training controls:** Add early stopping, learning-rate scheduling, and checkpoint selection based on validation performance.
6. **Reproducibility:** Centralize random seeds and record package versions, hardware details, and experiment configurations.
7. **Application reliability:** Make model-loading failures explicit and prevent untrained fallback models from being presented as usable trained models.
8. **Deployment:** Add automated tests, a reproducible environment specification, and a production deployment workflow.

## Reproducibility

For a more reproducible experiment:

- Use the same MNIST dataset version and preprocessing pipeline.
- Keep the noise factor, train/validation split, optimizer, batch size, and epoch count fixed.
- Record the Python, TensorFlow, NumPy, and scikit-image versions.
- Record the compute environment and whether training used CPU or GPU.
- Preserve the generated `training_summary.json` and evaluation outputs.
- Set and document random seeds consistently when comparing new runs.

The data-loading utility supports a seed argument for noise generation and defaults to a fixed seed in `prepare_data`. Exact reproducibility across different hardware and software environments is not guaranteed.

## Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Make your changes and add tests where appropriate.
4. Commit your changes: `git commit -m "Add: your feature"`.
5. Push the branch and open a Pull Request.

For substantial changes, describe the experiment, configuration, and evaluation results so that comparisons remain interpretable.

## Acknowledgements

- **MNIST** dataset for handwritten digit images.
- **TensorFlow and Keras** for model development and training.
- **scikit-image** for structural similarity evaluation.
- **Streamlit** for the interactive demonstration interface.

## Author

**Anubhab Samantaray**

- GitHub: [@Anubhab47677](https://github.com/Anubhab47677)
- LinkedIn: [Anubhab Samantaray](https://www.linkedin.com/in/anubhab-samantaray/)

Computer Science & Engineering — Artificial Intelligence and Machine Learning

---

<div align="center">

**If you find this project useful, consider giving the repository a star.**

*Built as an academic deep learning project on image denoising and comparative autoencoder evaluation.*

</div>
