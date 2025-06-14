#  Deepfake Detection System

<div align="center">

![Deepfake Detection](https://img.shields.io/badge/AI-Deepfake%20Detection-red?style=for-the-badge&logo=artificial-intelligence)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

</div>

---

##  What's This About?
Deep Architecture that combines Vision Transformers with Cross-Modal Fusion to distinguish between real and AI-generated fake images! 🕵️‍♂️ Built for the **SP Cup Competition 2024**, this hybrid architecture achieved outstanding performance across multiple datasets.

##  Model Architecture
Approach combines techniques:

###  Core Components
- **MViT (Multiscale Vision Transformer)**: Custom MViT transformer blocks with multi-head attention and Scales
- **CMF (Cross-Modal Fusion) Block**: Integrates RGB, frequency spectrum, and texture features
- **ConvNeXT Feature Extractor**: Pre-trained backbone for robust feature extraction
- **Multi-Modal Analysis**: Processes RGB images, Fourier spectrum, and Local Binary Patterns (LBP)

### Architecture Details
```
Input Image (224x224x3)
    ↓
┌─────────────────────────────────────┐
│     MViT-CMF Repeated Blocks        │
│  ┌─────────────┐  ┌─────────────┐   │
│  │ MViT Block  │→ │ CMF Block   │   │
│  │ - Patch     │  │ - RGB       │   │
│  │ - Attention │  │ - Spectrum  │   │
│  │ - Transform │  │ - LBP       │   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│        Feature Fusion               │
│  ┌─────────────┐  ┌─────────────┐   │
│  │ MViT-CMF    │  │ ConvNeXT    │   │
│  │ Features    │→ │ Features    │   │
│  │ (30,000)    │  │ (1,024)     │   │
│  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────┘
    ↓
FC Layers (31,024 → 1,000 → 128 → 2)
    ↓
Real/Fake Classification
```

## Performance Highlights

<div align="center">

| Metric | Training | Validation | Testing (DeepWild) |
|-----------|-------------|---------------|----------------------|
| **Accuracy** | `98.49%` | `97.94%` | `91.20%` |

</div>

## Training Dataset

The model was trained on a diverse combination of datasets:
- **SP Cup Competition 2024 Dataset** - Competition-grade deepfake samples
- **CelebHQ Dataset** - High-quality celebrity images
- **Tested on DeepWild Fake Dataset** - Real-world deepfake evaluation

## Project Structure

```
📁 deepfake-detection/
├──  train_script.py          # Main training script
├──  validate.py              # Validation & testing script
├──  models.py                # MViT-CMF model architecture
├──  utils.py                 # Utility functions
├──  best_model.pth           # Trained model weights
├──  validation_confusion_matrix.png
└──  README.md                # You're here! 👋
```

##  Quick Start

###  Installation

First, make sure you have Python 3.8+ installed, then install the required packages:

```bash
pip install torch torchvision tqdm matplotlib seaborn scikit-learn numpy opencv-python scikit-image
```

###  Training the Model

```bash
python train_script.py
```

**What happens during training:**
-  Loads and preprocesses your dataset
-  Initializes the MViT-CMF classifier model
-  Trains for 100 epochs with early stopping
-  Saves the best model based on validation accuracy
-  Generates training plots and metrics

###  Validating the Model

```bash
python validate.py
```

**Validation features:**
-  Comprehensive accuracy metrics
-  Confusion matrix visualization
-  Detailed classification report
-  Per-class performance analysis

##  Model Architecture Details

###  MViT Block Features:
- **Patch Embedding**: Converts 224×224 images into 16×16 patches
- **Multi-head Attention**: 8-head self-attention mechanism
- **Positional Encoding**: Spatial position information preservation
- **Residual Connections**: Skip connections for gradient flow

###  CMF Block Features:
- **Multi-modal Input**: RGB + Frequency Spectrum + LBP textures
- **Fourier Analysis**: 2D FFT for frequency domain analysis
- **Local Binary Patterns**: Texture feature extraction
- **Cross-Modal Attention**: Fusion of different feature modalities

###  Feature Extraction:
- **ConvNeXT Backbone**: Pre-trained feature extractor (1,024 features)
- **MViT-CMF Features**: Custom features (30,000 dimensions)
- **Feature Fusion**: Concatenated multi-scale representations

## 📈 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
|  **Batch Size** | `32` | Images per training batch |
|  **Learning Rate** | `1e-4` | Adam optimizer learning rate |
|  **Epochs** | `100` | Maximum training epochs |
|  **MViT-CMF Blocks** | `6` | Number of transformer-fusion blocks |
|  **Max Samples** | `100,000` | Maximum training samples |
|  **Patch Size** | `16×16` | Vision transformer patch size |
|  **Embedding Dim** | `256` | Transformer embedding dimension |

##  Key Features
-  **Hybrid Architecture**: Combines transformers with frequency analysis
-  **Multi-Modal Processing**: RGB, spectrum, and texture analysis
-  **Advanced Feature Fusion**: ConvNeXT + MViT-CMF integration
-  **Robust Training**: Early stopping and model persistence
-  **Comprehensive Metrics**: Detailed performance visualization
-  **Cross-Dataset Validation**: Tested on multiple deepfake datasets
-  **Attention Mechanisms**: Self-attention for long-range dependencies

##  Usage Examples

### Basic Validation
```python
from validate import validate_model

# Validate on default test set
results = validate_model('best_model.pth')
print(f"Accuracy: {results['accuracy']:.2f}%")
```

### Custom Dataset Validation
```python
from validate import validate_on_custom_dataset

# Test on your own dataset
results = validate_on_custom_dataset(
    model_path='best_model.pth',
    custom_test_path='path/to/your/test/data'
)
```

### Model Architecture Info
```python
from models import classifier_block

# Initialize model
model = classifier_block(num_classes=2, n_blocks=6)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
```

## 📊 Results Visualization

The validation script automatically generates:
- 🎨 **Confusion Matrix**: Visual representation of predictions
- 📈 **Performance Metrics**: Precision, Recall, F1-Score
- 🎯 **Class-wise Accuracy**: Real vs Fake detection rates
- 🌊 **Feature Analysis**: Multi-modal feature importance

## 🛠️ Customization

### Modify Model Architecture:
Edit `models.py` to adjust:
-  Number of MViT-CMF blocks (`n_blocks`)
-  Transformer embedding dimensions
-  CMF block configurations
-  Feature fusion strategies

### Training Parameters:
Edit `train_script.py` to adjust:
-  Learning rate and optimizer settings
-  Batch size and data loading
-  Number of epochs and early stopping
-  Model saving strategies

### Add New Datasets:
Update the data paths in both training and validation scripts to point to your datasets.

---

<div align="center">

**  Happy Deepfake Detecting!  **

*Made with ❤️  from NiceGuy*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername/deepfake-detection)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

</div>
