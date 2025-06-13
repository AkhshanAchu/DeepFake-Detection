# 🎭 Deepfake Detection System

<div align="center">

![Deepfake Detection](https://img.shields.io/badge/AI-Deepfake%20Detection-red?style=for-the-badge&logo=artificial-intelligence)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)


</div>

---

## 🌟 What's This About?

Welcome to our **Deepfake Detection System** - a powerful AI model that can distinguish between real and AI-generated fake images! 🕵️‍♂️ Built for the **SP Cup Competition 2024**, this model achieved outstanding performance across multiple datasets.

## 📊 Performance Highlights

<div align="center">

| 🎯 Metric | 📈 Training | 🔍 Validation | 🧪 Testing (DeepWild) |
|-----------|-------------|---------------|----------------------|
| **Accuracy** | `98.49%` 🔥 | `97.94%` ✨ | `91.20%` 💪 |

</div>

## 🎪 Training Dataset

Our model was trained on a diverse combination of datasets:
- 🏆 **SP Cup Competition 2024 Dataset** - Competition-grade deepfake samples
- 🌟 **CelebHQ Dataset** - High-quality celebrity images
- 🎭 **Tested on DeepWild Fake Dataset** - Real-world deepfake evaluation

## 🗂️ Project Structure

```
📁 deepfake-detection/
├── 🐍 train_script.py          # Main training script
├── 🔍 validate.py              # Validation & testing script
├── 🧠 models.py                # Model architecture
├── 🔧 utils.py                 # Utility functions
├── 💾 best_model.pth           # Trained model weights
├── 📊 validation_confusion_matrix.png
└── 📚 README.md                # You're here! 👋
```

## 🚀 Quick Start

### 1️⃣ Installation

First, make sure you have Python 3.8+ installed, then install the required packages:

```bash
pip install torch torchvision tqdm matplotlib seaborn scikit-learn numpy
```

### 2️⃣ Training the Model

```bash
python train_script.py
```

**What happens during training:**
- 🔄 Loads and preprocesses your dataset
- 🧠 Initializes the classifier model
- 📈 Trains for 100 epochs with early stopping
- 💾 Saves the best model based on validation accuracy
- 📊 Generates training plots and metrics

### 3️⃣ Validating the Model

```bash
python validate.py
```

**Validation features:**
- 🎯 Comprehensive accuracy metrics
- 📊 Confusion matrix visualization
- 📋 Detailed classification report
- 🎭 Per-class performance analysis

## 🔧 Model Architecture

Our model uses a **custom classifier block** architecture:
- 🏗️ Modular design with configurable blocks
- ⚡ Optimized for deepfake detection
- 🎯 Binary classification (Real vs Fake)
- 🧮 Parameter count: *Displayed during training*

## 📈 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| 🎯 **Batch Size** | `32` | Images per training batch |
| 📚 **Learning Rate** | `1e-4` | Adam optimizer learning rate |
| 🔄 **Epochs** | `100` | Maximum training epochs |
| 🧠 **Model Blocks** | `1` | Number of classifier blocks |
| 📊 **Max Samples** | `100,000` | Maximum training samples |

## 🎨 Features

- ✅ **High Accuracy**: 98.49% training accuracy
- 🔄 **Real-time Training**: Progress bars and live metrics
- 📊 **Comprehensive Validation**: Detailed performance analysis
- 💾 **Model Persistence**: Automatic best model saving
- 📈 **Visualization**: Training curves and confusion matrices
- 🎯 **Cross-dataset Testing**: Robust evaluation on multiple datasets

## 🎭 Usage Examples

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

## 📊 Results Visualization

The validation script automatically generates:
- 🎨 **Confusion Matrix**: Visual representation of predictions
- 📈 **Performance Metrics**: Precision, Recall, F1-Score
- 🎯 **Class-wise Accuracy**: Real vs Fake detection rates


## 🛠️ Customization

### Modify Training Parameters:
Edit `train_script.py` to adjust:
- 🎚️ Learning rate
- 📦 Batch size
- 🔄 Number of epochs
- 🧠 Model architecture

### Add New Datasets:
Update the data paths in both training and validation scripts to point to your datasets.

---

<div align="center">

**🎭 Happy Deepfake Detecting! 🕵️‍♂️**

*Made with ❤️ NiceGuy*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername/deepfake-detection)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

</div>
