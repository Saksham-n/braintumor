# Brain Tumor Detection: Deep Learning Architecture Comparison

## Project Overview

This project implements and compares five different deep learning architectures for brain tumor detection using MRI images. The goal is to provide a comprehensive comparative analysis of various neural network architectures for medical image classification tasks.

**Dataset**: Brain Tumor MRI images with two classes:
- `tumor`: MRI images containing brain tumors
- `no_tumor`: MRI images without tumors

## Project Structure

```
backend/
├── models/
│   ├── __init__.py          # Model package initialization
│   ├── cnn.py               # Convolutional Neural Network
│   ├── dnn.py               # Deep Neural Network (baseline)
│   ├── lstm.py              # CNN-LSTM hybrid model
│   ├── inception.py         # Inception-based CNN
│   ├── attention_cnn.py     # Attention-based CNN
│   ├── saved/               # Saved model weights
│   └── evaluations/         # Evaluation plots and metrics
├── train.py                 # Training script with CLI
├── evaluate.py              # Individual model evaluation script
├── compare_all_models.py    # Compare all models script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Models Implemented

### 1. CNN (Convolutional Neural Network)
**File**: `models/cnn.py`

A standard CNN architecture with:
- Three convolutional blocks (32, 64, 128 filters)
- MaxPooling layers for dimensionality reduction
- Dense layers for classification

**Why CNN**: CNNs are the standard architecture for image classification, preserving spatial relationships and learning hierarchical features.

### 2. DNN (Deep Neural Network)
**File**: `models/dnn.py`

A fully connected neural network baseline:
- Flattens input image (loses spatial structure)
- Multiple dense layers with dropout
- Serves as comparison baseline

**Why DNN**: Demonstrates the importance of preserving spatial structure. Included to show why CNNs outperform fully connected networks on images.

### 3. LSTM (CNN-LSTM Hybrid)
**File**: `models/lstm.py`

A hybrid architecture combining:
- CNN feature extractor
- LSTM layers for sequence processing
- Dense layers for classification

**Why LSTM**: Explores whether sequential modeling can capture additional patterns in medical images beyond spatial convolutions.

### 4. Inception-based CNN
**File**: `models/inception.py`

Multi-scale feature extraction using:
- Parallel convolutions at different scales (1x1, 3x3, 5x5)
- Inception blocks for multi-resolution features
- Global average pooling

**Why Inception**: Multi-scale features help detect tumors of various sizes, which is crucial for medical imaging.

### 5. Attention-based CNN
**File**: `models/attention_cnn.py`

CNN enhanced with channel-wise attention:
- Squeeze-and-Excitation style attention
- Focuses on relevant feature channels
- Improves interpretability

**Why Attention**: Attention mechanisms help the model focus on tumor-relevant regions while suppressing background noise.

## Dataset

### Expected Structure

```
dataset/
├── tumor/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── no_tumor/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Dataset Details

- **Image Size**: 224x224 RGB (can be configured)
- **Normalization**: Pixel values normalized to [0, 1]
- **Augmentation**: Training data includes rotation, shifts, flips, and zoom
- **Split**: 80% training, 20% validation (configurable)

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Organize your dataset**:
   - Place MRI images in `dataset/tumor/` and `dataset/no_tumor/` folders
   - Ensure images are in common formats (jpg, png, etc.)

## Usage

### Basic Training

Train a single model:
```bash
python train.py --model cnn --epochs 20
```

### Training All Models for Comparison

```bash
# CNN
python train.py --model cnn --epochs 20

# DNN
python train.py --model dnn --epochs 20

# LSTM
python train.py --model lstm --epochs 20

# Inception
python train.py --model inception --epochs 20

# Attention CNN
python train.py --model attention_cnn --epochs 20
```

### Advanced Options

```bash
python train.py \
    --model cnn \
    --epochs 30 \
    --batch_size 16 \
    --img_size 224 224 \
    --learning_rate 0.0001 \
    --validation_split 0.2
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture (`cnn`, `dnn`, `lstm`, `inception`, `attention_cnn`) | `cnn` |
| `--data_dir` | Path to dataset directory | `dataset` |
| `--epochs` | Number of training epochs | `20` |
| `--batch_size` | Batch size for training | `32` |
| `--img_size` | Image size (height width) | `224 224` |
| `--learning_rate` | Learning rate for Adam optimizer | `0.001` |
| `--validation_split` | Fraction of data for validation | `0.2` |

## Training Features

- **Data Augmentation**: Rotation, shifts, flips, and zoom for training data
- **Early Stopping**: Stops training if validation loss doesn't improve
- **Learning Rate Reduction**: Automatically reduces learning rate on plateau
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Progress Tracking**: Real-time training metrics

## Model Outputs

Trained models are saved in `models/saved/`:
- `{model_name}_best.h5`: Best model based on validation accuracy
- `{model_name}_final.h5`: Final model after all epochs

## Evaluation and Visualization

### Automatic Evaluation (During Training)

When you train a model, evaluation plots are automatically generated:
- **Training History**: Accuracy and loss curves for training and validation
- **Confusion Matrix**: Classification performance breakdown
- **ROC Curve**: Receiver Operating Characteristic curve with AUC score

Plots are saved in `models/evaluations/` directory.

### Individual Model Evaluation

Evaluate a specific trained model:

```bash
python evaluate.py --model cnn --data_dir dataset
```

This generates:
- Confusion matrix with percentages
- Training/validation curves (if history available)
- ROC curve with AUC score
- Classification report (precision, recall, F1-score)

### Compare All Models

Generate comprehensive comparison of all trained models:

```bash
python compare_all_models.py --data_dir dataset
```

This creates:
- **Side-by-side confusion matrices** for all models
- **ROC curve comparison** showing all models on one plot
- **Metrics comparison table** (Accuracy and AUC scores)
- **CSV file** with numerical metrics for further analysis

Output files:
- `models/evaluations/all_models_confusion_matrices.png`
- `models/evaluations/all_models_roc_comparison.png`
- `models/evaluations/all_models_metrics_comparison.png`
- `models/evaluations/model_comparison_metrics.csv`

## Results Comparison

After training all models, compare:
- **Training Accuracy**: How well models fit training data
- **Validation Accuracy**: Generalization performance
- **Training Loss**: Loss on training set
- **Validation Loss**: Loss on validation set
- **Confusion Matrix**: Per-class performance breakdown
- **ROC AUC Score**: Area under ROC curve (higher is better)

### Expected Observations

1. **CNN** should outperform **DNN** due to spatial structure preservation
2. **Inception** may capture multi-scale features better than basic CNN
3. **Attention CNN** should focus on relevant regions, potentially improving performance
4. **LSTM** performance depends on whether sequential patterns exist in the data
5. **ROC curves** help visualize trade-offs between true positive and false positive rates

## Research Context

This project aligns with comparative research in medical image classification:

- **Architectural Comparison**: Evaluates different deep learning paradigms
- **Feature Extraction**: Compares spatial (CNN) vs. sequential (LSTM) approaches
- **Attention Mechanisms**: Studies the impact of attention on medical imaging
- **Multi-scale Features**: Explores Inception-style architectures for varying tumor sizes

### Key Research Questions

1. Which architecture best captures tumor features in MRI images?
2. Do attention mechanisms improve tumor detection accuracy?
3. Can multi-scale features (Inception) detect tumors of varying sizes?
4. How do spatial (CNN) vs. sequential (LSTM) approaches compare?

## Technical Details

### Model Architectures

| Model | Parameters | Key Features |
|-------|-----------|--------------|
| CNN | ~500K-1M | Standard convolutions, spatial hierarchy |
| DNN | ~50M+ | Fully connected, baseline comparison |
| LSTM | ~1M-2M | CNN features + sequential processing |
| Inception | ~2M-5M | Multi-scale parallel convolutions |
| Attention CNN | ~1M-2M | Channel-wise attention mechanism |

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Image Format**: RGB (3 channels)
- **Normalization**: [0, 1] range

## Limitations and Considerations

1. **Dataset Size**: Results depend on dataset quality and size
2. **Generalization**: Models trained on one dataset may not generalize to others
3. **Medical Accuracy**: This is a research project, not for clinical use
4. **Hyperparameters**: Default hyperparameters may need tuning for specific datasets
5. **Computational Resources**: Training times vary by model complexity

## Evaluation Features

✅ **Confusion Matrix**: Visual representation of classification performance  
✅ **ROC Curves**: Receiver Operating Characteristic curves with AUC scores  
✅ **Training History**: Accuracy and loss curves for training/validation  
✅ **Metrics Comparison**: Side-by-side comparison of all models  
✅ **Classification Reports**: Precision, recall, F1-score per class  
✅ **CSV Export**: Numerical metrics for further analysis

