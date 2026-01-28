"""
Evaluation Script for Brain Tumor Detection Models

I'm learning how to evaluate models properly.
This script helps me understand how well my models are performing.

It creates plots:
- Confusion Matrix: Shows correct and incorrect predictions (I learned this is really useful!)
- Training History: Shows accuracy and loss over time (helps spot overfitting)
- ROC Curve: Shows how well the model separates classes (AUC score tells me how good it is)

I learned that just looking at accuracy isn't enough - you need to understand WHERE the model makes mistakes.

Usage:
    python evaluate.py --model cnn --data_dir dataset
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Set plot style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def load_model_and_data(model_path, data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Load a trained model and prepare data for evaluation.
    
    Parameters:
    - model_path: Path to saved model file
    - data_dir: Path to dataset folder
    - img_size: Image size
    - batch_size: Batch size
    - validation_split: Validation split ratio
    """
    # Load the saved model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Prepare data generator (no augmentation, just rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Normalize to 0-1
        validation_split=validation_split
    )
    
    # Load validation images
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,  # Don't shuffle for evaluation
        seed=42
    )
    
    # Get number of classes and class names
    num_classes = len(val_generator.class_indices)
    class_names = list(val_generator.class_indices.keys())
    
    return model, val_generator, num_classes, class_names


def plot_training_history(history, model_name, save_dir='models/evaluations'):
    """
    Plot training and validation accuracy/loss over epochs.
    
    Parameters:
    - history: Training history from model.fit()
    - model_name: Name of the model
    - save_dir: Where to save the plot
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title(f'{model_name.upper()} - Accuracy Curves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title(f'{model_name.upper()} - Loss Curves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir='models/evaluations'):
    """
    Plot confusion matrix.
    
    I learned about confusion matrices from tutorials - they're super helpful!
    They show:
    - How many correct predictions (diagonal)
    - How many wrong predictions (off-diagonal)
    - Which classes the model confuses with each other
    
    This helps me understand WHERE my model makes mistakes, not just HOW MANY mistakes.
    
    Parameters:
    - y_true: True labels (what the answer should be)
    - y_pred: Predicted labels (what my model predicted)
    - class_names: Names of classes (like 'tumor', 'no_tumor')
    - model_name: Name of the model
    - save_dir: Where to save the plot
    """
    # Create directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages for each cell
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,  # Show numbers in cells
        fmt='d',  # Format as integers
        cmap='Blues',  # Color scheme
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(
                j + 0.5, i + 0.7,
                f'({cm_percent[i, j]:.1f}%)',
                ha="center", va="center",
                color="red", fontsize=10, fontweight='bold'
            )
    
    # Set labels and title
    ax.set_title(f'{model_name.upper()} - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_pred_proba, class_names, model_name, save_dir='models/evaluations'):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    I learned about ROC curves from reading about model evaluation.
    They show the trade-off between:
    - True Positive Rate (how many tumors we correctly find)
    - False Positive Rate (how many non-tumors we incorrectly call tumors)
    
    AUC (Area Under Curve) score tells me how good the model is:
    - 1.0 = perfect
    - 0.5 = random guessing
    - Higher is better!
    
    Parameters:
    - y_true: True labels (one-hot encoded)
    - y_pred_proba: Predicted probabilities (not just class predictions)
    - class_names: Names of classes
    - model_name: Name of the model
    - save_dir: Where to save the plot
    """
    # Create directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert one-hot to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    
    if n_classes == 2:
        # Binary classification (tumor vs no_tumor)
        # Calculate ROC curve for positive class (tumor)
        fpr, tpr, _ = roc_curve(y_true_classes, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
        
    else:
        # Multi-class: plot one curve per class
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
        
        for i, class_name in enumerate(class_names):
            # Create binary labels for this class
            y_true_binary = (y_true_classes == i).astype(int)
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, linewidth=2, color=colors[i],
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title(f'{model_name.upper()} - ROC Curve', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()


def evaluate_model(model_path, data_dir, model_name, img_size=(224, 224), 
                  batch_size=32, validation_split=0.2, history=None):
    """
    Evaluate a model and generate all plots.
    
    Parameters:
    - model_path: Path to saved model
    - data_dir: Path to dataset
    - model_name: Name of the model
    - img_size: Image size
    - batch_size: Batch size
    - validation_split: Validation split
    - history: Training history (optional)
    """
    print("\n" + "="*60)
    print(f"Evaluating {model_name.upper()} Model")
    print("="*60 + "\n")
    
    # Load model and data
    model, val_generator, num_classes, class_names = load_model_and_data(
        model_path, data_dir, img_size, batch_size, validation_split
    )
    
    # Get predictions
    print("Generating predictions...")
    y_pred_proba = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get true labels
    y_true = val_generator.classes
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
    
    # Plot training history if available
    if history is not None:
        print("\nPlotting training history...")
        plot_training_history(history, model_name)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    print("-" * 60)
    
    # Calculate overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Plot ROC curve
    print("\nGenerating ROC curve...")
    plot_roc_curve(y_true_onehot, y_pred_proba, class_names, model_name)
    
    # Calculate AUC scores
    if num_classes == 2:
        auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        print(f"\nROC AUC Score: {auc_score:.4f}")
    else:
        auc_scores = []
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            auc_score = roc_auc_score(y_true_binary, y_pred_proba[:, i])
            auc_scores.append(auc_score)
            print(f"ROC AUC Score ({class_names[i]}): {auc_score:.4f}")
    
    print("\n" + "="*60)
    print(f"Evaluation Complete for {model_name.upper()}")
    print("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate brain tumor detection models')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'dnn', 'lstm', 'inception', 'attention_cnn'],
                        help='Which model to evaluate')
    
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset folder')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model file (default: models/saved/{model}_best.h5)')
    
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Image size (height width)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data used for validation')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path is None:
        model_path = f'models/saved/{args.model}_best.h5'
    else:
        model_path = args.model_path
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first using train.py")
        return
    
    # Evaluate model
    try:
        evaluate_model(
            model_path=model_path,
            data_dir=args.data_dir,
            model_name=args.model,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            validation_split=args.validation_split
        )
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
