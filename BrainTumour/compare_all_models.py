"""
Compare All Models Script

I'm comparing all my models to see which one works best!
This script helps me understand the differences between CNN, DNN, LSTM, Inception, and Attention models.

It creates:
- Side-by-side confusion matrices (easy to compare)
- ROC curve comparison (see which model separates classes best)
- Metrics comparison table (numbers to compare)

I learned that comparing models side-by-side is really helpful for understanding their strengths and weaknesses.

Usage:
    python compare_all_models.py --data_dir dataset
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
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


def load_model_predictions(model_path, val_generator):
    """
    Load a model and get its predictions.
    
    Parameters:
    - model_path: Path to saved model
    - val_generator: Validation data generator
    
    Returns:
    - y_pred: Predicted class labels
    - y_pred_proba: Predicted probabilities
    - model: The loaded model
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Get predictions
    y_pred_proba = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_pred, y_pred_proba, model


def compare_all_models(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Compare all trained models.
    
    I'm comparing all my models to see which architecture works best!
    This is really helpful for learning - I can see what works and what doesn't.
    
    This function:
    1. Loads all models I've trained
    2. Gets predictions from each model
    3. Calculates metrics (accuracy, AUC, etc.)
    4. Creates comparison plots so I can visualize the differences
    """
    # List of all model names
    models = ['cnn', 'dnn', 'lstm', 'inception', 'attention_cnn']
    
    # Prepare validation data
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=validation_split
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Get true labels
    y_true = val_generator.classes
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(val_generator.class_indices))
    class_names = list(val_generator.class_indices.keys())
    
    # Create evaluations directory
    eval_dir = 'models/evaluations'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    # Store results for each model
    results = {}
    predictions = {}
    probabilities = {}
    
    print("\n" + "="*60)
    print("Loading Models and Generating Predictions")
    print("="*60 + "\n")
    
    # Load all models
    for model_name in models:
        model_path = f'models/saved/{model_name}_best.h5'
        
        if os.path.exists(model_path):
            print(f"Loading {model_name.upper()}...")
            
            # Get predictions
            y_pred, y_pred_proba, model = load_model_predictions(model_path, val_generator)
            predictions[model_name] = y_pred
            probabilities[model_name] = y_pred_proba
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            
            # Calculate AUC score
            if len(class_names) == 2:
                # Binary classification
                auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                # Multi-class: average AUC for all classes
                auc_scores = []
                for i in range(len(class_names)):
                    y_true_binary = (y_true == i).astype(int)
                    auc_score = roc_auc_score(y_true_binary, y_pred_proba[:, i])
                    auc_scores.append(auc_score)
                auc_score = np.mean(auc_scores)
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'auc': auc_score,
                'confusion_matrix': cm
            }
        else:
            print(f"Warning: {model_path} not found. Skipping {model_name}.")
    
    # Check if we have any results
    if not results:
        print("No trained models found. Please train models first.")
        return
    
    # 1. Create comparison confusion matrices
    # I'm putting them side-by-side so I can easily compare
    print("\nGenerating comparison confusion matrices...")
    n_models = len(results)
    
    # Create subplots - one for each model
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    # Plot confusion matrix for each model
    # This helps me see which model makes fewer mistakes
    for idx, (model_name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,  # Show numbers
            fmt='d',  # Integer format
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx],
            cbar_kws={'label': 'Count'}
        )
        
        # Set title with accuracy
        axes[idx].set_title(f'{model_name.upper()}\nAccuracy: {result["accuracy"]:.3f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('True', fontsize=10)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(eval_dir, 'all_models_confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # 2. Create ROC curve comparison
    # I'm putting all ROC curves on one plot to compare them easily
    print("Generating ROC curve comparison...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get different colors for each model
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(results)))
    
    # Plot ROC curve for each model
    # The curve that's higher and more to the left is better!
    for idx, (model_name, result) in enumerate(results.items()):
        y_pred_proba = probabilities[model_name]
        
        if len(class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, color=colors[idx],
                   label=f'{model_name.upper()} (AUC = {roc_auc:.3f})')
        else:
            # Multi-class: average ROC curves
            fprs = []
            tprs = []
            for i in range(len(class_names)):
                y_true_binary = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
                fprs.append(fpr)
                tprs.append(tpr)
            
            # Interpolate to common thresholds
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(mean_fpr)
            for i in range(len(class_names)):
                mean_tpr += np.interp(mean_fpr, fprs[i], tprs[i])
            mean_tpr /= len(class_names)
            roc_auc = auc(mean_fpr, mean_tpr)
            
            ax.plot(mean_fpr, mean_tpr, linewidth=2, color=colors[idx],
                   label=f'{model_name.upper()} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve Comparison - All Models', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(eval_dir, 'all_models_roc_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # 3. Create metrics comparison table
    # This gives me a quick summary of which model is best
    print("Generating metrics comparison table...")
    
    # Create DataFrame with results
    # I'm collecting accuracy and AUC scores for each model
    model_names = [m.upper() for m in results.keys()]
    accuracies = [results[m]['accuracy'] for m in results.keys()]
    auc_scores = [results[m]['auc'] for m in results.keys()]
    
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'AUC Score': auc_scores
    })
    
    # Sort by accuracy (best first)
    # This makes it easy to see which model performed best
    metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.35, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color code rows by accuracy
    for i in range(1, len(metrics_df) + 1):
        accuracy = metrics_df.iloc[i-1]['Accuracy']
        if accuracy >= 0.9:
            color = '#90EE90'  # Light green
        elif accuracy >= 0.8:
            color = '#FFE4B5'  # Moccasin
        else:
            color = '#FFB6C1'  # Light pink
        
        # Color all cells in the row
        table[(i, 0)].set_facecolor(color)
        table[(i, 1)].set_facecolor(color)
        table[(i, 2)].set_facecolor(color)
    
    # Set title
    ax.set_title('Model Comparison - Accuracy and AUC Scores', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Save plot
    save_path = os.path.join(eval_dir, 'all_models_metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(metrics_df.to_string(index=False))
    print("="*60 + "\n")
    
    # Save CSV file
    csv_path = os.path.join(eval_dir, 'model_comparison_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare all trained models')
    
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset folder')
    
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Image size (height width)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data used for validation')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return
    
    # Compare models
    try:
        compare_all_models(
            data_dir=args.data_dir,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            validation_split=args.validation_split
        )
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
