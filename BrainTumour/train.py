"""
Training Script for Brain Tumor Detection

I'm learning deep learning and this is my training script.
I'm experimenting with different models to see which works best.

I learned that:
- Data augmentation helps when you don't have much data
- Early stopping prevents overfitting
- Learning rate reduction helps fine-tuning
- Saving the best model is important

Dataset should be organized like this:
    dataset/
        tumor/
            image1.jpg
            image2.jpg
        no_tumor/
            image1.jpg
            image2.jpg

Usage example:
    python train.py --model cnn --epochs 20
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import the model building functions
from models.cnn import build_model as build_cnn
from models.dnn import build_model as build_dnn
from models.lstm import build_model as build_lstm
from models.inception import build_model as build_inception
from models.attention_cnn import build_model as build_attention_cnn


def load_data(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Load images from the dataset folder.
    
    I learned about data augmentation from tutorials - it's really helpful!
    By creating variations of images, the model sees more examples and learns better.
    
    This function reads images from tumor/ and no_tumor/ folders and prepares them for training.
    It also splits the data into training and validation sets.
    """
    # Create data generator for training with augmentation
    # I learned that augmentation is super important when you don't have much data
    # It creates variations of images so the model sees more examples
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Convert pixel values from 0-255 to 0-1 (normalization)
        rotation_range=20,  # Rotate images up to 20 degrees - tumors can be at any angle
        width_shift_range=0.2,  # Shift images horizontally - position shouldn't matter
        height_shift_range=0.2,  # Shift images vertically
        horizontal_flip=True,  # Flip images horizontally - brain is symmetric
        zoom_range=0.2,  # Zoom in/out - tumors can be different sizes
        fill_mode='nearest',  # Fill empty spaces with nearest pixels
        validation_split=validation_split  # Use 20% for validation (I learned 80/20 split is common)
    )
    
    # Create data generator for validation (no augmentation, just rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Just normalize pixel values
        validation_split=validation_split
    )
    
    # Load training images
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,  # Resize all images to this size
        batch_size=batch_size,  # Number of images per batch
        class_mode='categorical',  # For multi-class classification
        subset='training',  # Use training subset
        shuffle=True,  # Shuffle the data
        seed=42  # For reproducibility
    )
    
    # Load validation images
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',  # Use validation subset
        shuffle=False,  # Don't shuffle validation data
        seed=42
    )
    
    # Get number of classes and class names
    num_classes = len(train_generator.class_indices)
    class_indices = train_generator.class_indices
    
    # Print dataset information
    print("\n" + "="*60)
    print("Dataset Information:")
    print("="*60)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {class_indices}")
    print("="*60 + "\n")
    
    return train_generator, validation_generator, num_classes, class_indices


def train_model(model_name, data_dir, img_size=(224, 224), epochs=20, batch_size=32, 
                learning_rate=0.001, validation_split=0.2):
    """
    Train a model on the brain tumor dataset.
    
    This function:
    1. Loads the data
    2. Builds the model
    3. Compiles the model
    4. Trains the model
    5. Saves the model
    """
    # Dictionary to map model names to their building functions
    model_builders = {
        'cnn': build_cnn,
        'dnn': build_dnn,
        'lstm': build_lstm,
        'inception': build_inception,
        'attention_cnn': build_attention_cnn
    }
    
    # Check if model name is valid
    if model_name not in model_builders:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(model_builders.keys())}")
        return None, None
    
    print("\n" + "="*60)
    print(f"Training {model_name.upper()} Model")
    print("="*60 + "\n")
    
    # Step 1: Load the data
    print("Loading data...")
    train_gen, val_gen, num_classes, class_indices = load_data(
        data_dir, img_size, batch_size, validation_split
    )
    
    # Step 2: Build the model
    print("Building model...")
    # Input shape is (height, width, channels) - channels=3 for RGB images
    input_shape = (img_size[0], img_size[1], 3)
    build_function = model_builders[model_name]
    model = build_function(input_shape=input_shape, num_classes=num_classes)
    
    # Step 3: Compile the model
    print("Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),  # Adam optimizer
        loss='categorical_crossentropy',  # Loss function for classification
        metrics=['accuracy']  # Track accuracy during training
    )
    
    # Print model summary
    print("\nModel Architecture:")
    print("-" * 60)
    model.summary()
    print("-" * 60)
    
    # Calculate steps per epoch
    # This tells the model how many batches to process per epoch
    steps_per_epoch = train_gen.samples // batch_size
    validation_steps = val_gen.samples // batch_size
    
    # Create directory to save models
    save_dir = 'models/saved'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Set up callbacks
    # Callbacks are functions that run during training - I learned these are really useful!
    callbacks_list = []
    
    # Early stopping: stop training if validation loss doesn't improve
    # I learned this prevents overfitting - when model stops improving, stop training
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Watch validation loss (not training loss!)
        patience=5,  # Wait 5 epochs before stopping - give it a chance
        restore_best_weights=True,  # Use best weights (not the last ones)
        verbose=1  # Print messages
    )
    callbacks_list.append(early_stopping)
    
    # Reduce learning rate if validation loss plateaus
    # I learned that reducing learning rate helps fine-tuning
    # When loss stops decreasing, maybe we need smaller steps
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce learning rate by half (make smaller steps)
        patience=3,  # Wait 3 epochs before reducing
        min_lr=1e-7,  # Don't go below this learning rate
        verbose=1
    )
    callbacks_list.append(reduce_lr)
    
    # Save best model during training
    # I learned to always save the best model, not just the last one!
    checkpoint = ModelCheckpoint(
        filepath=f'models/saved/{model_name}_best.h5',
        monitor='val_accuracy',  # Save when validation accuracy improves
        save_best_only=True,  # Only save best model (saves disk space)
        verbose=1
    )
    callbacks_list.append(checkpoint)
    
    # Step 4: Train the model
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}\n")
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Step 5: Print final results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Get final metrics from training history
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print("="*60 + "\n")
    
    # Step 6: Save final model
    final_model_path = f"models/saved/{model_name}_final.h5"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Generate evaluation plots if possible
    # I learned that visualizing results is really important for understanding model performance
    print("\nGenerating evaluation plots...")
    try:
        # Import evaluation functions
        # I created these functions to help me understand how well my models are doing
        from evaluate import plot_training_history, plot_confusion_matrix, plot_roc_curve
        
        # Prepare validation data for evaluation
        val_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=validation_split
        )
        
        val_gen_eval = val_datagen.flow_from_directory(
            data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Get predictions
        print("Generating predictions...")
        y_pred_proba = model.predict(val_gen_eval, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = val_gen_eval.classes
        class_names = list(val_gen_eval.class_indices.keys())
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, class_names, model_name)
        
        # Plot ROC curve
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
        plot_roc_curve(y_true_onehot, y_pred_proba, class_names, model_name)
        
        print("Evaluation plots generated successfully!")
        
    except ImportError:
        print("Warning: Could not import evaluation functions.")
        print("You can run evaluate.py separately to generate plots.")
    except Exception as e:
        print(f"Warning: Could not generate evaluation plots: {e}")
        print("You can run evaluate.py separately to generate plots.")
    
    return model, history


def main():
    """Main function that runs when script is executed."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Train brain tumor detection models')
    
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'dnn', 'lstm', 'inception', 'attention_cnn'],
                        help='Which model to train')
    
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset folder')
    
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Image size (height width)')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data for validation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        print("\nExpected structure:")
        print("  dataset/")
        print("    ├── tumor/")
        print("    │   ├── image1.jpg")
        print("    │   └── ...")
        print("    └── no_tumor/")
        print("        ├── image1.jpg")
        print("        └── ...")
        return
    
    # Check if tumor and no_tumor folders exist
    tumor_dir = os.path.join(args.data_dir, 'tumor')
    no_tumor_dir = os.path.join(args.data_dir, 'no_tumor')
    
    if not os.path.exists(tumor_dir):
        print(f"Error: '{tumor_dir}' folder not found.")
        return
    
    if not os.path.exists(no_tumor_dir):
        print(f"Error: '{no_tumor_dir}' folder not found.")
        return
    
    # Train the model
    try:
        train_model(
            model_name=args.model,
            data_dir=args.data_dir,
            img_size=tuple(args.img_size),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
