"""
CNN Model for Brain Tumor Detection

I'm learning about CNNs (Convolutional Neural Networks) and this is my first attempt.
CNNs are supposed to be good for images because they can detect patterns like edges and shapes.

I learned that:
- Convolutional layers detect features (like edges, corners, etc.)
- Pooling layers reduce image size (makes training faster)
- Dense layers do the final classification

This is a simple CNN with 3 convolutional blocks.
I'm starting simple to understand how it works before trying more complex models.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build a CNN model.
    
    I'm using Sequential model because it's easier to understand - layers go one after another.
    
    Parameters:
    - input_shape: Image size (height, width, channels). Default is 224x224 RGB
    - num_classes: How many classes to predict (2 = tumor or no_tumor)
    """
    # Create Sequential model - this is the simplest way
    model = keras.Sequential()
    
    # First convolutional block
    # I'm using 32 filters to start - I read that starting small is good
    # 3x3 is a common filter size for images
    # ReLU activation is standard - I learned it helps with training
    # padding='same' keeps the image size same after convolution
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    
    # MaxPooling reduces image size by taking max value in 2x2 blocks
    # This makes training faster and helps detect features at different scales
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional block
    # I'm increasing filters to 64 - more complex features need more filters
    # I read that increasing filters as you go deeper is a common pattern
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Third convolutional block
    # Even more filters (128) for even more complex patterns
    # By now the image is much smaller due to pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten converts 2D feature maps to 1D vector
    # Dense layers need 1D input, so we have to flatten
    model.add(layers.Flatten())
    
    # Dense layer with 128 neurons
    # I chose 128 because it's a common size - not too big, not too small
    model.add(layers.Dense(128, activation='relu'))
    
    # Dropout randomly turns off 50% of neurons during training
    # This helps prevent overfitting (memorizing training data)
    # I learned this is important for generalization
    model.add(layers.Dropout(0.5))
    
    # Output layer - one neuron per class
    # Softmax converts outputs to probabilities (they add up to 1.0)
    # This makes it easy to see which class the model thinks is most likely
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
