"""
Inception Model for Brain Tumor Detection

I learned about Inception networks from research papers.
The idea is to use parallel convolutions at different scales to detect features of different sizes.

The Inception block has 4 branches running in parallel:
1. 1x1 convolution (small features)
2. 1x1 then 3x3 convolution (medium features)
3. 1x1 then 5x5 convolution (large features)
4. MaxPooling then 1x1 convolution (pooled features)

Then all branches are concatenated together.

I'm implementing a simplified version - the original Inception is more complex!
This is my attempt to understand the concept.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def inception_block(x, filters):
    """
    Create an Inception block.
    
    This is the key idea - parallel convolutions at different scales.
    I'm using Functional API because we need to concatenate branches.
    
    Parameters:
    - x: Input tensor
    - filters: Number of filters for each branch
    
    Returns:
    - Concatenated output from all branches
    """
    # Branch 1: Just 1x1 convolution
    # 1x1 conv is like a "bottleneck" - reduces dimensions cheaply
    branch1 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x)
    
    # Branch 2: 1x1 then 3x3 convolution
    # The 1x1 conv reduces dimensions before the 3x3 conv
    # This is more efficient than just 3x3
    branch2 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x)
    branch2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(branch2)
    
    # Branch 3: 1x1 then 5x5 convolution
    # Similar to branch 2, but with larger kernel for bigger features
    branch3 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x)
    branch3 = layers.Conv2D(filters, (5, 5), activation='relu', padding='same')(branch3)
    
    # Branch 4: MaxPooling then 1x1 convolution
    # Pooling helps detect features at different scales
    branch4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch4 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(branch4)
    
    # Concatenate all branches
    # This combines features from all scales - this is the key idea!
    # All branches run in parallel, then we combine their outputs
    output = layers.Concatenate()([branch1, branch2, branch3, branch4])
    
    return output


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build an Inception-based model.
    
    I'm using multiple Inception blocks stacked together.
    This should help detect features at multiple scales throughout the network.
    
    Parameters:
    - input_shape: Image size
    - num_classes: Number of classes
    """
    # Use Functional API for Inception blocks
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution and pooling
    # I'm starting with a regular conv layer before Inception blocks
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # First Inception block
    # I'm using 32 filters per branch - starting small
    x = inception_block(x, filters=32)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second Inception block
    # Increasing filters to 64 - more complex features
    x = inception_block(x, filters=64)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third Inception block
    # Keeping at 64 filters - not increasing too much
    x = inception_block(x, filters=64)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Global Average Pooling
    # Instead of flattening, I average each feature map
    # This reduces parameters and helps with overfitting
    # I learned this from reading about modern CNN architectures
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
