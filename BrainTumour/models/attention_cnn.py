"""
Attention-based CNN Model for Brain Tumor Detection

I learned about attention mechanisms from reading research papers.
Attention helps models focus on important parts of the input.

For images, I'm using channel-wise attention (Squeeze-and-Excitation style):
1. Squeeze: Average all pixels in each feature map (get one number per feature map)
2. Excitation: Learn which feature maps are important
3. Scale: Multiply original features by importance weights

The idea is that some feature maps are more important than others.
Attention learns to emphasize important ones and suppress less important ones.

This is my attempt to implement attention - I'm still learning how it works!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def channel_attention_block(x, reduction=16):
    """
    Create a channel attention block.
    
    This implements Squeeze-and-Excitation attention.
    I learned about this from reading papers - it's supposed to help models focus.
    
    Parameters:
    - x: Input feature map
    - reduction: How much to reduce dimensions (default 16)
                 Smaller number = more parameters but more capacity
    
    Returns:
    - Attention-weighted feature map
    """
    # Get number of channels (feature maps)
    channels = x.shape[-1]
    
    # Step 1: Squeeze
    # Global Average Pooling: Average all pixels in each feature map
    # This gives us one number per feature map
    # It's like asking "how active is this feature map overall?"
    squeeze = layers.GlobalAveragePooling2D()(x)
    
    # Reshape to (batch, 1, 1, channels) for later operations
    # This shape is needed for the dense layers
    squeeze = layers.Reshape((1, 1, channels))(squeeze)
    
    # Step 2: Excitation
    # First dense layer: Reduce dimensions
    # This creates a bottleneck - forces the model to learn what's important
    # I'm dividing channels by reduction (default 16) to reduce parameters
    excitation = layers.Dense(channels // reduction, activation='relu')(squeeze)
    
    # Second dense layer: Back to original number of channels
    # Use sigmoid to get weights between 0 and 1
    # 1 = very important feature map, 0 = not important
    excitation = layers.Dense(channels, activation='sigmoid')(excitation)
    
    # Step 3: Scale
    # Multiply original features by attention weights
    # Important features get multiplied by ~1 (kept), unimportant by ~0 (suppressed)
    # This is the key - we're reweighting the feature maps!
    scaled = layers.Multiply()([x, excitation])
    
    return scaled


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build an Attention-based CNN model.
    
    I'm adding attention blocks after each convolutional block.
    This should help the model focus on important features at each stage.
    
    Parameters:
    - input_shape: Image size
    - num_classes: Number of classes
    """
    # Use Functional API because we need attention blocks
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    # Apply attention after pooling
    # I'm applying attention at each stage to see if it helps
    x = channel_attention_block(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Apply attention again
    x = channel_attention_block(x)
    
    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Apply attention one more time
    x = channel_attention_block(x)
    
    # Flatten for dense layers
    x = layers.Flatten()(x)
    
    # Classification head - same as other models
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
