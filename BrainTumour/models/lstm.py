"""
LSTM Model for Brain Tumor Detection

I'm experimenting with combining CNN and LSTM.
LSTMs are usually for sequences (like text or time series), but I want to try them on images.

The idea:
1. Use CNN to extract features from image
2. Reshape features into a sequence
3. Use LSTM to process the sequence
4. Classify

I'm not sure if this will work well, but experimenting is part of learning!
This is more of an exploration than a proven approach.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build a CNN-LSTM hybrid model.
    
    I'm using Functional API here because I need to reshape the data.
    Sequential API doesn't work well when you need to reshape.
    
    Parameters:
    - input_shape: Image size
    - num_classes: Number of classes
    """
    # Input layer - define what the model expects
    inputs = layers.Input(shape=input_shape)
    
    # CNN Feature Extractor
    # I'm using the same CNN structure as my basic CNN model
    # This extracts features from the image
    
    # First conv layer
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second conv layer
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third conv layer
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Global Average Pooling
    # This averages all pixels in each feature map
    # Instead of flattening, we get one number per feature map
    # Result: (batch_size, 128) - one value per feature map
    x = layers.GlobalAveragePooling2D()(x)
    
    # Reshape for LSTM
    # LSTM expects sequences, so I need to reshape
    # (batch, features) -> (batch, 1, features)
    # This creates a sequence with 1 timestep
    x = layers.Reshape((1, -1))(x)
    
    # Repeat the sequence to make it longer
    # I'm creating 8 timesteps - this is a bit artificial
    # I'm not sure if this is the best approach, but I'm experimenting
    x = layers.RepeatVector(8)(x)
    # Now shape is: (batch, 8, features)
    
    # First LSTM layer
    # return_sequences=True means output all timesteps (not just the last one)
    # I'm using 128 units - same as my CNN dense layer
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    
    # Second LSTM layer
    # return_sequences=False means only output the final timestep
    # This reduces the output size
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    
    # Classification head - same as other models
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model connecting inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
