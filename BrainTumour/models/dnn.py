"""
DNN Model for Brain Tumor Detection

This is a baseline model using only fully connected (dense) layers.
I'm including this to see how it compares to CNNs.

I learned that:
- DNNs flatten images, which loses spatial information
- This is NOT ideal for images, but I want to see the difference
- It's good to have a baseline to compare against

This model will probably perform worse than CNN, but that's okay - I'm learning!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build a DNN (fully connected) model.
    
    This is intentionally simple - I'm using it as a baseline.
    I expect CNN to perform better because it preserves spatial structure.
    
    Parameters:
    - input_shape: Image size
    - num_classes: Number of classes (2 for tumor/no_tumor)
    """
    model = keras.Sequential()
    
    # Flatten the image - this is where we lose spatial information
    # A 224x224x3 image becomes a single vector of 150,528 numbers!
    # This is a LOT of parameters - might be slow to train
    model.add(layers.Flatten(input_shape=input_shape))
    
    # First dense layer with 512 neurons
    # I'm using large numbers because we have a lot of input features
    # This creates many connections - might be overkill
    model.add(layers.Dense(512, activation='relu'))
    
    # Dropout to prevent overfitting
    # I'm using 0.5 (50%) which is common
    model.add(layers.Dropout(0.5))
    
    # Second dense layer - reducing size gradually
    # I read that gradually reducing is better than sudden drops
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Third dense layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # Output layer
    # Softmax gives probabilities for each class
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model
