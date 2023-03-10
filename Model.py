from tensorflow import keras
from tensorflow.keras import layers
from Transformer import transformer_encoder

from Data import *

import os

import yaml

#Load in configuration yaml for storing parameters.
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./Models/checkpoints/"+cfg["model_name"]
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

"""
Callback functions to add to the model fitting.
Includes:
    Early stopping - If the loss metric doesn't improve within patience=10 epochs, 
        discontinue training and restore the best model.

    Model Checkpointing - in case training is interupted, save checkpoint 
        every 'checkpoint_save_freq' batches as long as loss is improved. 
"""
callbacks = [
    keras.callbacks.EarlyStopping(patience=cfg["early_stopping"], monitor="loss", restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath=checkpoint_dir+"/" + cfg["model_name"]+"_checkpoint",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="loss",
        verbose=1,
        save_freq=cfg["checkpoint_save_freq"],
    )
]

'''
Function to build a Keras model using the Transformer network architecture found in 
Attention is All You Need (https://arxiv.org/abs/1706.03762) consisting of:
    n transformer_encoder blocks
    layers.GlobalAveragePooling1D
    n layers.Dense(dim, activation="relu")(x)
    n layers.Dropout(mlp_dropout)(x)
    either regression layer, or categorical layer

Arguments:
    input_shape: The shape of the input data.
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks: Number of tansformer blocks to add to the network.
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    regression=True,
    n_classes=1
'''
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    regression=True,
    n_classes=1
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    if regression == True:
        outputs = layers.Dense(1)(x) #regression
    else:
        outputs = layers.Dense(n_classes, activation="softmax")(x) #categorical
    return keras.Model(inputs, outputs)

"""
Function that checks if there have been any checkpoints made for this model name, 
and if there have been, load the existing model and return it.
Otherwise, create a new model, and return that.
"""
def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return build_model(
            input_shape,
            head_size=cfg["head_size"],
            num_heads=cfg["num_heads"],
            ff_dim=cfg["ff_dim"],
            num_transformer_blocks=cfg["num_transformer_layers"],
            mlp_units=[cfg["mlp_units"]],
            mlp_dropout=cfg["mlp_dropout"],
            dropout=cfg["dropout"],
        )