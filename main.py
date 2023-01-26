# Code adapted from https://keras.io/examples/timeseries/timeseries_transformer_classification/

from tensorflow import keras
from Model import build_model, make_or_restore_model, callbacks
from Data import *

import yaml

#Load in configuration yaml for storing parameters.
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

#building the model
model = build_model(
    input_shape,
    head_size=cfg["head_size"],
    num_heads=cfg["num_heads"],
    ff_dim=cfg["ff_dim"],
    num_transformer_blocks=cfg["num_layers"],
    mlp_units=[cfg["mlp_units"]],
    mlp_dropout=cfg["mlp_dropout"],
    dropout=cfg["dropout"],
)

model.compile(
    loss=cfg["loss"],
    optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
    metrics=cfg["metric"],
)

model.summary()

model.fit(
    # train_dataset,
    x_train,
    y_train,
    validation_split=cfg["validation_split"],
    epochs=cfg["num_epochs"],
    batch_size=cfg["batch_size"],
    callbacks=callbacks,
)

results = model.evaluate(x_test, y_test, verbose=1)

location="./Models/"+cfg["model_name"]
print(location)

model.save(location)