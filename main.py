# Code adapted from https://keras.io/examples/timeseries/timeseries_transformer_classification/
from config import loadConfig
from tensorflow import keras
from Model import build_model, make_or_restore_model, callbacks
from Data import *
from AttentionScores import *

# import yaml

#Load in configuration yaml for storing parameters.
# with open("config.yaml", "r") as ymlfile:
#     cfg = yaml.safe_load(ymlfile)
loadConfig()
#building the model
model = build_model(
    input_shape,
    head_size=cfg["head_size"],
    num_heads=cfg["num_heads"],
    ff_dim=cfg["ff_dim"],
    num_transformer_blocks=cfg["num_transformer_layers"],
    mlp_units=[cfg["mlp_units"]],
    mlp_dropout=cfg["mlp_dropout"],
    dropout=cfg["dropout"],
    regression=cfg["regression"],
    n_classes=n_classes     
)

model.compile(
    loss=cfg["loss"],
    optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
    metrics=cfg["metric"],
)

model.summary()

model.fit(
    x_train,
    y_train,
    validation_split=cfg["validation_split"],
    epochs=cfg["num_epochs"],
    batch_size=cfg["batch_size"],
    callbacks=callbacks,
    shuffle=cfg["shuffle"]
)

results = model.evaluate(x_test, y_test, verbose=1)

location="./Models/"+cfg["model_name"]
print(location)

model.save(location)

createHeatmapForCurrentModel()

saveAttentionScores(0.05)

arby_data = x_train[:95, :]
arby_predictions = model.predict(arby_data)
arby_predictions = arby_predictions[(arby_predictions >= 0) & (arby_predictions <= 1)]
print("Arby Predictions",arby_predictions)

print("Arby Mean: ",np.mean(arby_predictions))
print("Arby Max: ",np.max(arby_predictions))
print("Arby Min: ",np.min(arby_predictions))
print("Arby Median: ",np.median(arby_predictions))
