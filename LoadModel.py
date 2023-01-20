from tensorflow import keras
import yaml
from Data import *

#Load in configuration yaml for storing parameters.
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

location="./Models/" + cfg["model_name"]
model = keras.models.load_model(location)

print(model.history)

results = model.evaluate(x_test, y_test, verbose=1)
print(results)
