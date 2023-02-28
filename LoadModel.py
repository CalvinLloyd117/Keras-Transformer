from tensorflow import keras
from Data import *
from config import loadConfig

"""
Function loads in the currently configured model. To load a model, change the 
'model_name' parameter in the config.yaml. If the model exists, it will be 
returned.
"""
def loadConfiguredModel():
   loadConfig()
   location="./Models/" + cfg["model_name"]
   print("Loading ", location)
   model = keras.models.load_model(location)
   if model:
      print("Model Loaded Successfully!")
      return model
   else:
      print("Model load error.")

