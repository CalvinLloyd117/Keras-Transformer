from tensorflow import keras
# import tensorflow as tf
# import pydot
# import graphviz
# import seaborn as sb
# import yaml
# import matplotlib.pyplot as plt
from Data import *
# import csv
from config import loadConfig
# from readucr import readucr


##################################################################################
# Function loads in the currently configured model. To load a model, change the 
# 'model_name' parameter in the config.yaml. If the model exists, it will be 
# returned.
##################################################################################
def loadConfiguredModel():
   # with open("config.yaml", "r") as ymlfile:
   #    cfg = yaml.safe_load(ymlfile)
   #Load in configuration yaml for storing parameters.
   loadConfig()
   location="./Models/" + cfg["model_name"]
   print("Loading ", location)
   model = keras.models.load_model(location)
   if model:
      print("Model Loaded Successfully!")
      return model
   else:
      print("Model load error.")

# model=loadConfiguredModel()
