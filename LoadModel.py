from tensorflow import keras
import tensorflow as tf
import pydot
import graphviz
import seaborn as sb
import yaml
import matplotlib.pyplot as plt
from Data import *
import csv
from readucr import readucr

#Load in configuration yaml for storing parameters.
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

location="./Models/" + cfg["model_name"]
model = keras.models.load_model(location)

# attention_layer=model.layers[2]
# num_vals=1682
# _, attention_scores = attention_layer(x_train[:25], x_test[:25], return_attention_scores=True) # take one sample
# # fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[5,5,0.2]))
# # sb.heatmap(attention_scores[0, 0, :10, :10], annot=True, cbar=False)
# heatmap = sb.heatmap(attention_scores[0, 1, :num_vals, :num_vals], annot=False, cbar=True,
#     xticklabels=[idx for idx in x_labels[:num_vals]],
#     yticklabels=[idx for idx in x_labels[:num_vals]]
# )
# # fig.colorbar(axs[1].collections[0], cax=axs[2])
# heatmap.figure.savefig((cfg['model_name']+"_attention_heatmap.pdf"))
# plt.show()

# Output the data to a csv.

##################################################################################
#This predition uses the new regression layer to predict the Rank for the x_test set monkeys
predictions = model.predict(x_train, batch_size=cfg['batch_size'])
print("Train set predictions:")
print(predictions)
##################################################################################
##################################################################################
#Create a csv and output the rows from predictions 
filename = (cfg['model_name']+"_x_train_predictions.csv")
with open((cfg['base_URL']+"/Predictions/"+filename), 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     rowname = (cfg['model_name']+"_predictions")
     wr.writerow([rowname])
     for ln in predictions:
        wr.writerow(ln)
##################################################################################

##################################################################################
#This predition uses the new regression layer to predict the Rank for the x_test set monkeys
predictions = model.predict(x_test, batch_size=None)
print("Test set predictions:")
print(predictions)
##################################################################################
##################################################################################
#Create a csv and output the rows from predictions 
filename = (cfg['model_name']+"_x_test_predictions.csv")
with open((cfg['base_URL']+"/Predictions/"+filename), 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     rowname = (cfg['model_name']+"_predictions")
     wr.writerow([rowname])
     for ln in predictions:
        wr.writerow(ln)
##################################################################################

#Going to try some predictions with just the arby_2 data
# arby_x, arby_y = readucr((cfg['base_URL']+"gait_train_arby.csv"))
# arby_pred = model.predict(x_train[:, 1:55])
arby_data = x_train[:55, :]
print(arby_data.shape)
print(arby_data)
arby_pred = model.predict(arby_data)
print(arby_pred)

filename = (cfg['model_name']+"_x_train_predictions_arby.csv")
with open((cfg['base_URL']+"/Predictions/"+filename), 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     rowname = (cfg['model_name']+"_predictions")
     wr.writerow([rowname])
     for ln in arby_pred:
        wr.writerow(ln)