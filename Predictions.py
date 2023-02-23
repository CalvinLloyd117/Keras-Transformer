from Data import *
import csv
from config import loadConfig
from LoadModel import loadConfiguredModel

##################################################################################
#This predition uses the new regression layer to predict the Rank for the x_test set monkeys

##################################################################################
##################################################################################
#Create a csv and output the rows from predictions 
##################################################################################
def outputPredictionsToCsv(data, filename, verbose=False):
    loadConfig()
    model=loadConfiguredModel()
    predictions = model.predict(data, batch_size=cfg['batch_size'])
    if verbose:
        print("Predictions:")
        print(predictions)
    with open((cfg['base_URL']+"/Predictions/"+filename), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        rowname = (cfg['model_name']+"_predictions").replace(" ", "")
        wr.writerow([rowname])
        for ln in predictions:
            wr.writerow(ln)


filename = (cfg['model_name']+"_x_test_predictions.csv")
outputPredictionsToCsv(x_test, filename, verbose=True)


arby_data = x_train[:55, :]
filename=(cfg['model_name']+"_arby_predictions.csv")
outputPredictionsToCsv(arby_data, filename, verbose=True)

# model = keras.models.load_model(location)
model = loadConfiguredModel()
arby_predictions = model.predict(arby_data)
arby_predictions = arby_predictions[(arby_predictions >= 0) & (arby_predictions <= 1)]
print("Arby Predictions",arby_predictions)


print("Arby Mean: ",np.mean(arby_predictions))
print("Arby Max: ",np.max(arby_predictions))
print("Arby Min: ",np.min(arby_predictions))
print("Arby Median: ",np.median(arby_predictions))