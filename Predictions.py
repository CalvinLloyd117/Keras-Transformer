from Data import *
import csv
from config import loadConfig
from LoadModel import loadConfiguredModel

'''
Function creates a .cvs file containing the model predictions
Arguments:
    data: The data (in the format that the model is expecting)
    filename: a string indicating what to call the file.
    verbose (default = False): If set to true, the predictions 
    will be printed as well as output into a csv.

Returns:
    A .csv file with predictions. The file can be found in cfg['base_URL']/Predictions/filename
    The prediction column within the csv will be named cfg['model_name']+"_predictions" with
    all spaces removed.
'''
def outputPredictionsToCsv(data, filename, verbose=False):
    loadConfig() #Load the config file to use the parameters.
    model=loadConfiguredModel()

    #use the model to predfict on the data
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


filename = (cfg['model_name']+"_x_train_predictions.csv")
outputPredictionsToCsv(x_train, filename, verbose=True)

filename = (cfg['model_name']+"_x_test_predictions.csv")
outputPredictionsToCsv(x_test, filename, verbose=True)


arby_data = x_train[:95, :]
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