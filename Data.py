import numpy as np
from readucr import readucr
from generateLabels import generateLabels
import yaml
import tensorflow as tf
#Load in configuration yaml for storing parameters.
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

root_url = cfg["base_URL"]

#Split the input data into x_train and y_train
#x_train is all of the data except the first column of the csv, which becomes y_train
x_train, y_train = readucr((root_url + cfg["training_data"]), regression=cfg['regression'])

#Split the testing data into x_test and y_test. x_test is all testing data (columns) except column 1, 
#which becomes y_test.
x_test, y_test = readucr((root_url + cfg["testing_data"]), regression=cfg['regression'])

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

#If we are doing regression, all data exists within a single class, and we find where it 
# lies within the class using predictions.
if cfg['regression'] == True:
    n_classes = 1
else:
    n_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

print("Y: ", y_train)
print("X: ", x_train)
print(x_train.shape)
print(y_train.shape)

input_shape = x_train.shape[1:]

x_labels, y_labels = generateLabels(root_url + cfg["training_data"])


# n_classes = len(np.unique(y_train))
