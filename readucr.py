import numpy as np

"""
Function to split the datasets into x and y dimmensions.
y dimension is the first column of the dataset (what the model is trying to predict.)
The rest of the datasets are the training data.
Arguments: 
    filename: a file to convert to x, y. accepts a .csv file. 
        Should not include any strings, or any column names, or row names.
"""
def readucr(filename, regression):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    y = data[:, 0]
    x = data[:, 1:]
    if regression:
        return x, y
    #If we are doin categorical classification, we need to assign the y dimension to
    #an into to classify it.
    else:
        return x, y.astype(int)