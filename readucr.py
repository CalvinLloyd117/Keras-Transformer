import numpy as np

"""
Function to split the datasets into x and y dimmensions.
y dimension is the first column of the dataset (what the model is trying to predict.)
The rest of the datasets are the training data.
Arguments: 
    filename: a file to convert to x, y. accepts a .tsv file. 
        Should not include any strings, or any column names, or row names.
"""
def readucr(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    y = data[:, 0]
    x = data[:, 1:]
    return x, y

# def readucr(filename):
#     data = np.loadtxt(filename, delimiter="\t")
#     y = data[:, 0]
#     x = data[:, 1:]
#     return x, y.astype(int)