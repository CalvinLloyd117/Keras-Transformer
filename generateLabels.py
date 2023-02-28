import pandas as pd

"""
Function to create a list of labels (column names) for a dataset. 
Used for creating lebels for heatmapping and plotting.

Arguments:
    filename: the filename (including path) to the .csv dataset to read the columns from.
"""
def generateLabels(filename):
    df = pd.read_csv(filename)
    names=list(df.columns)
    #x is all column names except column 0
    x=names[1:]
    #column 0 is corresponding to the attribute we are trying to predict.
    y=names[0]
    return x, y