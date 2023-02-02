import pandas as pd

"""
Function to split the datasets into x and y dimmensions.
y dimension is the first column of the dataset (what the model is trying to predict.)
The rest of the datasets are the training data.
Arguments: 
    filename: a file to convert to x, y. accepts a .tsv file. 
        Should not include any strings, or any column names, or row names.
"""
def generateLabels(filename):
    df = pd.read_csv(filename)
    names=list(df.columns)
    x=names[1:]
    y=names[0]
    return x, y
    # y = data[1:1, 0]
    # x = data[1:1, 2:]
    # print("Y: ", y)
    # print("X: ", x)
    # print(x.shape)
    # print(y.shape)
    # return x, y

# def readucr(filename):
#     data = np.loadtxt(filename, delimiter="\t")
#     y = data[:, 0]
#     x = data[:, 1:]
#     return x, y.astype(int)