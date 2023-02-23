from tensorflow import keras
import seaborn as sb
import matplotlib.pyplot as plt
from Data import *
import math
from config import loadConfig
from LoadModel import loadConfiguredModel
#################################################################################
#Function produces an attention heatmap for the currently configured model.
#################################################################################
def createHeatmapForCurrentModel(verbose=True, save_fig=True):
    #Load in configuration yaml for storing parameters.
    # with open("config.yaml", "r") as ymlfile:
    #     cfg = yaml.safe_load(ymlfile)
    loadConfig()
    model=loadConfiguredModel()

    attention_layer=model.layers[2]

    _, attention_scores = attention_layer(x_train[:1], x_test[:1], return_attention_scores=True) # take one sample

    #Find the size of the attention scores
    _,num_heads,_,num_vals=attention_scores.shape
    if verbose:
        print("Creating heatmap for currently configured model.")
        print("num_vals: ", num_vals)
        print("Attention_scores Shape",attention_scores.shape)
        print("Attention_scores",attention_scores)

        #Code to play the attention of the network
        print(type(attention_scores))

    heatmap = sb.heatmap(attention_scores[0, 1, :num_vals, :num_vals], annot=False, cbar=True,
        xticklabels=[idx for idx in x_labels[:num_vals]],
        yticklabels=[idx for idx in x_labels[:num_vals]]
    )
    
    if save_fig:
        print("Saving figure as ", (cfg['model_name']+"_attention_heatmap.pdf"))
        heatmap.figure.savefig((cfg['model_name']+"_attention_heatmap.pdf"))

    plt.show()
    #End Plotting Code

def saveAttentionScores(min_attention_threshold, verbose=False):
    # with open("config.yaml", "r") as ymlfile:
    #         cfg = yaml.safe_load(ymlfile)

    location="./Models/" + cfg["model_name"]
    model = keras.models.load_model(location)
    if not model:
        return

    attention_layer=model.layers[2]
    _, attention_scores = attention_layer(x_train[:1], x_test[:1], return_attention_scores=True) # take one sample
    _,num_heads,_,num_vals=attention_scores.shape
    min_attention_threshold = min_attention_threshold
    num_high_attention = 0
    high_attention_scores = []
    names = []
    for head in range(num_heads):
        for val1 in range(math.floor(num_vals)):
            for val2 in range(math.floor(num_vals)):
                score = attention_scores[0, head, val1, val2].numpy()
                if score > min_attention_threshold:
                    if x_labels[val1] not in names:
                        names.append(x_labels[val1])
                    if x_labels[val2] not in names:
                        names.append(x_labels[val2])
                        num_high_attention+=1
    if verbose:
        print("Number of High Scoring attributes", num_high_attention)
        print("Names of Attributes", names)
    
    print("Saving Attention Scores as ", cfg["model_name"]+"_high_attention_names.txt")
    f = open(cfg["model_name"]+"_high_attention_names.txt", "a")
    for elem in names:
        f.write(elem+", ")
    f.close()

# saveAttentionScores(min_attention_threshold=0.05)
