from tensorflow import keras
import tensorflow as tf
import pydot
import graphviz
import seaborn as sb
import yaml
import matplotlib.pyplot as plt
from Data import *

#Load in configuration yaml for storing parameters.
with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

location="./Models/" + cfg["model_name"]
model = keras.models.load_model(location)

# print(model.history)

# results = model.evaluate(x_test, y_test, verbose=1)
# print(results)

# dot=keras.utils.model_to_dot(
#     model,
#     show_shapes=True,
#     show_dtype=True,
#     show_layer_names=True,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=96,
#     subgraph=True,
#     layer_range=None,
#     show_layer_activations=False,
# )

# model.summary()

# This block produces the messed up heat map for the model prediction.
#####################################################################################
# attention_layer = keras.Model(inputs=model.input, 
#                                  outputs=model.get_layer("multi_head_attention").output)

#https://stackoverflow.com/questions/70573362/tensorflow-how-to-extract-attention-scores-for-graphing
# idx_word = {v: k for k, v in x_test[0].items()}

# print("x_labels: ",x_labels)
# print("y_labels: ",y_labels)
attention_layer=model.layers[2]
num_vals=1682
_, attention_scores = attention_layer(x_train[:5], x_test[:5], return_attention_scores=True) # take one sample
# fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[5,5,0.2]))
# sb.heatmap(attention_scores[0, 0, :10, :10], annot=True, cbar=False)
heatmap = sb.heatmap(attention_scores[0, 1, :num_vals, :num_vals], annot=False, cbar=True,
    xticklabels=[idx for idx in x_labels[:num_vals]],
    yticklabels=[idx for idx in x_labels[:num_vals]]
)
# fig.colorbar(axs[1].collections[0], cax=axs[2])
heatmap.figure.savefig((cfg['model_name']+"_attention_heatmap.pdf"))
plt.show()

# # attention_scores = attention_layer(x_test[:1], y_test[:1],  return_attention_scores=True) # take one sample
# fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[5,5,0.2]))
# sb.heatmap(attention_scores[:5, :16, 0], annot=True, cbar=False, ax=axs[0])
# sb.heatmap(attention_scores[:5, :16, 0], annot=True, yticklabels=False, cbar=False, ax=axs[1])
# fig.colorbar(axs[1].collections[0], cax=axs[2])
# plt.show()
#######################################################################################
# head_num=1
# inp = tf.expand_dims(x_train[0,:], axis=0)

# emb = model.layers[1](model.layers[0]((inp)))
# attention=model.layers[2]._query_dense(emb)

#Attempting to correct the plot for weights, not predictions.
#######################################################################################
# attention = keras.Model(inputs=model.input, 
#                                  outputs=model.get_layer("multi_head_attention").output)

#Returns a series of multi dimensional arrays for the various components of the weights (attention_output, query, key, etc.)
#Not sure how to access them individually yet.
# attention_scores = attention.get_weights()




# print(attention_scores)
# print(attention_r)

# # attention_scores = attention_layer(x_test[:1], y_test[:1],  return_attention_scores=True) # take one sample
# fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[5,5,0.2]))
# attention_output= [var for var in attention_scores.variables if "value" in var.name]

# kernel = attention_output[0]
# bias = attention_output[1]

#Block for printing a single column heatmap (uninterprettable)
# print("Atttention Scores: ", attention_scores)
#################################################
#Small working block for attention?
# fig, axs = plt.subplots(ncols=1)
# for arr in attention_scores:
#     arr=np.squeeze(arr.numpy())
#     if(arr.shape != (4, 256)):
#         continue
#     print(arr.shape)
#     sb.heatmap(arr)
#     # sb.heatmap(arr, annot=True, yticklabels=False, cbar=False, ax=axs[1])
#     # fig.colorbar(axs[1].collections[0], cax=axs[2])
#     plt.show()
#####################################################
# sb.heatmap(attention_scores[5], annot=True, cbar=False, ax=axs[0])
# sb.heatmap(attention_scores[5], annot=True, yticklabels=False, cbar=False, ax=axs[1])
# fig.colorbar(axs[1].collections[0], cax=axs[2])
# plt.show()

# print(attention_output)
# sb.heatmap(attention_output, annot=True, cbar=False, ax=axs[0])
# sb.heatmap(attention_output, annot=True, yticklabels=False, cbar=False, ax=axs[1])
# fig.colorbar(axs[1].collections[0], cax=axs[2])
# plt.show()
#######################################################################################


# attention_scores = attention.predict(x_test)
# print(attention_r)
# attention_layer=model.layers[2]
# samples=5
# test_targets = tf.random.normal((samples, 8, 16))
# test_sources = tf.random.normal((samples, 4, 256))

# _, attention_scores = attention_layer(test_targets[:0], test_sources[:0], return_attention_scores=True) # take one sample
# fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[5,5,0.2]))
# sb.heatmap(attention_scores[0, 0, :, :], annot=True, cbar=False, ax=axs[0])
# sb.heatmap(attention_scores[0, 1, :, :], annot=True, yticklabels=False, cbar=False, ax=axs[1])
# fig.colorbar(axs[1].collections[0], cax=axs[2])
# plt.show()
# result = attention_layer.predict(x_test)
# print(result)

# This predition uses the new regression layer to predict the Rank for the x_train set monkeys
temp = model.predict(x_test)
print(temp)

# This predition uses the new regression layer to predict the Rank for the x_test set monkeys
# temp = model.predict(x_test)
# print(temp)

# print(x_train.shape)
# print(x_test.shape)
# print(temp.shape)

########################################################################################################
#Block taken from:
# https://stackoverflow.com/questions/64622833/interpreting-attention-in-keras-transformer-official-example
#Should return the exact style of heatmap that we want. Maybe the code can be modified.
########################################################################################################
# import seaborn as sns
# import matplotlib.pyplot as plt

# head_num=1
# inp = tf.expand_dims(x_train[0,:], axis=0)
# emb = model.layers[1](model.layers[0]((inp)))

# self_attn = model.layers[2]

# # compute Q, K, V
# query = self_attn._query_dense(emb)
# key = self_attn._key_dense(emb)
# value = self_attn._value_dense(emb)

# # separate heads
# query = self_attn._separate_heads(query, 1) # batch_size = 1
# key = self_attn._separate_heads(key, 1) # batch_size = 1
# value = self_attn._separate_heads(value, 1) # batch_size = 1
# # compute attention scores (QK^T)
# attention, weights = self_attn(query, key, value)

# idx_word = {v: k for k, v in keras.datasets.imdb.get_word_index().items()}
# plt.figure(figsize=(30, 30))
# sns.heatmap(
#     weights.numpy()[0][head_num], 
#     xticklabels=[idx_word[idx] for idx in inp[0].numpy()],
#     yticklabels=[idx_word[idx] for idx in inp[0].numpy()]
# )
########################################################################################################

