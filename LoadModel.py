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

dot=keras.utils.model_to_dot(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    subgraph=True,
    layer_range=None,
    show_layer_activations=False,
)

model.summary()
attention = keras.Model(inputs=model.input, 
                                 outputs=model.get_layer("multi_head_attention").output)

attention_scores = attention.predict(x_test)
# print(attention_r)

# attention_scores = attention_layer(x_test[:1], y_test[:1],  return_attention_scores=True) # take one sample
fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[5,5,0.2]))
sb.heatmap(attention_scores[:5, :16, 0], annot=True, cbar=False, ax=axs[0])
sb.heatmap(attention_scores[:5, :16, 0], annot=True, yticklabels=False, cbar=False, ax=axs[1])
fig.colorbar(axs[1].collections[0], cax=axs[2])
plt.show()

attention = keras.Model(inputs=model.input, 
                                 outputs=model.get_layer("multi_head_attention").output)

# attention_scores = attention.predict(x_test)
# print(attention_r)
attention_layer=model.layers[2]
samples=5
test_targets = tf.random.normal((samples, 8, 16))
test_sources = tf.random.normal((samples, 4, 256))

_, attention_scores = attention_layer(test_targets[:1], test_sources[:1], return_attention_scores=True) # take one sample
fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[5,5,0.2]))
sb.heatmap(attention_scores[0, 0, :, :], annot=True, cbar=False, ax=axs[0])
sb.heatmap(attention_scores[0, 1, :, :], annot=True, yticklabels=False, cbar=False, ax=axs[1])
fig.colorbar(axs[1].collections[0], cax=axs[2])
plt.show()
# result = attention_layer.predict(x_test)
# print(result)

# This predition uses the new regression layer to predict the Rank for the x_train set monkeys
# temp = model.predict(x_train)
# print(temp)

# This predition uses the new regression layer to predict the Rank for the x_test set monkeys
# temp = model.predict(x_test)
# print(temp)

# print(x_train.shape)
# print(x_test.shape)
# print(temp.shape)