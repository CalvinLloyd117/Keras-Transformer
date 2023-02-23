import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y


# root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
root_url = "./Data/"
x_train, y_train = readucr(root_url + "gait_train.tsv")
x_test, y_test = readucr(root_url + "gait_test.tsv")

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

n_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

# def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
#     # Normalization and Attention
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout
#     )(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs

#     # Feed Forward Part
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     return x + res

# def build_model(
#     input_shape,
#     head_size,
#     num_heads,
#     ff_dim,
#     num_transformer_blocks,
#     mlp_units,
#     dropout=0,
#     mlp_dropout=0,
# ):
#     inputs = keras.Input(shape=input_shape)
#     x = inputs
#     # for _ in range(num_transformer_blocks):
#     #     x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

#     x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
#     for dim in mlp_units:
#         x = layers.Dense(dim, activation="relu")(x)
#         x = layers.Dropout(mlp_dropout)(x)
#     # outputs = layers.Dense(n_classes, activation="softmax")(x)
#     outputs = layers.Dense(1, activation="linear")(x)
#     return keras.Model(inputs, outputs)

# input_shape = x_train.shape[1:]

# model = build_model(
#     input_shape,
#     head_size=256,
#     num_heads=4,
#     ff_dim=4,
#     num_transformer_blocks=4,
#     mlp_units=[128],
#     mlp_dropout=0.4,
#     dropout=0.25,
# )

# model.compile(
#     loss="mean_squared_error",
#     optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#     metrics=["accuracy"],
# )
# model.summary()

# callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

# model.fit(
#     x_train,
#     y_train,
#     validation_split=0.2,
#     epochs=500,
#     batch_size=10,
#     callbacks=callbacks,
# )

location="./Models/"+"classification_test2"
# print(location)

# model.save(location)

# model.evaluate(x_test, y_test, verbose=1)



import csv
model = keras.models.load_model(location)
predictions = model.predict(x_train[:57 ])
print(predictions)
predictions = predictions[(predictions >= 0) & (predictions <= 1)]
print(np.mean(predictions))
print(np.max(predictions))
print(np.min(predictions))
with open(("./Data"+"/Predictions/"+"classification_test_pred2.csv"), 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     rowname = ("classification_test2"+"_predictions")
     wr.writerow([rowname])
     for ln in predictions:
        wr.writerow(ln)