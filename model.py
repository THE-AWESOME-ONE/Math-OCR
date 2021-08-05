import itertools
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers

X_train = pickle.load(open("X_train.pickle", "rb"))
X_train = np.array(X_train)


y_train = pickle.load(open("y_train.pickle", "rb"))
y_train = np.array(y_train)

print("Training data loaded")
X_test = pickle.load(open("X_test.pickle", "rb"))
X_test = np.array(X_test)

y_test = pickle.load(open("y_test.pickle", "rb"))
y_test = np.array(y_test)
print("Testing data loaded")

model = keras.Sequential(
    [
        layers.Conv2D(45, 3, padding="valid", activation="relu", input_shape=(45, 45, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(82),
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)
print("Model Constructed \n Fitting.....")
epochs =30
model.fit(X_train, y_train, batch_size=64, epochs=epochs, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)

acc = model.history['accuracy']
val_acc = model.history['val_accuracy']

loss = model.history['loss']
val_loss = model.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model.save("Model/")
tfjs.converters.save_keras_model(model, "tfjsmodel")