# Load Dataset from scikit-learn library

from sklearn.datasets import load_iris
datasets = load_iris()

# Input Numbers

data   = datasets.data
target = datasets.target

import numpy as np
np.unique(target)

# Load Tensorflow and Keras
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Create a Model
model = Sequential()
# 1st Hidden Layer
model.add(Dense(8, input_dim = 4, activation = 'relu'))
# 2nd Hidden Layer
model.add(Dense(8, input_dim = 8, activation = 'relu'))
# Final Layer
model.add(Dense(3, input_dim = 8, activation = 'softmax'))

model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ['accuracy'])

from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(data,target, test_size = 0.2)

from tensorflow.keras.utils import to_categorical

update_train_target = to_categorical(train_target)

print(update_train_target[:5])

history = model.fit(train_data, update_train_target, epochs = 5)

import matplotlib.pyplot as plt

# plotting for loss function
plt.plot(history.history['loss'])
plt.xlabel("Iterations")
plt.ylabel("Loss Values")
plt.show()

# plotting for loss accuracy
plt.plot(history.history['accuracy'])
plt.xlabel("Iterations")
plt.ylabel("Accuracy Values")
plt.show()


# Predict target value and plotting
predicted_target = model.predict(test_data)

print(predicted_target)
print("Pre-Labels:    ",np.argmax(predicted_target,axis=1))
print("Actual Labels: ",test_target)
