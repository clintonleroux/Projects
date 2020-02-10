#!/usr/bin/env python
# coding: utf-8

# (Tutorial followed: "tensorflow.org/tutorials/keras/classification")
# 1 Import
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# 2 Load the data:

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()


# 3 Define the class_names of the labels:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 4 Shrink the data so that it's easier to process
train_images = train_images/255.0
test_images = test_images/255.0

# 5 Creating a model

model = keras.Sequential([
    #Input Layer(28 X 28 values of the pixels)
    keras.layers.Flatten(input_shape=(28,28)),
    #Hidden Layer
    keras.layers.Dense(128, activation="relu"),
    #Output Layer
    keras.layers.Dense(10, activation="softmax")
])


# 6 Set up perameters for models
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# 7 Train model
model.fit(train_images, train_labels, epochs=5)
#Shows data of tests(not required)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Accuracy: ",test_acc)

# 8 Make a prediction using the model

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    #Show image in greyScale
    plt.imshow(test_images[i], cmap= plt.cm.binary)
    #Prints the actual label for refernece
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    #Prints the name of the Highest prediction
    plt.title("Prediction" + class_names[np.argmax(prediction[i])])
    #Show image
    plt.show()