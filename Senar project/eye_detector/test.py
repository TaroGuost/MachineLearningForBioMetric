# -*- coding: utf-8 -*-

import os
import cv2

def load_image(file_path):
    return cv2.imread(file_path)

def extract_label(file_name):
    return 1 if "open" in file_name else 0 # open eyes are 1 & closed eyes are 0

train_path = "tran/"
image_files = os.listdir(train_path)
train_images = [load_image(train_path + file) for file in image_files]
train_labels = [extract_label(file) for file in image_files]

# this will probably not be an issue with the real deal, since all images will be the same size
# so we can re-train with that view from the pilot's eye from the instrument...
def preprocess_image(img, side = 96): # number of pixels on the smallest side
    # average eye aspect ratio is 1.87 by 1 (it requires an int, so I rounded 1.87 to 2)
    eye_aspect_ratio = 2
    min_side = min(img.shape[0], img.shape[1])
    img = img[:min_side, :min_side * eye_aspect_ratio]
    img = cv2.resize(img, (side * eye_aspect_ratio, side)) # average eye aspect ratio of 1.87 by 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img / 255.0
  
import matplotlib.pyplot as plt
# %matplotlib inline
preview_index = 50
plt.subplot(1,2,1)
plt.imshow(train_images[preview_index])
plt.subplot(1,2,2)
plt.imshow(preprocess_image(train_images[preview_index]), cmap="gray")
# some images are showing up wonky here b/c of your aspect ratio side multiplier ^^^
# it does allow to get the entire eye within the frame though so worth it...
# looks weird to us but the neural net will understand

for i in range(len(train_images)):
    train_images[i] = preprocess_image(train_images[i])
    
import numpy as np

train_images = np.expand_dims(train_images, axis = -1)
train_labels = np.array(train_labels)

import tensorflow as tf
print("Tensorflow:", tf.__version__)

layers = [
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tf.nn.relu, input_shape=train_images.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax) # probability for each of the classes (2 as of now)
]

# https://keras.io/optimizers/ (see Adam section)
# https://keras.io/losses/ (see sparse_categorical_accuracy)
model = tf.keras.Sequential(layers)
model.compile(optimizer='adam',
              # optimizer=tf.keras.optimizers.Adam(),
              # loss='binary_crossentropy',
              # loss=tf.losses.sparse_softmax_cross_entropy(),
              # loss=tf.keras.backend.sparse_categorical_crossentropy(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              # metrics=[tf.metrics.accuracy()])
""" TensorFlow 2.0.0 Beta
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.metrics.SparseCategoricalAccuracy()])
"""

# Training the model
model.fit(train_images, train_labels, epochs=10, batch_size=50)
# model.save_weights("model.tf")
model.save("model.h5")

import pickle
import cv2

test_file = "open_eye_104.jpg"
uploads = cv2.imread("test/" + test_file)

""" For Multiple Images...
import glob
import cv2
images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]
"""
test_path = "test/"
test_files = os.listdir(test_path)
test_images = [load_image(test_path + file) for file in test_files]
test_labels = [extract_label(file) for file in test_files]

eval_model = tf.keras.Sequential(layers)
eval_model.load_weights("model.h5")

right = 0
wrong = 0

for i in range(len(test_images)):
  temp = test_images[i]
  temp = [preprocess_image(temp)]
  eval_predictions = eval_model.predict(np.expand_dims(temp , axis= -1))
  if (test_labels[i] == 1 and eval_predictions[0][1] > eval_predictions[0][0]) or (test_labels[i] == 0 and eval_predictions[0][0] > eval_predictions[0][1]):
    right += 1
  else:
    wrong += 1

print(right/64)
    
    
    