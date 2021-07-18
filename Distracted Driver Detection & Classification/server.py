"""
Part 1: Compile the model and save the weights into a file
In this case, it is '5Lvgg_cnn_from_scratch.hdf5'
"""

"""
Part 2: load the weights that yielded the best validation accuracy
"""
import cv2  
import tensorflow as tf
import os

import numpy as np
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100,100,3),padding='same'),
      #tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
      #tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(100, activation='relu'),
      # tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   #Error function
model.load_weights('/content/drive/MyDrive/BTP_Driver_Distraction/5Lvgg_cnn_from_scratch.hdf5')
print("Weighted value is taken out")

"""
Part 3: Install the required Anvil library 
            !pip install anvil-uplink
"""

"""
Part 4: Define server & connect server with the client using key.
"""
import anvil.server
anvil.server.connect("CE7B4DRR3AXDILXG2IZ6OJ6L-Y3IHHGL6H72CQ2XG")
import anvil.media          # Import necessary media functions

"""
Part 5: Define server side code below the request
"""
dictClass = {}
dictClass['c0'] = "Safe Driving"
dictClass['c1'] = "Texting Right"
dictClass['c2'] = "Talking on the Phone Right"
dictClass['c3'] = "Texting Left"
dictClass['c4'] = "Talking on the Phone Left"
dictClass['c5'] = "Operating the Radio"
dictClass['c6'] = "Drinking"
dictClass['c7'] = "Reaching Behind"
dictClass['c8'] = "Hair & Makeup"
dictClass['c9'] = "Talking to Passenger"


@anvil.server.callable
def predict_class(file):
    X_data=[]
    with anvil.media.TempFile(file) as f:
        img2=cv2.imread(f)
        img=cv2.resize(img2,(100,100))
        X_data.append(img)

    X_data = np.array(X_data)
    X_data = X_data/255.0

    Y_pred = model.predict(X_data, verbose=0)
    y = np.zeros(Y_pred.shape[0])

    for i in range(len(Y_pred)):
        y[i] = np.argmax(Y_pred[i])
    Y_pred_case = []
    for i in y:
        Y_pred_case.append(f'c{int(i)}')
    return dictClass[Y_pred_case[0]]

"""
Part 6: Activate the server side link to accept the client request
"""
anvil.server.wait_forever()