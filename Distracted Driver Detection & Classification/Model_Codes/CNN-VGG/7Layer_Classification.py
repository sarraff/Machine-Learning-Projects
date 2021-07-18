import cv2  
import numpy as np 
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt 

#The h5py package is a Pythonic interface to the HDF5 binary data format
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint

# load the weights that yielded the best validation accuracy
model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100,100,3),padding='same'),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(100, activation='relu'),
      # tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights('/content/drive/MyDrive/BTP_Driver_Distraction/5Lvgg_cnn_from_scratch.hdf5')

print("Weighted value is taken out")


# Shows the prediction result and misclassification
row = 100
col = 100

def matrix():
    X_data = []
    img_seq=[]
    files = glob.glob(r"/content/drive/MyDrive/BTP_Driver_Distraction/Test500/test500/*.jpg")
    for myFile in files:
        image = cv2.imread (myFile) 
        img = cv2.resize(image,(row,col))
        X_data.append (img)
        img_seq.append(myFile.split('img_')[1].split('.jpg')[0])
    X_data = np.array(X_data)
    np.save('X_vgg_test', X_data)
    return img_seq

def Predict(model,X_input):
    X_input = X_input/255.0
    Y_pred = model.predict(X_input, verbose=0)
    y = np.zeros(Y_pred.shape[0])
    for i in range(len(Y_pred)):
        y[i] = np.argmax(Y_pred[i])
    return y

def Test_accuracy():
    Y_img_seq = matrix()
    X_test_500 = np.load('X_vgg_test.npy')
    Y_pred_case_int = Predict(model,X_test_500)
    Y_pred_case = []
    for i in Y_pred_case_int:
        Y_pred_case.append(f'c{int(i)}')
    print('Predicted Result:',Y_pred_case)
    #if Y_pred_case[0]==3:
     # print("It's the int dude")
    imgLabel = pd.read_csv("/content/drive/MyDrive/BTP_Driver_Distraction/Test500/test500/ImageLabels.csv")
    imgLabel["VGG-CNN"]=''
    imgLabel["Correct"]=''
    print(imgLabel.head(20))

    dict = {}
    with open('/content/drive/MyDrive/BTP_Driver_Distraction/Test500/test500/ImageLabels.csv', newline='') as File:  
        reader = csv.reader(File)
        for row in reader:
            dict[row[0]] = row[1]


    dictClass = {}
    dictClass['c0'] = "Safe Driving"
    dictClass['c1'] = "Texting Right"
    dictClass['c2'] = "Talking Phone Right"
    dictClass['c3'] = "Texting Left"
    dictClass['c4'] = "Talking Phone Left"
    dictClass['c5'] = "Operating Radio"
    dictClass['c6'] = "Drinking"
    dictClass['c7'] = "Reaching Behind"
    dictClass['c8'] = "Hair & Makeup"
    dictClass['c9'] = "Talking Passenger"



    correct = 100
    fig = plt.figure(figsize=(30,240))
    for i,val in enumerate(Y_img_seq):
        imgLabel.loc[imgLabel["image"]==int(val),"VGG-CNN"]=Y_pred_case[i]
        #print(val,dict[val],Y_pred_case[i])
        if (Y_pred_case[i] in str(imgLabel.loc[imgLabel["image"]==int(val)]["label"])):
            correct = correct+1
            imgLabel.loc[imgLabel["image"]==int(val),"Correct"]='Sahi'

            if i<200:
                image = cv2.imread (r"/content/drive/MyDrive/BTP_Driver_Distraction/Test500/test500/img_"+str(int(val))+".jpg") 
                img = cv2.resize(image,(150,150))
                ax = fig.add_subplot(50,5,i + 1, xticks=[], yticks=[])
                ax.imshow(img)

                valuez = ""
                j=0
                for k in range(0,10):
                    xyz = "c"+str(k)
                    if xyz in dict[val]:
                        if j>0:
                            valuez = valuez + " or " + dictClass[xyz]
                        else :
                            valuez = dictClass[xyz]
                            j = j+1
            ax.set_title("{} ({})".format(dictClass[Y_pred_case[i]],valuez),color=("blue"))

        else :
            imgLabel.loc[imgLabel["image"]==int(val),"Correct"]='Galat'
            if i<200:
                image = cv2.imread (r"/content/drive/MyDrive/BTP_Driver_Distraction/Test500/test500/img_"+val+".jpg") 
                #print(val,dict[val],Y_pred_case[i])
                img = cv2.resize(image,(150,150))
                ax = fig.add_subplot(50,5,i + 1, xticks=[], yticks=[])
                ax.imshow(img)

                valuez = ""
                j=0
                for k in range(0,10):
                    xyz = "c"+str(k)
                    if xyz in dict[val]:
                        if j>0:
                            valuez = valuez + " or " + dictClass[xyz]
                        else :
                            valuez = dictClass[xyz]
                            j = j+1
                             
            ax.set_title("{} ({})".format(dictClass[Y_pred_case[i]],valuez),color=("red"))


    print(imgLabel.head(20))
    total = len(Y_img_seq)
    genAcc = (correct/total) *100
    print(correct, "Images classified correctly out of ",total, "images")
    print("General Accuracy: ",genAcc)
    
Test_accuracy()