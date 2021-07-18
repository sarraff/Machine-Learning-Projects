import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import cv2
import glob

##############GENERALIZE CHECKING DATA GENERATE#################################

row = 64
col = 64
a = row*col*3

oimg1 = cv2.imread(r"D:\Kaggle_Distracted_Driver\Check_vectorize\img_104.jpg")
img1 = cv2.resize(oimg1,(row,col))

vector_newX = np.reshape(img1,(row*col*3,1))

imgs = glob.glob(r"D:\Kaggle_Distracted_Driver\Generalize_images\*.jpg")

for img in imgs:
    oriimg = cv2.imread(img)
    img0 = cv2.resize(oriimg,(row,col))
    flat = img0.reshape(a,1)
    vector_newX = np.c_[vector_newX,flat]
    print(img)

vector_newX = vector_newX.T

finalX_gen = vector_newX[1:,:]

###################IMPORT TRAINING DATA##########################################
data = pd.read_csv(r'D:\Kaggle_Distracted_Driver\Pracrice images\full_64x64.csv'  , header = None)
print(data)

Xo = data.drop(data.columns[-1], axis=1)
print(Xo)

Yo = data[data.columns[-1]]
print(Yo)

X_train,X_test,y_train,y_test=train_test_split(Xo , Yo , test_size = 0.15, random_state = 100)
#print(X_train)
#print(y_train)
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

y_generalize = classifier.predict(finalX_gen)

print(y_generalize)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
