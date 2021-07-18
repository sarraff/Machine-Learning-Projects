import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import cv2
import glob

# Train data on sobel processing file and save data into Sobel_train.csv
data = pd.read_csv(r'/content/drive/MyDrive/BTP_Driver_Distraction/Sobel_train.csv'  , header = None)
print(data)

Xo = data.drop(data.columns[-1], axis=1)
print(Xo)

Yo = data[data.columns[-1]]
print(Yo)

#X_train,X_test,y_train,y_test=train_test_split(Xo , Yo , test_size = 0.15, random_state = 50)
#print(X_train)
#print(y_train)
classifier = DecisionTreeClassifier()
classifier.fit(Xo,Yo)
#y_pred = classifier.predict(X_test)
#print("Split Accuracy:",metrics.accuracy_score(y_test,y_pred))
