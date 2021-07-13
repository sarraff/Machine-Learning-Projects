# Detecting-Distracted-Driver-&-Classification App

## [App-Link](https://Y3IHHGL6H72CQ2XG.anvil.app/WVESFDGA3J3OVVBZ55A4IH3L)

## Introduction
The goal of the app is to label whether the car driver is driving safe or performing any activity that might result in a accident or any harm to others. Since the project is based on multi-class classification problem, with a total of 10 classes including a class of ‘safe driving’, the app will also label the car driver class among the given 10 classes. 
The image below gives the 10 classes:
# ![Classes](Model_Codes/Images/Classes.png)


### How it is done 
The app is built using the platform [`Anvil`](https://anvil.works/build) which is a free Python-based drag-and-drop web app builder. It provide a full-stack web application where the client and server codes are in python. In the app, The client code will ask the user for the image as input and will send the image data as a string to the server. Now, the server will resize the image, convert it into input format, pass it through the computed model, and will predict the result. After prediction, the server will send the string to the client code and the client code will output the status and class of the input image. 

### Prerequisites before running the app
1. Compile model beforehand and save model weights in a file.
2. Load all the weights in a new defined structured parameter ( structure same as model which is being defined to classify the results)
```
model.load_weights('_path_to_file_having_weights_')
```
3. Install the required library
```
!pip install anvil-uplink
```
4. Define server & connect server with the client using key.
```
import anvil.server
anvil.server.connect("CE7B4DRR3AXDILXG2IZ6OJ6L-Y3IHHGL6H72CQ2XG")
```
5.  Define server side code below the request
```
@anvil.server.callable
def predict_class(file)
    return result
```
6. Activate the server side link to accept the client request
```
anvil.server.wait_forever()
```