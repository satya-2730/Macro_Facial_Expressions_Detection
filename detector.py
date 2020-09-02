#importing the necessory libraries
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import cv2
from model import MacroFacialExpressionsDetection
import numpy as np
import tensorflow as tf
import threading
import os
import sys,time

config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.15
tf_session = tf.compat.v1.Session(config=config)
set_session(tf_session)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #path to the haarcascade xml file for the frontal face
model = MacroFacialExpressionsDetection("model.json", "model_weights.h5") #path to you json file as well as the weights file
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0) #initializing the webcam 


class MacroFacialExpressionsDetection(object):

    EXPRESSIONS_LIST = ["Angry", "Disgust","Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.compile()
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.predictions = self.loaded_model.predict(img)
        return MacroFacialExpressionsDetection.EXPRESSIONS_LIST[np.argmax(self.predictions)]


def MacroExpressionDetection(any,any1):

	while True:

		ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.2, 5)
		
		for (x, y, w, h) in faces:
		
			img = gray[y:y+h, x:x+w]
			roi = cv2.resize(img, (48, 48))
			prediction_model = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
			cv2.putText(frame, prediction_model, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	 
		cv2.imshow('Macro Expression Detector', frame)

		if cv2.waitKey(1) == ord('q'):
			print("\n Exiting")
			break
			
	cv2.destroyAllWindows()

expressionThread = threading.Thread(target=MacroExpressionDetection, args=('any1','any2')) 
expressionThread.start()
