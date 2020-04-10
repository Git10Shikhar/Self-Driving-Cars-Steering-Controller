#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:35:58 2020

@author: vivek
"""

import tensorflow as tf

import cv2
import os
import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from itertools import islice
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout


# DATA PREPROCESSING ------------------>

DATA_FOLDER = 'dataset'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

def preprocess(img):
    
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    return resized

def return_data():

    X = []
    y = []
    features = []

    with open(TRAIN_FILE) as file:
        for line in islice(file,None):
            
            path ,angle= line.strip().split()
            full_path = os.path.join(DATA_FOLDER, path)
            X.append(full_path)
            y.append(float(angle) * scipy.pi / 180)

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(preprocess(img))

    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')

    f=open("features", "wb")
    pickle.dump(features, f, protocol=4)
    f.close
    
    f=open("labels", "wb")
    pickle.dump(labels, f, protocol=4)
    f.close

return_data()


# TRAIN ------------------------------->


def CNN():

    model = Sequential()
    
    
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu',input_shape=(100,100,1)))
    
    model.add(MaxPooling2D((2, 2), padding='valid'))
    
    model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
  
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3),activation='relu', padding='same'))
  
    model.add(MaxPooling2D((2, 2), padding='valid'))
    
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
   
    model.add(MaxPooling2D((2, 2), padding='valid'))
    
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
  
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(1))
    
    
    model.compile(optimizer='adam',loss="mse",metrics=['acc'])
    
    filepath = "Autopilot.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    
    model.summary()


    return model, callbacks_list

        

def Load():
    
    f=open("features", "rb")
    features = np.array(pickle.load(f))
    f.close()
    
    f=open("labels", "rb")
    labels = np.array(pickle.load(f))
    f.close()
    
    return features, labels


def main():
    features, labels = Load()
    
    features, labels = shuffle(features, labels)
    
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,test_size=0.3)
    
    
    train_x = train_x.reshape(train_x.shape[0], 100, 100, 1)
    
    test_x = test_x.reshape(test_x.shape[0], 100, 100, 1)
    
    model, callbacks_list = CNN()
    
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, batch_size=32, callbacks=callbacks_list)
    
    model.summary()
    
    model.save('Autopilot.h5')

main()





# RUN ON VIDEO  ------------------------------>



model = load_model('Autopilot.h5')

def op_image_processing(img):
    
    img=cv2.resize(img,(100,100))
    img=np.array(img,dtype=np.float32)
    img=np.reshape(img,(-1,100,100,1))
    return img

def prediction(model,img):
    
    processed=op_image_processing(img)
    steer_angle=float(model.predict(processed, batch_size=1))
    steer_angle=steer_angle * 60
    return steer_angle


steer = cv2.imread('steering.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

vid = cv2.VideoCapture('run.mp4')
while (vid.isOpened()):
    
    ret, frame = vid.read()
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    steering_angle = prediction(model, gray)
    print(steering_angle)
    
    cv2.imshow('frame', cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(
        steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
    
    
