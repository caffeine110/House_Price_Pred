#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:57:06 2018

@author: gaurav
@team : predict11
"""


##################################################################################
### Importing libraries
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten
from keras.layers import Dense

#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_absolute_error 

#from matplotlib import pyplot as plt
#import pandas as pd
#import numpy as np
#from xgboost import XGBRegressor


################################################################################
### importing Data
#import preprocessing
from preprocessing import X_train, X_test, Y_train, Y_test
#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test


#################################################################################
### Building the Model

### Sequential Layer
R_model = Sequential()

### The Input Layer
R_model.add(Dense(86, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

### The Hidden Layers :
R_model.add(Dense(86, kernel_initializer='normal',activation='relu'))
R_model.add(Dense(64, kernel_initializer='normal',activation='relu'))
R_model.add(Dense(32, kernel_initializer='normal',activation='relu'))

### The Output Layer :
R_model.add(Dense(1, kernel_initializer='normal',activation='linear'))


### Compile the network:
R_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

### print the summary
R_model.summary()


###########################################################################################
### Saving the model
checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]



###########################################################################
### training the model

R_model.fit(X_train, Y_train, epochs=500, batch_size=16, validation_split = 0.2, callbacks=callbacks_list)

#Weights-171--67565.87308
#wights_file = 'Weights-478--18738.19831.hdf5' # choose the best checkpoint 



#########################################################################################
### Save model Weights
"""
SaveFileName = 'SavedModel/Saved_model_weights.h5'
Saved_model = R_model.save_weights(SaveFileName)
#type(R_model)

weights_file = 'checkpoints/Weights-170--67364.26215.hdf5' # choose the best checkpoint 

### Load the saved Weights
#Saved_model = keras.engine.sequential.Sequential
Saved_model.load_weights(weights_file) # load it
Saved_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
"""

Y_pred = R_model.predict(X_test)

print(Y_pred)
print(Y_test)




#########################################################################################
### Predictions plot graph
### Accuracy graph
y_original = Y_test[50:100]
y_predicted = Y_pred[50:100]

### importing matplotlib
import matplotlib.pyplot as plt

plt.plot(y_original, 'r')
plt.plot(y_predicted, 'b')

plt.show()


#########################################################################################
### Accure Scores
import sklearn.metrics

### Calculating the Varience Score
res1 = sklearn.metrics.explained_variance_score(Y_test, Y_pred)
print("Varience Score is : ",res1)



#######################################################################################
### parameters tuning and optimisation
#fine-tuning
#grid search


