#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:57:06 2018

@author: gaurav gaurav
@team : predict11
"""


##################################################################################
### Importing libraries
from keras.callbacks import ModelCheckpoint


################################################################################
### importing Data
#import preprocessing
def get_Data():
    from preprocessing import X_train, X_test, Y_train, Y_test
    #print(X_train)
    #print(X_test)
    #print(Y_train)
    #print(Y_test
    return X_train, X_test, Y_train, Y_test



#################################################################################
### Building the Model
def build_Model(X_train, X_test, Y_train, Y_test):
    #import keras
    from keras.models import Sequential
    #from keras.layers import Dense, Activation, Flatten
    from keras.layers import Dense

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

    return R_model



################################################################################
### printing the model summary
def get_Model_Summary(R_model):    
    ### print the summary
    R_model.summary()



#########################################################################################
### Save model Weights
def save_Model(R_model):    
    SaveFileName = 'SavedModel/Saved_model_weights.h5'
    Saved_model = R_model.save_weights(SaveFileName)
    type(R_model)



def make_Predictions(Saved_model,X_test):
    Y_pred = Saved_model.predict(X_test)
    print(Y_pred)
    return Y_pred



# list all data in history
# summarize history for loss
def plot_Loss(history):
    ### importing matplotlib
    import matplotlib.pyplot as plt
        
    #dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    

# summarize history for accuracy
def plot_Accuracy(history):
    ### importing matplotlib
    import matplotlib.pyplot as plt
    
    #dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])
    print(history.history.keys())
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()





#######################################################################################
### parameters tuning and optimisation
#fine-tuning
#grid search



def main():

    ### importing Data
    #import preprocessing
    X_train, X_test, Y_train, Y_test = get_Data()


    ### Building the Model
    R_model = build_Model(X_train, X_test, Y_train, Y_test)

    
    ### Saving the checkpoints
    checkpoint_name = 'checkpoints/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
    callbacks_list = [checkpoint]
            
    ### training the model    
    history = R_model.fit(X_train, Y_train, epochs=20, batch_size=8, validation_split = 0.2, callbacks=callbacks_list)

    print(type(history))
    print(history.history.keys())
    print(history.history.values())
    

    ### save the model
    save_Model(R_model)


    plot_Accuracy(history)
    plot_Loss(history)


    print("Program Exicuted Succesfully")




if __name__ == "__main__":
    main()