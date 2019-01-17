#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:06:16 2019

@author: gaurav gaurav
@team : predict11
"""


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



#########################################################################################

def make_Predictions(Saved_model,X_test):
    Y_pred = Saved_model.predict(X_test)
    print(Y_pred)
    return Y_pred




#########################################################################################
### Predictions plot graph
### Accuracy graph
def plot_Accuracy_Graph(Y_test, Y_pred):
    y_original = Y_test[50:100]
    y_predicted = Y_pred[50:100]
    
    ### importing matplotlib
    import matplotlib.pyplot as plt

    plt.plot(y_original, 'r')
    plt.plot(y_predicted, 'b')
    plt.ylabel('predicted-b/original-r')
    plt.xlabel('n')
    plt.legend(['predicted', 'original'], loc='upper left')

    plt.show()



#########################################################################################
### Accure Scores
def accuracy_Score(Y_test, Y_pred):
    import sklearn.metrics
    ### Calculating the Varience Score
    res1 = sklearn.metrics.explained_variance_score(Y_test, Y_pred)
    print("Varience Score is : ",res1)






def main():
    
    X_train, X_test, Y_train, Y_test = get_Data()

    
    ### choose the best checkpoint
    weights_file = 'checkpoints/Weights-010--71248.10666.hdf5' 
    

    ### Load the saved Weights
    Saved_model = build_Model(X_train, X_test, Y_train, Y_test)
    Saved_model.load_weights(weights_file) # load it
    Saved_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


    ### make predictions
    Y_pred = make_Predictions(Saved_model, X_test)

    ### Accuracy Score
    plot_Accuracy_Graph(Y_test, Y_pred)
    accuracy_Score(Y_test, Y_pred)

    pass





if __name__ == '__main__' :
    main()