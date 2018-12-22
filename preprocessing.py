#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 07:32:26 2018
@author: gaurav
@team : predict11

"""

#####################################################################################
### importing the pandas and numpy
import pandas as pd
import numpy as np


### Features column Names
col_Names = ["date","price" ,"bedrooms" ,"bathrooms","sqft_living", "sqft_lot",
             "floors", "waterfront" ,"view","condition","grade","sqft_above","sqft_basement",
             "yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15" ]



###############################################################################
### filePath to open dataset
fileName = "data/kc_house_data.csv"

### Read The csv
df = pd.read_csv(fileName,usecols=col_Names)


### Slice the dates in column to year
df["date"] = df["date"].str[0:4]


### change dataframe datatypes to numeric
df = df.apply(pd.to_numeric)


### Calculate the age of house
### Calculate the [ current date - year of Built ]
df["h_age"] = df["date"]-df["yr_built"]


### Function to find tehe renovated age of house
### if zero found : keep the same,  else: current_date - date_of_renovated
def renovated(d):
    if d == 0:
        d = 0
    else:
        d = 2015-d
    return d


## Function call to apply renovated date
df["r_age"] = df['yr_renovated'].apply(renovated)


### transfer label to the end and drop prev label
df["orig_price"] = df["price"]

df = df.drop("date",1)
df = df.drop("price",1)
df = df.drop("yr_built",1)
df = df.drop("yr_renovated",1)


### convert to numeric
df = df.apply(pd.to_numeric)



############################################################################
### from sk learn import labeleEncoder and  one hot encoder

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

### labelEncoder
labelencoder_X = LabelEncoder()


### convert zip codes into numbers
df["zipcode"] = labelencoder_X.fit_transform(df["zipcode"])



#################################################################
#### to check cateogorical features
labelencoder_X.fit(df['zipcode'])
labelencoder_X.classes_



###################################################################
### converting into numpy arreys
X = df.iloc[:,0:-1].values
Y = df.iloc[:,18].values



### one hot encoding of the categorical features
onehotencoder = OneHotEncoder(categorical_features = [11])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
print(X)


####################################################################
### Saving the intermediate data in files
dataF_X = pd.DataFrame(X)
dataF_Y = pd.DataFrame(Y)

### filenames to save data
filePath = "data/encoded_X.csv"
filePath = "data/encoded_Y.csv"

dataF_X.to_csv(filePath, index=False)
dataF_Y.to_csv(filePath, index=False)



###################################################################
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




#################################################################
### Display the data

#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test)

#df.info
#df.count()
#df.columns
#print(df.count)
#print(df.dtypes)
#print(df.columns)
#print(df.head())
#print(df.info)

