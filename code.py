# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:41:00 2019

@author: Rayyan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# splitting dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Encoding categorical data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

# making the confussion matrix => used to make a table of predicted and actual result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# 
from matplotlib.colors import ListedColormap
X_set , y_set =X_train,y_train