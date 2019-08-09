# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

# Function to create model, Multi-Layer Perceptron (Neural Network) modified as regressor

"""

from keras.models import Sequential
from keras.layers import Dense

def create_simple_nn(optimizer='adam'):  
    # create model, as a regressor
    model = Sequential()
    model.add(Dense(8, input_dim=4, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform'))
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_squared_error'])  
    return model

