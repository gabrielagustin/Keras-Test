# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

# Function to create model, Multi-Layer Perceptron (Neural Network) modified as regressor

"""

from keras.models import Sequential
from keras.layers import Dense

def create_simple_nn(optimizer='adam', activation='relu'):  
    # create model, as a regressor
    model = Sequential()
    model.add(Dense(5, input_dim=4, kernel_initializer='uniform', activation=activation))
    model.add(Dense(6, kernel_initializer='uniform', activation=activation))
    model.add(Dense(1, kernel_initializer='uniform'))
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_squared_error'])  
    return model

