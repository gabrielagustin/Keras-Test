# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

# Function to create model, Multi-Layer Perceptron (Neural Network) modified as regressor

"""


def create_simple_nn():  
    # create model, as a regressor
    model = Sequential()
    model.add(Dense(8, input_dim=4, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform'))
    return model

