# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

MLP as regressor using Keras

"""


import lectura

import numpy as np

import tensorflow as tf  
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def build_regressor():
    regressor = Sequential()
    ### falta ver como optimizar el numero de nueronas por capas ocultas usando Keras
    regressor.add(Dense(units=8, input_dim=4))
    regressor.add(Dense(units=8))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='sgd', loss='mean_squared_error',  metrics=['acc', 'mse'])
    return regressor



path = "/media/gag/Datos/Estancia_Italia_2018/Trabajo_Sentinel_NDVI_CONAE/Modelo/"

fileTraining = "DataSet_Training_1km.csv"
fileTest = "DataSet_Test_1km.csv"

### OJO!! ambos conjuntos de datos estan normalizados utilizando sus valores 
### maximos y minimos relativos, NO del set completo. MEJORAR!!!

dataTraining = lectura.read(path + fileTraining)
print(list(dataTraining.columns))
print ("Numero muestras training: " + str(len(dataTraining)))

dataTest = lectura.read(path + fileTest)
print ("Numero muestras test: " + str(len(dataTest)))


### se separan ambos conjuntos de entranamiento y prueba, en entrada (X) y salida (y)
y_train = np.array(dataTraining["SM_SMAP"])
y_train = 10**(y_train)
del dataTraining["SM_SMAP"]
X_train = dataTraining

yTest = np.array(dataTest["SM_SMAP"])
yTest = 10**(yTest)
del dataTest["SM_SMAP"]
xTest = dataTest



regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=20)
results=regressor.fit(X_train,y_train)



