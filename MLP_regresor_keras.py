# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

MLP as regressor using Keras. 
Knowing that KerasRegressor is a Wrappers for the Scikit-Learn API !!!
"""

import lectura

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf  
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def build_regressor():
    regressor = Sequential()
    ### falta ver como optimizar el numero de nueronas por capas ocultas usando Keras
    regressor.add(Dense(units=9, input_dim=4))
    regressor.add(Dense(units=9))
    regressor.add(Dense(units=1))
    # regressor.compile(optimizer='sgd', loss='mean_squared_error',  metrics=['acc', 'mse'])
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])
    return regressor



path = "/home/gag/MyProjects/Keras-Test/"

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
# y_train = 10**(y_train)
del dataTraining["SM_SMAP"]
X_train = dataTraining

y_test = np.array(dataTest["SM_SMAP"])
# y_test = 10**(y_test)
del dataTest["SM_SMAP"]
X_test = dataTest



regressor = KerasRegressor(build_fn=build_regressor, batch_size=50,epochs=10)
results=regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')

plt.show()