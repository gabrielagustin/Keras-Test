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

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy


def create_simple_nn():  
    # create model, as a regressor
    # Here you should use something like Scikit-learn to find the best architecture
    model = Sequential()
    model.add(Dense(8, input_dim=4, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform'))
    return model


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


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

snn_model = create_simple_nn()  
snn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae','accuracy'])  

snn = snn_model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test), shuffle=True)  

plt.figure(0)  
plt.plot(snn.history['mean_absolute_error'],'r')  
plt.plot(snn.history['val_mean_absolute_error'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("MSE")  
plt.title("Training MSE vs Validation MSE")  
plt.legend(['train','validation'])

plt.figure(1)  
plt.plot(snn.history['loss'],'r')  
plt.plot(snn.history['val_loss'],'g')  
plt.xticks(np.arange(0, 11, 2.0))  
plt.rcParams['figure.figsize'] = (8, 6)  
plt.xlabel("Num of Epochs")  
plt.ylabel("Loss")  
plt.title("Training Loss vs Validation Loss (MSE)")  
plt.legend(['train','validation'])

plt.show()  

