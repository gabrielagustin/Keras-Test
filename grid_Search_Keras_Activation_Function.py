# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

Use scikit-learn to grid search to find the best parameters
    - Optimization Algorithm

"""

import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

import nn_Model_Keras
import lectura

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

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=nn_Model_Keras.create_simple_nn, batch_size=80, epochs=10, verbose=0)
# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
# grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


""" 
Using: 
        batch_size=80, epochs=10
        'optimizer': 'Adam'
        
Best: -6.938906 using {'activation': 'relu'}
-22.206605 (3.418575) with: {'activation': 'softmax'}
-7.055343 (0.986810) with: {'activation': 'softplus'}
-8.841389 (2.226597) with: {'activation': 'softsign'}
-6.938906 (1.365278) with: {'activation': 'relu'}
-8.873299 (2.232288) with: {'activation': 'tanh'}
-8.885092 (2.237665) with: {'activation': 'sigmoid'}
-9.081818 (2.450853) with: {'activation': 'hard_sigmoid'}
-7.272140 (0.823711) with: {'activation': 'linear'}
"""









