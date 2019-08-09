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
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
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
Best: -6.827617 using {'optimizer': 'Adam'}
-7.211086 (0.631136) with: {'optimizer': 'SGD'}
-7.002374 (1.016755) with: {'optimizer': 'RMSprop'}
-8.298661 (1.424133) with: {'optimizer': 'Adagrad'}
nan (nan) with: {'optimizer': 'Adadelta'}
-6.827617 (1.192136) with: {'optimizer': 'Adam'}
-7.150356 (1.164060) with: {'optimizer': 'Adamax'}
-7.261933 (0.852300) with: {'optimizer': 'Nadam'}
"""









