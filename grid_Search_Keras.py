# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

Use scikit-learn to grid search to find the best parameters
    - the batch size and epochs

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
model = KerasRegressor(build_fn=nn_Model_Keras.create_simple_nn, verbose=0)
# snn_model = nn_Model_Keras.create_simple_nn()  
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size,
                     epochs=epochs)
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
Best: -6.729004 using {'batch_size': 80, 'epochs': 10}
-7.307594 (0.883138) with: {'batch_size': 10, 'epochs': 10}
-7.760955 (0.626575) with: {'batch_size': 10, 'epochs': 50}
-7.377692 (0.697249) with: {'batch_size': 10, 'epochs': 100}
-7.530388 (0.884904) with: {'batch_size': 20, 'epochs': 10}
-7.414847 (0.748094) with: {'batch_size': 20, 'epochs': 50}
-7.594826 (0.812177) with: {'batch_size': 20, 'epochs': 100}
-7.256536 (1.021333) with: {'batch_size': 40, 'epochs': 10}
-7.404524 (0.856856) with: {'batch_size': 40, 'epochs': 50}
-7.402548 (0.808072) with: {'batch_size': 40, 'epochs': 100}
-7.063264 (0.953157) with: {'batch_size': 60, 'epochs': 10}
-7.367025 (0.715037) with: {'batch_size': 60, 'epochs': 50}
-7.382340 (0.838285) with: {'batch_size': 60, 'epochs': 100}
-6.729004 (1.178125) with: {'batch_size': 80, 'epochs': 10}
-7.442276 (0.800250) with: {'batch_size': 80, 'epochs': 50}
-7.717917 (0.728243) with: {'batch_size': 80, 'epochs': 100}
-6.994207 (1.183363) with: {'batch_size': 100, 'epochs': 10}
-7.294369 (0.860499) with: {'batch_size': 100, 'epochs': 50}
-7.870209 (0.850234) with: {'batch_size': 100, 'epochs': 100}
"""









