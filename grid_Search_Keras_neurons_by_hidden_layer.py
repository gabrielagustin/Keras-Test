# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

Use scikit-learn to grid search to find the best parameters
    - Numbers of neurons by hidden layer 

"""

import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

import lectura

from keras.models import Sequential
from keras.layers import Dense

def create_simple_nn(optimizer='adam', activation='relu', neurons1=1, neurons2=1):  
    # create model, as a regressor
    model = Sequential()
    model.add(Dense(neurons1, input_dim=4, kernel_initializer='uniform', activation=activation))
    model.add(Dense(neurons2, kernel_initializer='uniform', activation=activation))
    model.add(Dense(1, kernel_initializer='uniform'))
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_squared_error'])  
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

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasRegressor(build_fn=create_simple_nn, batch_size=80, epochs=10, verbose=0)
# snn_model = nn_Model_Keras.create_simple_nn()  
# define the grid search parameters
neurons1 = np.arange(1,8,1)
neurons2 = np.arange(1,8,1)
param_grid = dict(neurons1=neurons1, neurons2=neurons2)
print(param_grid)
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
        'optimizer': 'Adam' just for default!!!
         batch_size=80, epochs=10
         activation functions = 'relu'

Best: -6.940726 using {'neurons1': 5, 'neurons2': 6}
-26.381092 (3.423127) with: {'neurons1': 1, 'neurons2': 1}
-20.570062 (10.200297) with: {'neurons1': 1, 'neurons2': 2}
-15.341615 (11.234250) with: {'neurons1': 1, 'neurons2': 3}
-13.872204 (7.367320) with: {'neurons1': 1, 'neurons2': 4}
-13.403936 (7.357169) with: {'neurons1': 1, 'neurons2': 5}
-7.747360 (1.009484) with: {'neurons1': 1, 'neurons2': 6}
-15.083940 (11.405193) with: {'neurons1': 1, 'neurons2': 7}
-26.381246 (3.423156) with: {'neurons1': 2, 'neurons2': 1}
-19.283514 (6.626469) with: {'neurons1': 2, 'neurons2': 2}
-13.109744 (7.588139) with: {'neurons1': 2, 'neurons2': 3}
-7.709576 (1.238220) with: {'neurons1': 2, 'neurons2': 4}
-14.096664 (7.216613) with: {'neurons1': 2, 'neurons2': 5}
-14.734232 (7.052353) with: {'neurons1': 2, 'neurons2': 6}
-7.726136 (1.348445) with: {'neurons1': 2, 'neurons2': 7}
-26.381235 (3.423152) with: {'neurons1': 3, 'neurons2': 1}
-7.768581 (1.201637) with: {'neurons1': 3, 'neurons2': 2}
-13.197095 (7.510347) with: {'neurons1': 3, 'neurons2': 3}
-13.038339 (7.568310) with: {'neurons1': 3, 'neurons2': 4}
-7.787880 (1.267978) with: {'neurons1': 3, 'neurons2': 5}
-19.022686 (6.995179) with: {'neurons1': 3, 'neurons2': 6}
-14.775355 (11.625495) with: {'neurons1': 3, 'neurons2': 7}
-19.044851 (6.963871) with: {'neurons1': 4, 'neurons2': 1}
-12.829669 (7.724511) with: {'neurons1': 4, 'neurons2': 2}
-14.857298 (11.566275) with: {'neurons1': 4, 'neurons2': 3}
-7.474805 (1.774071) with: {'neurons1': 4, 'neurons2': 4}
-7.388255 (0.708174) with: {'neurons1': 4, 'neurons2': 5}
-6.959149 (1.228794) with: {'neurons1': 4, 'neurons2': 6}
-14.708142 (11.673114) with: {'neurons1': 4, 'neurons2': 7}
-26.381221 (3.423173) with: {'neurons1': 5, 'neurons2': 1}
-13.962790 (7.313489) with: {'neurons1': 5, 'neurons2': 2}
-19.017768 (7.002124) with: {'neurons1': 5, 'neurons2': 3}
-7.296564 (1.098349) with: {'neurons1': 5, 'neurons2': 4}
-7.290928 (1.116122) with: {'neurons1': 5, 'neurons2': 5}
-6.940726 (1.030243) with: {'neurons1': 5, 'neurons2': 6}
-7.270795 (1.638029) with: {'neurons1': 5, 'neurons2': 7}
-13.638166 (7.553012) with: {'neurons1': 6, 'neurons2': 1}
-20.830015 (10.163070) with: {'neurons1': 6, 'neurons2': 2}
-20.580100 (10.502992) with: {'neurons1': 6, 'neurons2': 3}
-7.458323 (1.536502) with: {'neurons1': 6, 'neurons2': 4}
-14.556591 (11.779210) with: {'neurons1': 6, 'neurons2': 5}
-7.261567 (1.524116) with: {'neurons1': 6, 'neurons2': 6}
-7.125081 (1.315450) with: {'neurons1': 6, 'neurons2': 7}
-13.520828 (7.669182) with: {'neurons1': 7, 'neurons2': 1}
-7.164493 (1.174737) with: {'neurons1': 7, 'neurons2': 2}
-20.392753 (10.439637) with: {'neurons1': 7, 'neurons2': 3}
-20.620430 (10.448150) with: {'neurons1': 7, 'neurons2': 4}
-14.694527 (11.686022) with: {'neurons1': 7, 'neurons2': 5}
-14.502270 (11.821358) with: {'neurons1': 7, 'neurons2': 6}
-14.336532 (11.933569) with: {'neurons1': 7, 'neurons2': 7}


"""









