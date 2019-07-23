# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

"""

import lectura

path = "/media/gag/Datos/Estancia_Italia_2018/Trabajo_Sentinel_NDVI_CONAE/Modelo/"

fileTraining = "DataSet_Training_1km.csv"
fileTest = "DataSet_Test_1km.csv"
data = lectura.read(path + fileTraining)
print ("Numero muestras training: " + str(len(data)))

dataTest = lectura.read(path + fileTest)
print ("Numero muestras test: " + str(len(dataTest)))



## se debe normalizar las variables pero el problema es que debo hacerlo a la totalidad de las muestras
## y pierdo relación entre pixeles

