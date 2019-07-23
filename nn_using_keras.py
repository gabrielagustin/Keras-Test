# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

"""

import pandas as pd

path = "/media/gag/Datos/Estancia_Italia_2018/Trabajo_Sentinel_NDVI_CONAE/Modelo/"

fileTraining = "DataSet_Training_1km.csv"
fileTest = "DataSet_Test_1km.csv"
data = pd.read_csv(path + fileTraining, sep=',', decimal=",")
print ("Numero muestras training: " + str(len(data)))


dataTest = pd.read_csv(path + fileTest, sep=',', decimal=",")
print ("Numero muestras test: " + str(len(dataTest)))