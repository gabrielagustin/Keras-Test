# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Created on Mon Jul 22 16:16:04 2019

@author: gag

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import seaborn as sns


###----------------------------
#### convert a range to another range
#OldRange = (OldMax - OldMin)  
#NewRange = (NewMax - NewMin)  
#NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
###----------------------------


def normalizado(c):
    """
    function that normalizes the variables between 0 and 1
    """
    min = np.min(c)
    max = np.max(c)
    new = (c -min)/(max-min)
    OldRange = (max  - min)
    NewRange = (1 - 0.1)
    new = (((c - min) * NewRange) / OldRange) + 0.1
    return new
    


def read(file):
    data = pd.read_csv(file, sep=',', decimal=",")
    #print (data)
    # print ("Numero inicial de muestras: " + str(len(data)))
    ### a cada variable se la convierte a al rango utilizado

    data.SM_SMAP = data.SM_SMAP *100
    data.PP = data.PP * 0.1
    ### Temperatura de SMAP
    data.T_s = data.T_s -273.15
    ### Temperatura de MODIS
    # fig, ax = plt.subplots()
    # sns.distplot(data.T_s)
    dataNew = data[(data.T_s_modis > 250)]
    data = dataNew
    data.T_s_modis = data.T_s_modis -273.15
    # fig, ax = plt.subplots()
    # sns.distplot(data.T_s_modis)    

    ### Evapotranspiración de MODIS
#    fig, ax = plt.subplots()
#    sns.distplot(data.Et)
    dataNew = data[(data.Et < 3000)]
    data = dataNew
#    fig, ax = plt.subplots()
#    sns.distplot(data.Et)
    
    data.Et = ((data.Et*0.1)/8)/0.035
    # fig, ax = plt.subplots()
    # sns.distplot(data.Et)
    # plt.show()
    
    # print ("Numero de muestras: " + str(len(data)))


    ## se filtra el rango de valores de backscattering
    perc5Back = math.ceil(np.percentile(data.Sigma0,1))
    # print("percentile back 5: " + str(perc5Back))
    perc90Back = math.ceil(np.percentile(data.Sigma0, 99))
    # print("percentile back 95: " + str(perc90Back))
    dataNew = data[(data.Sigma0 > -18) & (data.Sigma0 < -4)]
    data = dataNew
    # print("Numero de muestras: " + str(len(data)))    

    
    


    ### se filtra el rango de valores de HR
#    perc5HR = math.ceil(np.percentile(data.HR,0))
##    print "percentile HR 5: " + str(perc5HR)
#    perc90HR = math.ceil(np.percentile(data.HR, 99))
#    print "percentile HR 95: " + str(perc90HR)
#    dataNew = data[(data.HR > 17.83) & (data.HR < 83.63)]
#    data = dataNew


    #print "Numero de muestras: " + str(len(data))

#    # se filtra el rango de valores de RSOILTEMPC
#    perc5Ts = math.ceil(np.percentile(data.T_s,0))
##    print ("percentile Ts 5: " + str(perc5Ts))
#    perc90Ts = math.ceil(np.percentile(data.T_s, 95))
##    print ("percentile Ts 95: " + str(perc90Ts))
#    dataNew = data[(data.T_s > perc5Ts) & (data.T_s < 25)]
#    data = dataNew
    #print ("Numero de muestras: " + str(len(data)))


    #se filtra el rango de valores de evapotransporacion
    #perc10Et = math.ceil(np.percentile(data.Et, 5))
    #print ("percentile Et 5: " + str(perc10Et))
    #perc90Et = math.ceil(np.percentile(data.Et, 75))
    #print ("percentile Et 90: " + str(perc90Et))
    #print ("Filtro por Et")
#    dataNew = data[(data.Et > 50) & (data.Et <= 450)]
#    data = dataNew
#    dataNew = data[(data.Et > 550) ]
#    data = dataNew
    ###-----------------------------------------------------------------------
    ### reglas para filtrar los datos
    ## se filtra el rango de valores de NDVI
    #dataNew = data[ (data.NDVI_30m_B > 0.1) & (data.NDVI_30m_B < 0.49)]
    dataNew = data[ (data.NDVI > 0.1) & (data.NDVI < 0.8)]
    data = dataNew
    # print ("Filtro por NDVI")
    # print ("Numero de muestras: " + str(len(data)))

    del data['NDVI']
    del data['Date']
    
    # print("-----------------------------------------------------------------------")
    # print(data.describe())
    # print("-----------------------------------------------------------------------")

#    print('---MLR---------------------------------------------------------------')
#    print ("Variables sin normalizar")
#    print(data.describe())
#    print('------------------------------------------------------------------')

    data.PP = normalizado(data.PP)
    data.Sigma0 = normalizado(data.Sigma0)
    data.T_s = normalizado(data.T_s)
    data.Et = normalizado(data.Et)
    data.T_s_modis = normalizado(data.T_s_modis)

    del data['T_s_modis']
    # del data['T_s']

    # print('------------------------------------------------------------------')
    # print(data.describe())
    # print('------------------------------------------------------------------')
    return data
