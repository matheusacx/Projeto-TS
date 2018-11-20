# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 00:31:44 2018

@author: mathe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import BlindSearch as bs

def processdata(datafile,dimension):
    
    """
    Funcao que separa os dados da seguinte forma:
        
    Treinamento: 60%
    Validação: 20%
    Teste: 20%
    """
    
    data = np.loadtxt(datafile)
    serie = pd.Series(data)
    laggedata = pd.concat([serie.shift(i) for i in range(dimension+1)],axis=1 )

    trainindex=int(np.floor(0.6*len(data)))
    valindex=int(np.floor(0.8*len(data)))
    
    #Treinamento
    trainset = laggedata.iloc[dimension:trainindex,1:dimension+1]
    traintarget = laggedata.iloc[dimension:trainindex,0]
    
    #Validação
    valset = laggedata.iloc[trainindex:valindex,1:dimension+1]
    valtarget = laggedata.iloc[trainindex:valindex,0]
    
    #Testes
    testset = laggedata.iloc[valindex:len(data),1:dimension+1]
    testtarget =  laggedata.iloc[valindex:len(data),0]
    
    return (trainset,traintarget,valset,valtarget,testset,testtarget)

def createRF(trainset, traintarget, valset, valtarget):
    """
    Funcao que cria o 'melhor' modelo de regressão utilizando
    o algoritmo RandomForest.
    """
    
    (best_model, best_predicts, best_erro, best_parameters) = bs.gridRF(trainset, traintarget, valset, valtarget)
    
    return (best_model, best_predicts, best_erro, best_parameters)

def predict(testset,testtarget,model):
    
    predicts = model.predict(testset)
    erro = metrics.mean_squared_error(testtarget,predicts)
    x= range(len(testtarget))
    plt.plot(x,predicts,'b--',label='predicts')
    plt.plot(x,testtarget,'r',label='real')
    plt.legend()
    return (predicts,erro)

    



(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata('airlines2.txt',13)
(best_model, best_predicts, best_erro, best_parameters) = createRF(data_set, data_target, val_set, val_target)
(predicts,erro) = predict(pred_set, pred_target, best_model) 





