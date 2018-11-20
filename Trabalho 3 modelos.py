# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 02:23:47 2018

@author: mathe
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import BlindSearch as bs

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

"""
 - Modelos utilizados:
     
     - MLP
     - RandomForest
     - SVR
"""

def esperaenter():
    raw_input('Aperte Enter para prosseguir.' + '\n')

def processdata(datafile,dimension, valid = True):

    data = np.loadtxt(datafile)
    serie = pd.Series(data)
    laggedata = pd.concat([serie.shift(i) for i in range(dimension+1)],axis=1 )

    
    if valid == False: 
        
        #Treinamento 80%
        trainset = laggedata.iloc[dimension:int(np.floor(0.8*len( laggedata))),1:dimension+1]
        traintarget = laggedata.iloc[dimension:int(np.floor(0.8*len( laggedata))),0]
        
        #Teste 20%
        testset =  laggedata.iloc[int(np.floor(0.8*len( laggedata))):len( laggedata),1:dimension+1]
        testtarget =  laggedata.iloc[int(np.floor(0.8*len( laggedata))):len( laggedata),0]
    
        return (trainset,traintarget,testset,testtarget)
    
    
    if valid == True:
        
        trainindex=int(np.floor(0.7*len(data)))
        valindex=int(np.floor(0.8*len(data)))
         
        #Treinamento 60%
        trainset = laggedata.iloc[dimension:trainindex,1:dimension+1]
        traintarget = laggedata.iloc[dimension:trainindex,0]
        
        #Validação 20%
        valset = laggedata.iloc[trainindex:valindex,1:dimension+1]
        valtarget = laggedata.iloc[trainindex:valindex,0]
        
        #Teste 20%
        testset = laggedata.iloc[valindex:len(data),1:dimension+1]
        testtarget =  laggedata.iloc[valindex:len(data),0]
        
        return (trainset,traintarget,valset,valtarget,testset,testtarget)


def createmodel(trainset,traintarget,valset,valtarget, opmodel):
    
    if opmodel == 'MLP':
        (best_model, best_predicts, best_erro, best_param) = bs.gridMLP(trainset,traintarget,valset,valtarget)
        return best_model
    
    if opmodel == 'RF':
        (best_model, best_predicts, best_erro, best_param) = bs.gridRF(trainset,traintarget,valset,valtarget)
        return best_model
    
    if opmodel == 'SVR':
        (best_model, best_predicts, best_erro, best_param) = bs.gridSVR(trainset,traintarget,valset,valtarget)
        return best_model
        
    
    
def predict(testset,testtarget,model):
  
    predicts = model.predict(testset)
    erro = metrics.mean_squared_error(testtarget,predicts)

    return (predicts,erro)


"""
Dimension: ?

- Valor encontrado através da análise da autocorrelação
"""

dimension = 13
datafile = 'airlines2.txt'

(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)
"""
#MLP

(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)
mlp_reg = createmodel(data_set, data_target, val_set, val_target, 'MLP')
(pred_mlp, erro_mlp) = predict(pred_set, pred_target, mlp_reg)

mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs')
mlp.fit(data_set,data_target)
pred_mlp = mlp.predict(pred_set)
erro_mlp = metrics.mean_squared_error(pred_target, pred_mlp)


#RandomForest

(data_set, data_target,pred_set, pred_target) = processdata(datafile , dimension, valid=False)
rf = RandomForestRegressor()
rf.fit(data_set, data_target)
(pred_rf, erro_rf) = predict(pred_set, pred_target, rf)

#RandomForest valid=True
(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)
rf_reg = createmodel(data_set, data_target, val_set, val_target, 'RF')
(pred_rf, erro_rf) = predict(pred_set, pred_target, rf_reg)

"""
#SVR

(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)
svr_reg = createmodel(data_set, data_target, val_set, val_target,'SVR')
(pred_svr, erro_svr) = predict(pred_set, pred_target, svr_reg)


#Erros

#print 'Mlp ERRO: %.3f' %erro_mlp
#print 'RandomForest ERRO: %.3f' %erro_rf
print 'SVR ERRO: %.3f' %erro_svr

#Plotagem
x = range(len(pred_target))

plt.plot(x, pred_target, 'r--', label = 'Real')
#plt.plot(x, pred_mlp, label = 'MLP predicted')
#plt.plot(x, pred_rf, label = 'RandomForest predicted')
plt.plot(x, pred_svr, label = 'SVR predicted')
plt.legend()
























