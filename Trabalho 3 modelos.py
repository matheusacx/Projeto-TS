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

"""
 - Modelos utilizados:
     
     - MLP
     - AutoRegressivo
     - Kneighbors
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

    if opmodel == 'KN': 
        (best_model, best_predicts, best_erro, best_param) = bs.gridKN(trainset,traintarget,valset,valtarget)
        return best_model
        
    
def predict(testset,testtarget,model):
  
    predicts = model.predict(testset)
    erro = metrics.mean_squared_error(testtarget,predicts)

    return (predicts,erro)

def createmodelAR(trainset,traintarget,dimension):    
    trainset[dimension+1]=1 
    coefs = np.linalg.pinv(trainset).dot(traintarget)
    #print coefs
    return coefs

def predictAR(coefs,testset,testtarget,dimension):
    
    testset[dimension+1]=1
    predicts = testset.dot(coefs)
    erro = metrics.mean_squared_error(testtarget, predicts)

    return (predicts, erro)

dimension = 13
datafile = 'airlines2.txt'

(data_set, data_target, val_set, val_target, pred_set, pred_target) = processdata(datafile , dimension)

#MLP

mlp_reg = createmodel(data_set, data_target, val_set, val_target, 'MLP')
(pred_mlp, erro_mlp) = predict(pred_set, pred_target, mlp_reg)

#Kneighbors

kn_reg = createmodel(data_set, data_target, val_set, val_target, 'KN')
(pred_kn, erro_kn) = predict(pred_set, pred_target, kn_reg)

#AutoRegressivo

(train_set, train_target, test_set, test_target) = processdata(datafile, dimension, valid=False)
coefs = createmodelAR(train_set, train_target, dimension)
(pred_AR, erro_AR) = predictAR(coefs, test_set, test_target, dimension) 

#Modelo Híbrido - Média

pred_hmean = (pred_mlp + pred_AR + pred_kn)/3
erro_hmean = metrics.mean_squared_error(test_target, pred_hmean)

#Modelo Híbrido - Mediana

pred_hmedian = np.median((pred_mlp, pred_AR, pred_kn), axis = 0)
erro_hmedian = metrics.mean_squared_error(test_target, pred_hmedian)

#Erros

print 'Modelos RNA'
print 'MlP Erro: %.4f' %erro_mlp
print 'AR  Erro: %.4f' %erro_AR
print 'KN  Erro: %.4f +' + '\n' %erro_kn

print 'Modelos Híbridos' + '\n'
print 'Hmean Erro: %.4f' %erro_hmean 
print 'Hmedian Erro: %.4f' %erro_hmedian


#Plotagem
x = range(len(pred_target))

#3 Modelos
plt.plot(x, pred_target, 'r--', label = 'Real')
plt.plot(x, pred_mlp, label = 'MLP predicted')
plt.plot(x, pred_AR, label = 'AR predicted')
plt.plot(x, pred_kn, label = 'KN predicted')
plt.legend()
plt.figure

#Modelos Híbridos
plt.plot(x, pred_target, 'r--', label = 'Real')
plt.plot(x, pred_hmean, label = 'Hmean')
plt.plot(x, pred_hmedian, label = 'Hmedian')
plt.legend()
plt.figure()





















