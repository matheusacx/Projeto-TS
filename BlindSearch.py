# -*- coding: utf-8 -*-


import numpy as np
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn import svm
    
def gridSVR(X_train,Y_train,X_val,Y_val):

    C_r = range(8,10)
    Epsilon_r = range(-5,8)
    Gamma_r = range(-5,8)
    
    bestModel = 1
    bestError = 9999999999999999
    bestPredicts = 1
    bestParam = (0,0,0)
    
    for c in C_r:
        for e in Epsilon_r:
            for g in Gamma_r:
                model = svm.SVR(kernel = 'linear', C=10**c,gamma=10**g,epsilon=10**e)
                model.fit(X_train,Y_train)
                predicts = model.predict(X_val)
                erro = metrics.mean_squared_error(Y_val,predicts)
                
                if erro < bestError:
                    bestError = erro
                    bestModel = model
                    bestPredicts = predicts
                    bestParam = (c,g,e)
                    print bestError
                    print bestParam
                    
    return (bestModel,bestPredicts,bestError,bestParam)


def gridRF(X_train,Y_train,X_val,Y_val):
    
    """
    Observação: Alto custo de CPU
    """
    
    n_estimators = [2,4,8]
    max_features = ['auto','sqrt','log2',None]
    min_samples_leaf = np.linspace(0.1, 0.5, 3)
    min_samples_split = np.linspace(0.1, 1.0, 3)
    bootstrap = [True,False]
    
    best_erro = 9999999999
    
    for nest in n_estimators:
        for mf in max_features:
            for msl in min_samples_leaf:
                for msp in min_samples_split:
                    for bts in bootstrap:
                    
                        rf = RandomForestRegressor(n_estimators = nest, max_features = mf, min_samples_leaf = msl, min_samples_split = msp, bootstrap = bts)
                        rf.fit(X_train, Y_train)
                        
                        predict = rf.predict(X_val)
                        erro = metrics.mean_squared_error(Y_val, predict)
                        
                        if erro < best_erro:
                            
                            best_erro = erro
                            best_model = rf
                            best_predicts = predict
                            best_param = (nest,mf,msl,msp)
                            
                            print best_param
                            print best_erro
                            
            
    return (best_model, best_predicts, best_erro, best_param)
                        
                        
                    
def gridMLP(X_train,Y_train,X_val,Y_val):      
                     
    """
    Melhores parametros encontrados do laço for abaixo:
        
    ('relu', 'lbfgs', 0.00055)
    """
    
    activation = ['relu','tanh']
    solver = ['lbfgs']
    alpha = np.linspace(0.0001, 0.001, 8)
    
    best_erro = 999999999

    
    for act in activation:
        for slv in solver:
            for aph in alpha:
                mlp = MLPRegressor(activation = act, solver = slv, alpha = aph)
                mlp.fit(X_train, Y_train)
                
                predict = mlp.predict(X_val)
                erro = metrics.mean_squared_error(Y_val, predict)
                
                if erro < best_erro:
                    
                    best_erro = erro
                    best_model = mlp
                    best_param = (act, slv, aph)
                    best_predicts = predict
                    
                    #print best_param
                    #print best_erro
                    
    
    return (best_model, best_predicts, best_erro, best_param)
                        

def gridKN(X_train,Y_train,X_val,Y_val):

    n_neighbors = np.linspace(1,10,10)
    weights = ['uniform','distance']
        
    best_erro = 999999999

    for ngh in n_neighbors:
        for w in weights:
            kn = KNeighborsRegressor(n_neighbors = ngh, weights = w, algorithm = 'auto', p = 2)
            kn.fit(X_train,Y_train)

            predict = kn.predict(X_val)
            erro = metrics.mean_squared_error(Y_val, predict)

            if erro < best_erro:

                best_erro = erro
                best_model = kn
                best_param = (ngh, w)
                best_predicts = predict

    return (best_model, best_predicts, best_erro, best_param)

