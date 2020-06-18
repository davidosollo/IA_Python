#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:04:50 2020

"Clasifica Personas con diabetes

@author: davidosollo
"""
#Init Libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Clase Perceptron
class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.errorMin = 99999999
        

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi,0))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            if self.errorMin > errors:    #Get the best
                 self.wMin_ = self.w_
            self.errors_.append(errors)
        return self

    def net_input(self, X, Op):
        """Calculate net input"""
        if Op == 0:
            return np.dot(X, self.w_[1:]) + self.w_[0]
        else:
            return np.dot(X, self.wMin_[1:]) + self.wMin_[0]
            

    def predict(self, X, Op):
        """Return class label after unit step"""
        return np.where(1/(1+np.exp(-self.net_input(X,Op)))> 0.5, 1, 0)
        #return np.where(self.net_input(X,Op) >= 0.0, 1, 0)
    
    def predict_proba(self, X):
        return 1/(1+np.exp(-self.net_input(X,1)))
        
"""Vamos a leer una rchivo llamado diabetes.csv """
#df = pd.read_csv('diabetes.csv')
df = pd.read_csv('cancer.csv')
X = np.asanyarray(df.drop(columns=['Class']))
Y = np.asanyarray(df[['Class']]).T.ravel()
print(X.shape)
print(Y.shape)

p, n = X.shape
for i in range(n):
  #X[i,:] = (X[i,:]-X[i,:].min())/(X[i,:].max()-X[i,:].min())
  X[:,i] = (X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())
  

""" Inicializar Perceptron """
ppn = Perceptron(eta=0.02, n_iter=50)
ppn.fit(X, Y)
#y=ppn.predict(X,1)

""" Grafica de Error por epocas """
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

m = np.random.randint(p)
print('m=',m)
print('Probabilidad: ')
print(ppn.predict_proba(X[m, :]))
print('Predicción: ')
print(ppn.predict(X[m, :],1))
print('Valor Esperado: ')
print(Y[m])

Yest = np.zeros((p,))
for i in range(p):
  Yest[i]=ppn.predict(X[i,:],1)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(Y,Yest))
print('Confusion matrix: \n', confusion_matrix(Y,Yest))


