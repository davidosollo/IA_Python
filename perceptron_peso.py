#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:04:50 2020

"Clasifica Personas con sobre peso

@author: davidosollo
"""
#Init Libs
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

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
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

#Dibujar Linea resultado con los resultados de los pesos    
def draw_2d_percep(model):
  w1, w2 = model.w_[1], model.w_[2]
  b =  model.w_[0]
  plt.plot([-120, 120],[(1/w2)*(-w1*(-120)-b),(1/w2)*(-w1*120-b)],'--k')
  
# Generar Personas con peso y ALtura aleatoria siguiendo una distribucion Normal        
NumPers = 40 #Numero de Personas a Generar

X = np.zeros((2, NumPers))
#np.random.seed(30)
X[0:NumPers] = np.random.normal(loc=1.65, scale=.18, size=(1,NumPers))  #Peso
X[1:NumPers] = np.random.normal(loc=67.9, scale= 15, size=(1,NumPers))  #Altura

Y, Z =  np.where( (X[1] / X[0]**2) < 25 ,-1, 1) , X[1] / X[0]**2        #Formula para determinar sobre peso

#Normalizar los datos
normalized_X = preprocessing.normalize(X, axis=1, norm='l2')
X=np.transpose(normalized_X)

#Inicializar Perceptron
ppn = Perceptron(eta=0.05, n_iter=50)
ppn.fit(X, Y)
y=ppn.predict(X)

#Grafica de Error por epocas
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()


#Graficar los datos y la recta resultante de los pesos
X=np.transpose(X)

_, p = X.shape

Obeso = 0;
Flaco = 0;

for i in range(p):
  if Y[i] == 1:
    plt.plot(X[0,i],X[1,i], 'og',label='Obeso' if Obeso == 0 else "" )   #Obeso Verde
    Obeso = 1
  else:
    plt.plot(X[0,i],X[1,i], 'ob', label='No Obeso' if Flaco == 0 else "")   #No Obeso Azul
    Flaco = 1
     
plt.title('Perceptrón')
plt.grid('on')
plt.ylim([0,0.5])
plt.xlim([0,0.5])
plt.xlabel('Peso')
plt.ylabel('Altura')
plt.legend(loc='upper left')

X=np.transpose(X)
draw_2d_percep(ppn)

plt.show()

