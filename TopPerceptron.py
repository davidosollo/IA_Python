#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:27:28 2020
Perceptron Compuertas Logicas
@author: davidosollo
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

  def __init__(self, n_inputs, learning_rate, epochs):
    self.w = - 1 + 2 * np.random.rand(n_inputs)
    self.b = - 1 + 2 * np.random.rand()
    self.eta = learning_rate
    self.epochs = epochs

  def predict(self, X):
    _, p = X.shape
    y_est = np.zeros(p)
    for i in range(p):
      y_est[i] = np.dot(self.w, X[:,i])+self.b
      if y_est[i] >= 0:
        y_est[i]=1
      else:
        y_est[i]=0
    return y_est

  def fit(self, X, Y):
    _, p = X.shape
    for _ in range(self.epochs):
      for i in range(p):
        y_est = self.predict(X[:,i].reshape(-1,1))
        self.w += self.eta * (Y[i]-y_est) * X[:,i]
        self.b += self.eta * (Y[i]-y_est)

def draw_2d_percep(model):
  w1, w2, b = model.w[0], model.w[1], model.b 
  plt.plot([-2, 2],[(1/w2)*(-w1*(-2)-b),(1/w2)*(-w1*2-b)],'--k')
  
  # Instanciar el modelo
model = Perceptron(2, 0.1,50)

# Datos Input
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

# And Gate
#Y = np.array([0, 0, 0, 1])
# OR Gate
Y = np.array([0, 1, 1, 1])
# XOR Gate
#Y = np.array([0, 1, 1, 0])
    
# XOR Gate


# Entrenar
model.fit(X,Y)

# Predicción
#model.predict(X)



# Primero dibujemos los puntos
_, p = X.shape
for i in range(p):
  if Y[i] == 0:
    plt.plot(X[0,i],X[1,i], 'or')
  else:
    plt.plot(X[0,i],X[1,i], 'ob')

plt.title('Perceptrón')
plt.grid('on')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

draw_2d_percep(model)