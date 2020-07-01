import numpy as np
import pandas as pd

class Activation:
  def step(self, x):
    if x >= 0:
      return 1
    return 0

  def softmax(self, x):
    return np.divide (1, 1 + np.exp(x))

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_prime(self, x):
    return self.sigmoid(x) * (1 - self.sigmoid(x))

  def tanh(self, x):
    return np.divide( (np.exp(x) - np.exp(-x)), (np.exp(x) + np.exp(-x)) )

  def tanh_prime(self, x):
    return 1 - np.divide( pow(np.exp(x)-np.exp(-x), 2), pow(np.exp(x)+np.exp(-x), 2) )

  def relu(self, x):
    max_val = np.amax(x)
    if max_val >= 0:
      return max_val
    return 0

  def relu_prime(self, x):
    if np.amax(x) >= 0:
      return 1
    return 0