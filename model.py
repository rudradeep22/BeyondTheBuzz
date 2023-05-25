import numpy as np
from numpy.random import default_rng
import math

def ReLU(x):
    return np.maximum(x,0)

def ReLU_derivative(x):
    der = [ 1 if value>0 else 0 for value in x]
    return np.array(der, dtype=float)

def softmax(x):
    b = x.max()
    y = np.exp(x-b)
    return y/y.sum()

def cross_entropy_loss(y, y_predict):
  return -np.sum(y * np.log(y_predict))

def integer_to_one_hot(x):
  result = np.zeros(10)
  result[x] = 1
  return result

class ANN():
    def __init__(self, input_size, output_size, learning_rate, num_hidden_layers, num_hidden_nodes):
      self.input_size = input_size
      self.output_size = output_size
      self.lr = learning_rate
      self.hidden_layers = num_hidden_layers
      self.hidden_nodes = num_hidden_nodes

    def __params(self):
        rng = default_rng(12345)
        self.weights = []
        self.biases = []
        self.activations = []
        self.nodes = [self.input_size]+ self.hidden_nodes + [self.output_size] 
        for i in range(self.hidden_layers+1):
            weight = rng.normal(0, 1/math.sqrt(self.nodes[i]), (self.nodes[i+1], self.nodes[i]))
            bias = np.ones(self.nodes[i+1])
            self.weights.append(weight)
            self.biases.append(bias)
    
    def __forward(self, sample, y):
       a = sample.flatten()
       for i, weight in enumerate(self.weights):
          z = np.dot(weight, a) + self.biases[i]
          if i < len(self.weights) - 1:
             a = ReLU(z)
          else:
             a = softmax(z)
          self.activations.append(a)
        
       one_hot_y = integer_to_one_hot(y)
       loss = cross_entropy_loss(one_hot_y, a)
       one_hot_guess = np.zeros(10)
       one_hot_guess[np.argmax(a)] = 1

       return loss, one_hot_guess

    def __forward_dataset(self, X, y):
        losses = np.empty(X.shape[0])
        one_hot_guesses = np.empty((X.shape[0], 10))

        for i in range(X.shape[0]):
          losses[i], one_hot_guesses[i] = self.__forward(X[i], y[i])
        
        y_one_hot = np.zeros((y.size, 10))
        y_one_hot[np.arange(y.size), y] = 1

        # Expected correct guesses 6 000/60 000, assuming perfect randomness
        correct_guesses = np.sum(y_one_hot * one_hot_guesses)
        correct_guess_percent = format((correct_guesses / y.shape[0]) * 100, ".2f")
        print(f'No. of correct guesses : ({correct_guess_percent}%)')
        return one_hot_guesses

    def __backprop(self, sample, y):
       grad_weights = [None] * len(self.weights)
       grad_bias = [None] * len(self.biases)
       grad_activation = [None] * (len(self.weights) -1)
       one_hot_y = integer_to_one_hot(y)

       for i in range(len(self.weights) - 1, -1, -1):
            if i == len(self.weights) - 1: # last layer
                y =  one_hot_y[:, np.newaxis]
                a = self.activations[i][:, np.newaxis]
                a_prev = self.activations[i-1][:, np.newaxis]

                grad_weights[i] = np.dot((a-y), a_prev.T)
                grad_bias[i] = a - y
            else:
                w_next = self.weights[i+1]
                a_next = self.activations[i+1][:, np.newaxis]
                y = one_hot_y[:, np.newaxis]
                a = self.activations[i][:, np.newaxis]
                if i > 0:
                    a_prev = self.activations[i-1][:, np.newaxis]
                else:
                    a_prev = sample.flatten()[:, np.newaxis]
                    
                    # Activation gradient
                if i == len(self.weights) - 2: # second_last layer
                    dCda = np.dot(w_next.T, (a_next - y))
                    grad_activation[i] = dCda
                else:
                    dCda_next = grad_activation[i+1]
                    dCda = np.dot(w_next.T, (ReLU_derivative(a_next) * dCda_next))
                    grad_activation[i] = dCda
        
                x = ReLU_derivative(a) * dCda
                grad_weights[i] = np.dot(x, a)
                grad_bias[i] = x
            self.weights[i] -= grad_weights[i] * self.lr
            self.biases[i] -= grad_bias[i][1] * self.lr
    
    def train(self, sample, y):
       loss, one_hot_guess = self.__forward(sample=sample, y=y)
       self.__backprop(sample=sample, y=y)
    
    def train_epoch(self, X, y):
       print('------One epoch------')
       for i in range(X.shape[0]):
          self.train(X[i], y[i])
       print("Finished Training this epoch")
    
    def train_all(self, X, y):
       self.__params()
       self.train_epoch(X, y)
       self.__forward_dataset(X, y)

    def predict(self, X, y):
       one_hot_guess = self.__forward_dataset(X, y)
       return one_hot_guess


