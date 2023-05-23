import numpy as np
import pandas as pd

data = pd.read_csv('mnist_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.


def ReLU(X):
    return np.maximum(X,0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z))
    return A

def one_hot(y):
    one_hot_enc = np.zeros((10,1))
    one_hot_enc[y] = 1
    return one_hot_enc

def ReLU_deriv(Z):
    return Z > 0

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

class ANN():
    def __init__(self, input_size, output_size, learning_rate, num_hidden_layers, num_hidden_nodes):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
    
    def __params(self):
        num_params = self.num_hidden_layers + 1
        self.weights = []
        self.biases = []
        for i in range(num_params):
            if i == 0:
                weight = np.random.rand(self.num_hidden_nodes[0], self.input_size)
                bias = np.random.rand(self.num_hidden_nodes[0], 1)
            elif i == num_params-1:
                weight = np.random.rand(self.output_size, self.num_hidden_nodes[-1])
                bias = np.random.rand(self.output_size, 1)
            else:
                weight = np.random.rand(self.num_hidden_nodes[i], self.num_hidden_nodes[i-1])
                bias = np.random.rand(self.num_hidden_nodes[i], 1)
            self.weights.append(weight)
            self.biases.append(bias)
    
    def __forward(self, X):
        self.Z_list = []
        self.A_list = []
        inp = X
        self.A_list.append(X)
        print('weight is : ',self.weights[0])
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i],inp) + self.biases[i]
            if i == len(self.weights)-1 :
                pass#a = softmax(z)
            else:
                a = ReLU(z)
            self.Z_list.append(z)
            self.A_list.append(a)
            inp = a

    def __loss(self, y_predict, y): 
        y_true = one_hot(y)       
        return -np.sum(y_true * np.log(y_predict))
    
    def __backward(self, X, y):
        m = X.shape[0]
        y_true = one_hot(y)
        grad_z = []
        grad_W = []
        grad_b = []
        dZ2 = self.A_list[-1] - y_true
        dW2 = 1 / m * dZ2.dot(self.A_list[-2].T)
        print('DW is: ',dW2)
        db2 = 1 / m * dZ2
        grad_z.append(dZ2)
        grad_W.append(dW2)
        grad_b.append(db2)
        for i in range(len(self.weights)-2, -1, -1):
            dZ = np.dot(self.weights[i+1].T, grad_z[-1]) * ReLU_deriv(self.Z_list[i])
            dW = 1 / m * np.dot(dZ, self.A_list[i].T)
            db = 1 / m * dZ
            grad_z.append(dZ)
            grad_W.append(dW)
            grad_b.append(db)
        return grad_W, grad_b
    
    def train(self, X, y, num_epoch):
        self.__params()
        for epoch in range(num_epoch):
            self.__forward(X)
            grad_w, grad_b = self.__backward(X, y)
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - self.lr * grad_w[-(1+i)]
                self.biases[i] -= self.lr * grad_b[-(1+i)]
            if epoch%10 == 0:
                print("Iteration: ", epoch)
                predictions = get_predictions(self.A_list[-1])
                print(get_accuracy(predictions, y))

model = ANN(784, 10, 0.1, 1, [10])
model.train(X_train, Y_train, 10)



