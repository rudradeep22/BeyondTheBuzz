import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score 
from model3 import ANN

data = pd.read_csv('mnist_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) 

data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

data2 = pd.read_csv('mnist_test.csv')
data2 = np.array(data2)
np.random.shuffle(data2)
data2 = data2.T
y_test = data2[0]
X_test = data2[1:n]
X_test = X_test / 255.

model = ANN(784, 10, 0.1, 1, [10])
model.train(X_train, Y_train, 10)
y_pred = model.predict(X_test[0])
acc = accuracy_score(y_test[0], y_pred)
print(f'Accuracy is: {acc}')
