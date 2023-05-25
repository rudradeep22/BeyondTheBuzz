import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score 
from model import ANN

model = ANN(784, 10, 0.1, 2, [32, 32])
data_train = pd.read_csv('C:/Users/rudra/OneDrive/Desktop/Python_Work/BYB A1/mnist_train.csv')
data_train = np.array(data_train)
X_train = data_train[:, 1:]
X_train = X_train/255
Y_train = data_train[:, 0]

data_test = pd.read_csv('C:/Users/rudra/OneDrive/Desktop/Python_Work/BYB A1/mnist_test.csv')
data_test = np.array(data_test)
X_test = data_test[:, 1:]
X_test = X_test/255
Y_test = data_test[:, 0]

for i in range(3):
    model.train_all(X_train, Y_train)

y_pred = model.predict(X_test, Y_test)
y_hat = []
for i in y_pred:
    prediction = np.argmax(i)
    y_hat.append(prediction)
acc = accuracy_score(Y_test, y_hat)
print(f'Accuracy is: {acc*100}%')
