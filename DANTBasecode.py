

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import math

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Limit the range of x to avoid overflow
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)



def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encoded = np.zeros((n_labels, n_unique_labels))
    one_hot_encoded[np.arange(n_labels), labels] = 1
    return one_hot_encoded

def forward_propagation(X, parameters):
    W, V, b, c = parameters

    # Hidden layer activation
    Z = np.dot(W, X) + b
    H = sigmoid(Z)

    # Output layer activation
    U = np.dot(V, H) + c
    Y_hat = softmax(U)

    #return Y_hat, [Z, H, U]
    return Y_hat

def predict(parameters, X):
    # Obtain output of model_fn given the test data 
    activated_output = forward_propagation(X, parameters)
    return np.argmax(activated_output, axis=0)

def prediction(parameters, X):
    # Obtain output of model_fn given the test data 
    activated_output, outputs = forward_propagation(X, parameters)
    return np.argmax(activated_output, 0)

def train_network(samples, labels, hidden_layer_size, adaptation_parameter, learning_rate):
    n_samples, n_features = samples.shape
    n_labels = np.unique(labels).size
    n_hidden = hidden_layer_size

    # Initialize weights and biases
    np.random.seed(0)
    W = np.random.randn(n_hidden, n_features)
    V = np.random.randn(n_labels, n_hidden)
    b = np.zeros((n_hidden, 1))
    c = np.zeros((n_labels, 1))
    u = np.zeros((n_hidden, 1))
    d = 0

    # Convert labels to one-hot encoding
    Y = one_hot_encode(labels)

    while True:
        for i in range(n_samples):
            # Forward propagation
            xi = samples[i].reshape(n_features, 1)
            gf = sigmoid(b + np.dot(W, xi))
            gy = softmax(c + np.dot(V, gf))

            # Backpropagation
            delta_c = -(Y[i].reshape(n_labels, 1) - gy)
            delta_v = np.dot(delta_c, gf.T)
            delta_b = np.dot(V.T, delta_c) * gf * (1 - gf)
            delta_w = np.dot(delta_b, xi.T)

            # Domain adaptation regularizer
            gd = sigmoid(d + np.dot(u.T, gf))
            delta_d = adaptation_parameter * (1 - gd)
            delta_u = adaptation_parameter * (1 - gd) * gf
            tmp = adaptation_parameter * (1 - gd) * u * gf * (1 - gf)
            delta_b += tmp
            delta_w += np.dot(tmp, xi.T)

            # Regularizer for other domain
            #j = np.random.randint(n_samples, n_samples*2)
            #xj = samples[j].reshape(n_features, 1)
            j = np.random.randint(n_samples)
            xj = samples[j].reshape(n_features, 1)
            #j = np.random.randint(n_prime)
            #xj = samples[j].reshape(1, -1)
            gf_j = sigmoid(b + np.dot(W, xj))
            gd_j = sigmoid(d + np.dot(u.T, gf_j))
            delta_d -= adaptation_parameter * gd_j
            delta_u -= adaptation_parameter * gd_j * gf_j
            tmp = -adaptation_parameter * gd_j * u * gf_j * (1 - gf_j)
            delta_b += tmp
            delta_w += np.dot(tmp, xj.T)

            # Update parameters
            W -= learning_rate * delta_w
            V -= learning_rate * delta_v
            b -= learning_rate * delta_b
            c -= learning_rate * delta_c
            u += learning_rate * delta_u
            d += learning_rate * delta_d
            print (np.max(np.abs(delta_w)) )
        # Stopping criterion
        if np.max(np.abs(delta_w)) < 1e-6:
            
            break

    return W, V, b, c


# Load the CSV file into a pandas DataFrame
data = pd.read_csv('/Users/akk0018/Documents/ML/MLProject/mnist_train.csv')

# Get the first 100 records of the DataFrame
first_100 = data.head(500000)

X_train, X_test, y_train, y_test = train_test_split(first_100.drop('label', axis=1).values, first_100['label'].values, test_size=0.2, random_state=40)

# Call the function with the required inputs
samples = (X_train, y_train)
adaptation_parameter = 0.01
learning_rate = 0.001
hidden_layer_size = 200
samples = (X_train, y_train)
W, V, b, c = train_network(X_train, y_train, hidden_layer_size, adaptation_parameter, learning_rate)
#print ( W,V,b,c)




# Get predictions for test data
y_pred = predict((W, V, b, c), X_test.T)

# Print classification report
print(classification_report(y_test, y_pred))