import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report









import numpy as np

def street_view_house_numbers(samples, hidden_layer_size, adaptation_parameter, learning_rate):
    # Unpacking input samples
    X, y = samples
    n = X.shape[0] # Number of samples
    print(n)
    print(X[0].shape)
    print(X.shape)
    print(X[0].shape)
    print(X.shape[1])
    
    
    # Random initialization of neural network parameters
    print ("Assignment")
   # W = np.random.randn(hidden_layer_size, X.shape[1])
    W = np.random.randn(X.shape[1], hidden_layer_size)
    #print (W.shape)
    V = np.random.randn(y.shape[0], hidden_layer_size)
    b = np.zeros((hidden_layer_size,))
    c = np.zeros((y.shape[0],))
    u = 0
    d = 0
    #print ("Assignment1")
    # Stopping criterion
    stopping_criterion = False
        
    prev_loss = float('inf')
    tol = 1e-4
    epochs = 0
    #print ("Start While")
    while not stopping_criterion:
       # print ("inside While")
        epochs += 1
        for i in range(n):

            # Forward propagation
            
            Gf = sigmoid(np.dot( X[i],W) + b)
            #Gf = sigmoid(np.dot(W, X[i].reshape(1, -1)) + b)
            
            #Gf = sigmoid(np.dot(W, X[i].values.reshape(1, -1)) + b)
            #Gf = sigmoid(np.dot(X[i].values.reshape(1, -1), W.T) + b)
            
            Gy = softmax(np.dot(V, Gf) + c)
            
            # Backpropagation
            delta_c = -(y[i] - Gy)
            delta_V = np.outer(delta_c, Gf)
            delta_b = np.dot(V.T, delta_c) * (Gf * (1 - Gf))
            delta_W = np.outer(delta_b, X[i])
            
            # Domain adaptation regularizer from current domain
            Gd_Gf = sigmoid(np.dot(d, Gf) + u)
            delta_d = adaptation_parameter * (1 - Gd_Gf)
            delta_u = adaptation_parameter * (1 - Gd_Gf) * Gf
            
            tmp = adaptation_parameter * (1 - Gd_Gf) * u * Gf * (1 - Gf)
            delta_b += tmp
            delta_W += np.outer(tmp, X[i])
            
            # Domain adaptation regularizer from other domain
            j = np.random.randint(n)
            Gf_xj = sigmoid(np.dot(W.T, X[j]) + b)
            Gd_Gf_xj = sigmoid(np.dot(d, Gf_xj) + u)
            delta_d -= adaptation_parameter * Gd_Gf_xj
            delta_u -= adaptation_parameter * Gd_Gf_xj * Gf_xj
            
            tmp = -adaptation_parameter * Gd_Gf_xj * u * Gf_xj * (1 - Gf_xj)
            delta_b += tmp
            delta_W += np.outer(tmp, X[i].T)
            
            #print (learning_rate)
            #print (W.shape)
            #print("Update neural network parameters")
            #print (delta_W.shape)
            # Update neural network parameters
            W -= learning_rate * delta_W.T
            V -= learning_rate * delta_V
            b -= learning_rate * delta_b
            c -= learning_rate * delta_c
            
            # Update domain classifier
            u += learning_rate * delta_u
            d += learning_rate * delta_d
        
        # Update stopping criterion
        #stopping_criterion = True # Add stopping criterion here
        print("term1", np.sum(np.square(y - Gy)))
        print("term1", Gy)
        print("term1", y )
        print("term2", np.sum(np.square(Gd_Gf - 1)))
        print("term3", np.sum(np.square(Gd_Gf_xj)))
        
        loss = np.sum(np.square(y - Gy)) + adaptation_parameter * np.sum(np.square(Gd_Gf - 1)) + adaptation_parameter * np.sum(np.square(Gd_Gf_xj))
        print ("Loss :::::----->" ,(prev_loss - loss) ,"Previous Loss", prev_loss)
        #if prev_loss - loss < tol:
         #   stopping_criterion = True
        #prev_loss = loss
        
    if (epochs>5 ):
        stopping_criterion=True
        
    return W, V, b, c, u, d

def sigmoid(x):
    #print ("Before Sigmoid ::", x)
    sigmoid_scr= 1 / (1 + np.exp(-x))
    if sigmoid_scr[0] =='nan' :
        exit
    return sigmoid_scr

def softmax(x):
    #print ("Before softmax ::", x)
    exp_scores = np.exp(x)
    if exp_scores[0] =='nan' :
        exit
    return exp_scores / np.sum(exp_scores)

# Load the MNIST dataset . Test Data set 
mnist = datasets.load_digits()


# Split the dataset into training and testing sets. 80% to 20%
#X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=40)

##X = mnist.drop('label', axis=1).values
#y = mnist['label'].values
#print (y_train)

# Call the function with the required inputs
#samples = (X, y) # Replace X and y with your actual input data
#samples = (X_train,X_test)
#hidden_layer_size = 2 # Replace D with the desired hidden layer size
#adaptation_parameter = 0.1 # Replace lambda_val with the desired adaptation parameter
#learning_rate = 0.001 # Replace mu with the desired learning rate

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
#mnist = fetch_openml('mnist_784')
mnist = datasets.load_digits()


# Split the dataset into training and testing sets. 80% to 20%
#X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=40)
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=40)

# Convert labels to integers
#y_train = y_train.astype(int)
#y_test = y_test.astype(int)

# Normalize pixel values to the range [0, 1]
#X_train = X_train / 255.0
#X_test = X_test / 255.0

# Reshape the input data to match the expected input shape of the model
#X_train = X_train.T
#X_test = X_test.T

# Set the adaptation parameter, learning rate, and hidden layer size
adaptation_parameter = 0.05
learning_rate = 0.001
hidden_layer_size = 1

# Call the function with the required inputs
samples = (X_train, y_train)
W, V, b, c, u, d = street_view_house_numbers(samples, hidden_layer_size, adaptation_parameter, learning_rate)

#W, V, b, c, u, d = street_view_house_numbers(samples, hidden_layer_size, adaptation_parameter, learning_rate)
