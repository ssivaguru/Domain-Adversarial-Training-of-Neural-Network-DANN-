

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Limit the range of x to avoid overflow
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)

def get_data() :
    # load the data
    train_df = pd.read_csv('/Users/akk0018/Documents/ML/SVM_Assignment2/data/mnist_train.csv')
    test_df=pd.read_csv('/Users/akk0018/Documents/ML/SVM_Assignment2/data/mnist_test.csv')
    crossdomain_test=pd.read_csv('/Users/akk0018/Documents/ML/SVM_Assignment2/data/train_mnist-m.csv')
    #first_100 = train_df.head(500000)
    #X_train, X_test, y_train, y_test = train_test_split(first_100.drop('label', axis=1).values, first_100['label'].values, test_size=0.5, random_state=40)

    #X_train = train_df.drop('label', axis=1).values
    #y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    
    X_test = crossdomain_test.drop('label', axis=1).values
    y_test = crossdomain_test['label'].values
    
    W_test = test_df.drop('label', axis=1).values
    w_test = test_df['label'].values
    

    return X_train, X_test, y_train, y_test ,W_test,w_test


# Define the cross-entropy loss function
'''def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))'''
def cross_entropy_loss(y_true, y_pred):
    y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    y_pred = y_pred.reshape(-1, y_true.shape[1]) if y_pred.ndim == 1 else y_pred
    epsilon = 1e-9 # small constant to avoid division by zero
    activated_output = softmax(y_pred)
    loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
    return loss, accuracy


def prediction(parameters, X):
    # Obtain output of model_fn given the test data 
    activated_output, outputs = forward_propagation(X, parameters)
    return np.argmax(activated_output, 0)

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

def train_network(samples, labels, jsjtest, jlebel, hidden_layer_size, adaptation_parameter, learning_rate):
    n_samples, n_features = samples.shape
    n_jsamples, n_jfeatures = jsjtest.shape
    n_labels = np.unique(labels).size
    n_jlabels = np.unique(jlebel).size
    n_hidden = hidden_layer_size

    # Initialize weights and biases
    np.random.seed(0)
    W = np.random.randn(n_hidden, n_features) 
    V = np.random.randn(n_labels, n_hidden)
    b = np.zeros((n_hidden, 1))
    c = np.zeros((n_labels, 1))
    u = np.zeros((n_hidden, 1))
    #d = np.zeros((n_labels, 1))
    d=0
    ''' # Initialize the weights and biases
    W = np.random.randn(784, n_hidden) * 0.01
    b = np.zeros((1, n_hidden))
    V = np.random.randn(n_hidden, 10) * 0.01
    c = np.zeros((1, 10))
    U = np.random.randn(n_hidden, 10) * 0.01
    d = np.zeros((1, 10))
    # Convert labels to one-hot encoding'''
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
            j = np.random.randint(n_jsamples)
            xj = jsjtest[j].reshape(n_jfeatures, 1)
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
        if np.max(np.abs(delta_w)) < 6e-6:
            
            break

    return W, V, b, c

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Load the CSV file into a pandas DataFrame
X_train, J_train, y_train, k_train ,W_test,w_test=get_data()
#data = pd.read_csv('/Users/akk0018/Documents/ML/MLProject/mnist_train.csv')
#data2=pd.read_csv('/Users/akk0018/Documents/ML/MLProject/mnist_test.csv')

# Get the first 100 records of the DataFrame
##first_100 = data.head(500000)

#X_train, X_test, y_train, y_test = train_test_split(first_100.drop('label', axis=1).values, first_100['label'].values, test_size=0.5, random_state=40)
#Z_train, Z_test, p_train, p_test = train_test_split(first_100.drop('label', axis=1).values, first_100['label'].values, test_size=0.2, random_state=40)

# Call the function with the required inputs
#Xsamples = (X_train, y_train)
adaptation_parameter = 0.01
learning_rate = 0.001
hidden_layer_size = 250
Xsamples = (X_train, y_train)
Jsamples = (X_train, y_train)
W, V, b, c = train_network(X_train, y_train, J_train, k_train, hidden_layer_size, adaptation_parameter, learning_rate)
#print ( W,V,b,c)






# Get predictions for test data
y_pred = predict((W, V, b, c), W_test.T)
y_pred_train = predict((W, V, b, c), X_train.T)
print (y_pred)

print (w_test)


#test_loss,test_acc = cross_entropy_loss(w_test,y_pred)

# Make predictions on the training and test data
#y_train_pred = predict((W, V, b, c), X_train)
#y_test_pred = predict((W, V, b, c), W_test)

# Calculate the training and test accuracy
train_acc = calculate_accuracy(y_train, y_pred_train)
test_acc = calculate_accuracy(w_test, y_pred)

print("Training accuracy:", train_acc)
print("Test accuracy:", test_acc)

# Print classification report
#print(classification_report(w_test, y_pred))

#print(test_loss,test_acc)y
accuracy = accuracy_score(w_test, y_pred)
precision = precision_score(w_test, y_pred, average='weighted')
recall = recall_score(w_test, y_pred, average='weighted')
f1 = f1_score(w_test, y_pred, average='weighted')

print(f' Test accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1}')

accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_train, y_pred_train, average='weighted')
recall = recall_score(y_train, y_pred_train, average='weighted')
f1 = f1_score(y_train, y_pred_train, average='weighted')
print(f' Training accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1}')
