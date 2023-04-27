import numpy as np
from tensorflow import keras
import pandas as pd
import sys
import h5py

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

train_mnist_m = pd.read_csv('train_mnist-m.csv')

X_train = np.repeat(X_train [:, :, :, np.newaxis], 3, axis=3)
X_test = np.repeat(X_test[:, :, :, np.newaxis], 3, axis=3)

height = 28
width = 28
channels = 3
MNIST_M_PATH = './Datasets/MNIST_M/mnistm.h5'

input_size = height * width * channels
# Flatten the input
X_train = X_train.reshape((-1, input_size))
X_test = X_test.reshape((-1, input_size))
#target = train_mnist_m.drop('label', axis=1).to_numpy().reshape(-1, 784)


with h5py.File(MNIST_M_PATH, 'r') as mnist_m:
    mnist_m_train_x, mnist_m_test_x = mnist_m['train']['X'][()], mnist_m['test']['X'][()]


mnist_m_train_x = mnist_m_train_x.reshape((-1, input_size))
mnist_m_test_x = mnist_m_test_x.reshape((-1, input_size))

mnist_m_train_x, mnist_m_test_x = mnist_m_train_x / 255.0, mnist_m_test_x / 255.0
mnist_m_train_x, mnist_m_test_x = mnist_m_train_x.astype('float32'), mnist_m_test_x.astype('float32')

mnist_m_train_y, mnist_m_test_y = y_train, y_test



# Normalize the input
X_train = X_train / 255.0
X_test = X_test / 255.0

#
# Convert the labels to one-hot encoding
#label = train_mnist_m['label']
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[mnist_m_test_y]

# Set the hyperparameters
learning_rate = 0.3
n_epochs = 10
batch_size = 1
n_hidden = 128

# Initialize the weights and biases
W = np.random.randn(input_size, n_hidden) * 0.01
b = np.zeros((1, n_hidden))
V = np.random.randn(n_hidden, 10) * 0.01
c = np.zeros((1, 10))
U = np.random.randn(n_hidden, 10) * 0.01
d = np.zeros((1, 10))


adaption_rate = 0.1


def one_hot_encode(labels):
    num_labels = len(labels)
    num_classes = 10
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels] = 1
    return one_hot

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the softmax activation function
def softmax(x):
    # Subtract the maximum value for numerical stability
    x -= np.max(x, axis=1, keepdims=True)
    
    # Compute the exponentials
    exp_x = np.exp(x)
    
    # Normalize by the sum
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Define the cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Iterate over the training set
for epoch in range(n_epochs):
    # Shuffle the training set
    perm = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[perm]
    X_target_shuffled = mnist_m_train_x[perm]
    y_train_shuffled = y_train_onehot[perm]

    # Iterate over the mini-batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        X_target = X_target_shuffled[i:i+batch_size]
        
        # Compute the forward pass
        h = sigmoid(np.dot(X_batch, W) + b)
        y_pred = softmax(np.dot(h, V) + c)
        
        # Compute the gradients
        dV = np.dot(h.T, y_pred - y_batch) / batch_size
        dc = np.mean(y_pred - y_batch, axis=0)
        dh = np.dot(y_pred - y_batch, V.T) * h * (1 - h)
        db = np.mean(dh, axis=0)
        dW = np.dot(X_batch.T, dh) / batch_size
        
        #domain correction
        gd = sigmoid(np.dot(h, U) + d)        
        dd = adaption_rate * (1 - gd)
        du = adaption_rate * np.dot(h.T, (1 - gd))        
        tmp = (np.dot(U , (adaption_rate * (1 - gd)).T)  * np.dot(h , (1 - h).T)).reshape(128, 1)
        db = db + tmp.flatten()
        dW = dW + (tmp * X_batch).T
        
        #other domain
        gf_target = sigmoid(np.dot(X_target, W) + b)
        gd_target = softmax(np.dot(gf_target, U) + d)        
        dd = dd - adaption_rate * (gd_target)
        du = du - adaption_rate * np.dot(gf_target.T, gd_target)  
        tmp = -1 * adaption_rate * (np.dot(U , (gd_target).T)  * np.dot(gf_target , (1 - gf_target).T)).reshape(128, 1)
        db = db + tmp.flatten()
        dW = dW + (tmp * X_target).T
        
        
        # Update the weights and biases
        V -= learning_rate * dV
        c -= learning_rate * dc
        W -= learning_rate * dW
        b -= learning_rate * db
        
        U = U + learning_rate*du
        d = d + learning_rate*dd
    
    # Compute the training loss and accuracy
    h = sigmoid(np.dot(X_train, W) + b)
    y_train_pred = softmax(np.dot(h, V) + c)
    train_loss = cross_entropy_loss(y_train_onehot, y_train_pred)
    train_acc = np.mean(np.argmax(y_train_pred, axis=1) == y_train)

    # Compute the test loss and accuracy
    h = sigmoid(np.dot(mnist_m_test_x, W) + b)
    y_test_pred = softmax(np.dot(h, V) + c)
    test_loss = cross_entropy_loss(y_test_onehot, y_test_pred)
    test_acc = np.mean(np.argmax(y_test_pred, axis=1) == mnist_m_test_y)

    # Print the results
    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - target_loss: {test_loss:.4f} - target_acc: {test_acc:.4f}")