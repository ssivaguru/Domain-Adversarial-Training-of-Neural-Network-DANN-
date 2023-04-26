import numpy as np
from tensorflow import keras
import os
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np


def img_transform_numpy(image, image_size):
    # Convert image to NumPy array
    image = np.array(image)

    # Convert image to grayscale
    image = Image.fromarray(image).convert('L')

    # Resize image
    image = image.resize((image_size, image_size))

    # Convert image to float and scale to [0, 1]
    image = np.array(image).astype(np.float32)
    image /= 255.0

    return image

def createMNSTM():
    train_path = '/home/siva/Project/jupyter/Domain-Adversarial-Training-of-Neural-Network-DANN-/MNIST-M/training'
    test_path = '/home/siva/Project/jupyter/Domain-Adversarial-Training-of-Neural-Network-DANN-/MNIST-M/testing'
    # Load the training data
    train_data = []
    train_labels = []
    image_size=28
    for i in range(10):
        class_path = os.path.join(train_path, str(i))
        for file_name in os.listdir(class_path):
            image_path = os.path.join(class_path, file_name)
            image = Image.open(image_path).convert('RGB')
            resized_image = img_transform_numpy(image, image_size)
            train_data.append(np.array(resized_image).flatten())
            train_labels.append(i)

    # Load the test data
    test_data = []
    test_labels = []
    for i in range(10):
        class_path = os.path.join(test_path, str(i))
        for file_name in os.listdir(class_path):
            image_path = os.path.join(class_path, file_name)
            image = Image.open(image_path).convert('RGB')
            resized_image = img_transform_numpy(image, image_size)
            test_data.append(np.array(resized_image).flatten())
            print(np.array(resized_image).shape)
            test_labels.append(i)

    # Convert to DataFrame
    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels)
    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)
    train_data = pd.DataFrame(data=train_data, columns=[f"pixel{i}" for i in range(train_data.shape[1])])
    train_data['label'] = train_labels
    test_data = pd.DataFrame(data=test_data, columns=[f"pixel{i}" for i in range(test_data.shape[1])])
    test_data['label'] = test_labels


    train_data.to_csv('train_mnist-m.csv', index=False)
    test_data.to_csv('test_mnist-m.csv', index=False)
    
# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Flatten the input
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
#target = train_mnist_m.drop('label', axis=1).to_numpy().reshape(-1, 784)

# Normalize the input
X_train = X_train / 255.0
X_test = X_test / 255.0
#target = target / 255.0

#label = train_mnist_m['label']
# Convert the labels to one-hot encoding
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

# Set the hyperparameters
learning_rate = 0.1
n_epochs = 100
batch_size = 1
n_hidden = 128

# Initialize the weights and biases
W = np.random.randn(784, n_hidden) * 0.01
b = np.zeros((1, n_hidden))
V = np.random.randn(n_hidden, 10) * 0.01
c = np.zeros((1, 10))
U = np.random.randn(n_hidden, 10) * 0.01
d = np.zeros((1, 10))


adaption_rate = 0.01

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
    y_train_shuffled = y_train_onehot[perm]

    # Iterate over the mini-batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        #random_index = np.random.randint(0, X_train_shuffled.shape[0])
        # Select the corresponding dataset
        #X_target = X_train_shuffled[random_index]
        
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
        #gd = sigmoid(np.dot(h, U) + d)        
        #dd = dd + adaption_rate * (1 - gd)
        #du = du + adaption_rate * np.dot(h.T, (1 - gd))        
        #tmp = (np.dot(U , (adaption_rate * (1 - gd)).T)  * np.dot(h , (1 - h).T)).reshape(128, 1)
        #db = db + tmp.flatten()
        #dW = dW + (tmp * X_batch).T
        
        #other domain
        #gf_target = sigmoid(np.dot(X_target, W) + b)
        #gd_target = softmax(np.dot(gf_target, U) + d)        
        #dd = dd - adaption_rate * (gd_target)
        #du = du - adaption_rate * np.dot(gf_target.T, gd_target)  
        #tmp = -1 * adaption_rate * (np.dot(U , (gd_target).T)  * np.dot(gf_target , (1 - gf_target).T)).reshape(128, 1)
        #db = db + tmp.flatten()
        #dW = dW + (tmp * X_target).T
        
        
        # Update the weights and biases
        V -= learning_rate * dV
        c -= learning_rate * dc
        W -= learning_rate * dW
        b -= learning_rate * db
        
        #U = U + learning_rate*du
        #d = d + learning_rate*dd
    
    # Compute the training loss and accuracy
    h = sigmoid(np.dot(X_train, W) + b)
    y_train_pred = softmax(np.dot(h, V) + c)
    train_loss = cross_entropy_loss(y_train_onehot, y_train_pred)
    train_acc = np.mean(np.argmax(y_train_pred, axis=1) == y_train)

    # Compute the test loss and accuracy
    #h = sigmoid(np.dot(target, W) + b)
    #y_test_pred = softmax(np.dot(h, U) + d)
    #test_loss = cross_entropy_loss(y_test_onehot, y_test_pred)
    #test_acc = np.mean(np.argmax(y_test_pred, axis=1) == label)

    # Print the results
    print(f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}")

