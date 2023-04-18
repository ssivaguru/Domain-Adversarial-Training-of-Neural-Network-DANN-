import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist.data, mnist.target.astype(int)

# Normalize pixel values to range [0, 1]
X = X / 255.0

# Split data into source (Xs, ys) and target (Xt, yt) datasets
Xs, Xt, ys, yt = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
ys = encoder.fit_transform(ys.reshape(-1, 1))
yt = encoder.fit_transform(yt.reshape(-1, 1))

# Initialize neural network parameters
D = 50
W = np.random.randn(X.shape[1], D)
V = np.random.randn(D, ys.shape[1])
b = np.zeros(D)
c = np.zeros(ys.shape[1])
u = 0
d = 0

# Set hyperparameters
lambda_ = 0.1
mu = 0.1
n_epochs = 5

# Train neural network with domain adaptation
for epoch in range(n_epochs):
    for i in range(Xs.shape[0]):
        # Forward propagation on source domain
        Gf_xi = sigmoid(b + np.dot(Xs[i], W))
        Gy_Gf_xi = softmax(c + np.dot(Gf_xi, V))

        # Backward propagation on source domain
        delta_c = -(ys[i] - Gy_Gf_xi)
        delta_V = np.outer(Gf_xi, delta_c)
        delta_b = np.dot(V, delta_c) * Gf_xi * (1 - Gf_xi)
        delta_W = np.outer(Xs[i], delta_b)

        # Forward propagation on target domain
        j = np.random.randint(Xt.shape[0])
        Gf_xj = sigmoid(b + np.dot(Xt[j], W))
        Gd_Gf_xj = sigmoid(d + np.dot(Gf_xj, u))

        # Backward propagation on target domain
        delta_d = lambda_ * (1 - Gd_Gf_xj)
        delta_u = lambda_ * (1 - Gd_Gf_xj) * Gf_xj
        delta_b += delta_d * Gf_xi * (1 - Gf_xi)
        delta_W += np.outer(Xt[j], delta_d * u * Gf_xj * (1 - Gf_xj))

        # Update neural network parameters
        V -= mu * delta_V
        c -= mu * delta_c
        W -= mu * delta_W
        b -= mu * delta_b

        # Update domain classifier
        u += mu * delta_u
        d += mu * delta_d

    # Compute accuracy on source and target domains
    correct_source = 0
    correct_target = 0
    for i in range(Xs.shape[0]):
        Gf_xi = sigmoid(b + np.dot(Xs[i], W))
        Gy_Gf_xi = softmax(c + np.dot(Gf_xi, V))
