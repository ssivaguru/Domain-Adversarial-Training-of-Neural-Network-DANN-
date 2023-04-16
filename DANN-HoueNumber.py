
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
data = pd.read_csv('/Users/akk0018/Documents/ML/MLProject/mnist_train.csv')

# Split the dataset into input features (X) and target variable (y)
X = data.iloc[:, 1:].values.astype('float32')
y = data.iloc[:, 0].values.astype('int32')

# Normalize the input features
X /= 255

# Convert the target variable to one-hot encoded format
y = to_categorical(y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
inputs = Input(shape=(784,))
x = Dense(512, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))

# Define the adversarial domain adaptation model
inputs = Input(shape=(784,))
x = Dense(512, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
features = x

adversary = Dense(64, activation='relu')(features)
adversary = Dropout(0.2)(adversary)
adversary = Dense(1, activation='sigmoid')(adversary)

outputs = Dense(10, activation='softmax')(features)

model = Model(inputs=inputs, outputs=[outputs, adversary])

# Compile the model with two loss functions: categorical cross-entropy for the digit recognition task, binary cross-entropy for the adversarial domain adaptation task
model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model on the training set, passing both the input features and target variable to the model
model.fit(X_train, [y_train, np.zeros((len(y_train), 1))], batch_size=128, epochs=10)

# Evaluate the model on the validation set
# Evaluate the model on the validation set
print (model.evaluate(X_val, [y_val, np.zeros((len(y_val), 1))], verbose=0))
loss, digit_loss, domain_loss, digit_acc, domain_acc = model.evaluate(X_val, [y_val, np.zeros((len(y_val), 1))], verbose=0)

print(f'Digit recognition validation loss: {digit_loss:.4f}, digit recognition validation accuracy: {digit_acc:.4f}')
print(f'Adversarial domain adaptation validation loss: {domain_loss:.4f}, adversarial domain adaptation validation accuracy: {domain_acc:.4f}')

