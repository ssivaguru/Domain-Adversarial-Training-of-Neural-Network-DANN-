import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ITERATIONS = 6000
LEARNING_RATE = 0.07

def one_hotencoder(Y):
    one_hotencoder_Y = np.zeros((Y.size, Y.max() + 1)) # Initialize an array with dimensions of Amount of Total Labels by Amount of Distinct Labels (10 digits in this case) 
    one_hotencoder_Y[np.arange(Y.size), Y] = 1 # For each row in the zeroed array, set the value at the corresponding digit as an index to 1 
    one_hotencoder_Y = one_hotencoder_Y.T
    return one_hotencoder_Y

def sigmoid(x):
    # Sigmoid activation function (returns a number between 0 and 1)
    return 1 / (1 + np.exp(-x))

def define_neurons(X, Y):
    Y = one_hotencoder(Y) # Convert the labels into 
    input_layer_neurons = X.shape[0] # size of input layer
    hidden_layer_neurons = 10 # hidden layer of size 10
    ouput_layer_neurons = Y.shape[0] # size of output layer
    return (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)
    

def initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons):

    # Initialize the weights and biases for the nodes in the hidden layer
    weights_hidden = np.random.randn(hidden_layer_neurons, input_layer_neurons) * 0.1 # Random weights
    biases_hidden = np.zeros((hidden_layer_neurons, 1)) # Zeroed biases

    # Initialize the weights and biases for the nodes in the output layer
    weights_output = np.random.randn(ouput_layer_neurons, hidden_layer_neurons) * 0.1 # Random weights
    biases_output = np.zeros((ouput_layer_neurons, 1)) # Zeroed biases

    parameters = {"weights_hidden": weights_hidden,
                  "biases_hidden": biases_hidden,
                  "weights_output": weights_output,
                  "biases_output": biases_output}
    
    return parameters
    
def forward_propagation(X, parameters):
    # Extract weights and biases
    weights_hidden, biases_hidden, weights_output, biases_output = parameters['weights_hidden'], parameters['biases_hidden'], parameters['weights_output'], parameters['biases_output']

    # Raw output of hidden layer 
    unactivated_hidden = np.dot(weights_hidden, X) + biases_hidden

    # Tanh activated output of hidden layer
    activated_hidden = np.tanh(unactivated_hidden)
    
    # Raw output of output layer
    unactivated_output = np.dot(weights_output, activated_hidden) + biases_output

    # Sigmoid activated output of output layer
    activated_output = sigmoid(unactivated_output)

    outputs = {"unactivated_hidden": unactivated_hidden,
               "activated_hidden": activated_hidden,
               "unactivated_output": unactivated_output,
               "activated_output": activated_output}
    
    return activated_output, outputs

def loss_fn(activated_output, Y, parameters):
    m = Y.shape[1] 

    Y = one_hotencoder(Y)

    # Calculate Costs
    logs = np.multiply(np.log(activated_output), Y) + np.multiply((1 - Y), np.log(1 - activated_output))
    cost = - np.sum(logs) / m
    cost = float(np.squeeze(cost))
                                    
    return cost

def backward_propagation(parameters, outputs, X, Y):
    m = X.shape[1]
    
    Y = one_hotencoder(Y)

    # Extract the weights, biases and neuron outputs
    weights_hidden = parameters['weights_hidden']
    weights_output = parameters['weights_output']
    activated_hidden = outputs['activated_hidden']
    activated_output = outputs['activated_output']
   
    # Backprop calculations 
    dunactivated_output = activated_output - Y
    dweights_output = (1 / m) * np.dot(dunactivated_output, activated_hidden.T)
    dbiases_output = (1 / m) * np.sum(dunactivated_output, axis=1, keepdims=True)
    dunactivated_hidden = np.multiply(np.dot(weights_output.T, dunactivated_output), 1 - np.power(activated_hidden, 2))
    dweights_hidden = (1 / m) * np.dot(dunactivated_hidden, X.T) 
    dbiases_hidden = (1 / m) * np.sum(dunactivated_hidden, axis=1, keepdims=True)
    
    gradients = {"dweights_hidden": dweights_hidden, "dbiases_hidden": dbiases_hidden, "dweights_output": dweights_output,"dbiases_output": dbiases_output}
    
    return gradients

def gradient_descent(parameters, gradients):
    global LEARNING_RATE
    # Extract weights and biases
    weights_hidden = parameters['weights_hidden']
    biases_hidden = parameters['biases_hidden']
    weights_output = parameters['weights_output']
    biases_output = parameters['biases_output']
   
    # Extract gradients to apply 
    dweights_hidden = gradients['dweights_hidden']
    dbiases_hidden = gradients['dbiases_hidden']
    dweights_output = gradients['dweights_output']
    dbiases_output = gradients['dbiases_output']

    # Apply gradients to weights and biases
    weights_hidden = weights_hidden - LEARNING_RATE * dweights_hidden
    biases_hidden = biases_hidden - LEARNING_RATE * dbiases_hidden
    weights_output = weights_output - LEARNING_RATE * dweights_output
    biases_output = biases_output - LEARNING_RATE * dbiases_output
    
    # New weights and biases
    parameters = {"weights_hidden": weights_hidden, "biases_hidden": biases_hidden,"weights_output": weights_output,"biases_output": biases_output}
    
    return parameters

def model_fn(X, Y, hidden_layer_neurons, num_iterations):
    # Define the # of neurons for each layer based on input and output configurations
    (input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons) = define_neurons(X, Y)

    # Initialize weights and biases for the amount of neurons specified previously 
    parameters = initialize_parameters(input_layer_neurons, hidden_layer_neurons, ouput_layer_neurons)
    
    # Epochs for model_fn building 
    for i in range(0, num_iterations):
        # Get output of the model_fn with inputs and current parameters
        activated_output, outputs = forward_propagation(X, parameters)

        # Calculate the cost 
        cost = loss_fn(activated_output, Y, parameters)

        # Calculate new gradients 
        gradients = backward_propagation(parameters, outputs, X, Y)

        # Apply gradients and calculate new parameters
        parameters = gradient_descent(parameters, gradients)

        print("{}/{} (cost = {})".format(i + 1, num_iterations, cost))

    return parameters

def prediction(parameters, X):
    # Obtain output of model_fn given the test data 
    activated_output, outputs = forward_propagation(X, parameters)
    return np.argmax(activated_output, 0)

print('Initializing...')

np.random.seed(3)

# Read MNIST Dataset in from CSV file
print('Reading CSV Data...')
data = pd.read_csv('/Users/akk0018/Documents/ML/MLProject/mnist_train.csv')

# Convert data to a numpy array
print('Converting Data...')
data = np.array(data)

# Get dimensions of input data
m, n = data.shape

# Randomize sequencing of input data
np.random.shuffle(data) 

print("Splitting Data in Training and Testing...")
# Isolate first 500 digits for testing/validation
test_data = data[0:1000].T
Y_test = test_data[0] # Testing Labels
X_test = test_data[1:n] # Testing Features
X_test = X_test / 255 # Normalize pixel data

## Isolate rest of digits for training
train_data = data[1000:m].T
Y_train = train_data[0] # Training Labels
X_train = train_data[1:n] # Training Features
X_train = X_train / 255 # Normalize pixel data

# Transpose Labels for compatibility with NN
Y_train = Y_train.T
Y_test = Y_test.T

# Convert Labels to Numpy Array
Y_train = np.array([Y_train])
Y_test = np.array([Y_test])

# Train the model_fn and save the weights and biases
print("Building model_fn...")
parameters = model_fn(X_train, Y_train, 10, ITERATIONS)

# Weights + Biases after training
weights_hidden = parameters['weights_hidden']
biases_hidden = parameters['biases_hidden']
weights_output = parameters['weights_output']
biases_output = parameters['biases_output']




######### Below section cuurently to verfiy manually how the model_fn is working ####

print("model_fn Complete!")

# Do some visual tests to show user how the model_fn predicts various examples
tests = input("Enter the amount of visual tests you would like to do: ")

for i in range(0, int(tests)):
    current_image = X_train[:, i, None] # Get the digit at index 'i'
    label = Y_train[0][i] # Get the actual digit to compare our prediction with 
    predictions = prediction(parameters, current_image) # Get the model_fn's prediction with the chosen digit
    
    print("Prediction: " + str(predictions[0]))
    if predictions[0] == label:
        print("Correct")
    else: 
        print("Incorrect (Actual: " + str(label) + ")")
   
    current_image = current_image.reshape((28, 28)) * 255 # Convert the normalized digit pixels back to a plottable image
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


predictions = prediction(parameters, X_test) # Get all the predictions by the model_fn given the test data set

correct_predictions = np.sum(predictions == Y_test[0]) # Correct predictions where the predicted digit equals the actual digit
total_digits = Y_test[0].size # Total amount of test digits

print('')
print('Correct Predictions: ' + str(correct_predictions)) 
print('Total Digits Tested: ' + str(total_digits))
print('Accuracy: ' + str(np.round((correct_predictions / total_digits) * 100, 1)) + '%') # Calculate accuracy of the model_fn 
