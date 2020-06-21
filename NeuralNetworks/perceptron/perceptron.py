import numpy as np
from numpy import genfromtxt


def stepFunction(t):
    """
    This function determines if Wx+b >= 0?
    Input:
    * t: this is the result of the score function
    Output:
    * returns 1 if t is positive
    * returns 0 if t is negative
    """
    if t >= 0:
        return 1
    return 0

def softmax(L):
    """
    Takes as input a list of numbers, and returns the list of values given by the softmax function

    Input:
    * L: array of inputs
    """
    return np.divide (1, 1 + np.exp(L))

def prediction(X, W, b):
    """
    This will provide the results. It calculates the Matrix
    Multiplication of the weights and the inputs and adds the
    biased.
    Inputs:
    * X: this is the matrix of inputs
    * W: this is the matrix of weights
    * b: this is the scalar for the bias
    Output:
    * returns 1 if calculation is positive
    * returns 0 if calculation is negative
    """
    result = (np.matmul(X,W)+b)[0]
    print("SOFTMAX = {}".format(softmax(result)))
    return stepFunction(result)

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    """
    This calculates the new boundary line based on the weight, input, and learning rate

    Inputs:
    * X: array of inputs
    * y: array of labels
    * W: array of weights
    * b: bias
    * learn_rate: normalizer variable (α)
    Output:
    * returns the modified array of weights and the bias scalar
    """
    # for each element in the input array
    for i in range(len(X)):
        # Calculates the step function of the results of the input
        y_hat = prediction(X[i],W,b)
        # If prediction = 0
        if y[i]-y_hat == 1:
            # Change weight to weight + α * input
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            # Change b to b + α
            b += learn_rate
        # If prediction = 1
        elif y[i]-y_hat == -1:
            # Change weight to weight - α * input
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            # Change b to b - α
            b -= learn_rate
    # Return new boundary
    return W, b

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    """
    This function runs the perceptron algorithm repeatedly on the dataset,
    and returns a few of the boundary lines obtained in the iterations,
    for plotting purposes.

    Input:
    * X: array of (x,y) coordinates
    * y: array of labels
    Output:
    * returns an array of scalars that represent boundary lines
    """
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

def main():
    np.random.seed(42)
    # Read CSV file into a numpy array
    my_data = genfromtxt('data.csv', delimiter=',')
    # Transform and upack each row (column)
    x0, x1, y = my_data.T
    # Group X[0] and X[1] into a single array and transform back
    X = np.vstack((x0, x1)).T
    # Train Perceptron
    print(trainPerceptronAlgorithm(X, y))

main()