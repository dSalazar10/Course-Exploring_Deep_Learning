import numpy as np
import pandas as pd

class NeuralNetwork(object):
  def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
    print("\n\nInit Function\n", "input_nodes = {}".format(input_nodes), "hidden_nodes = {}".format(hidden_nodes), "output_nodes = {}".format(output_nodes))
    # Set number of nodes in input, hidden and output layers.
    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes

    # Initialize weights
    self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                    (self.input_nodes, self.hidden_nodes))
    self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                    (self.hidden_nodes, self.output_nodes))
    self.lr = learning_rate
    
    self.activation_function = lambda x : 1 / (1 + np.exp(-x))
    self.activation_prime = lambda x : self.activation_function(x) * (1 - self.activation_function(x))

  def train(self, features, targets):
    ''' Train the network on batch of features and targets. 
    
        Arguments
        ---------
        
        features: 2D array, each row is one data record, each column is a feature
        targets: 1D array of target values
    
    '''
    print("train")
    n_records = features.shape[0]
    delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
    delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
    for _ in range(iterations):
      for X, y in zip(features, targets):
        final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
        # Implement the backproagation function below
        delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                    delta_weights_i_h, delta_weights_h_o)
      self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

  def forward_pass_train(self, X):
    ''' Implement forward pass here 
      
        Arguments
        ---------
        X: features batch

    '''
    #### Implement the forward pass here ####
    ### Forward pass ###
    hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer

    hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

    final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer

    final_outputs = self.activation_function(final_inputs) # signals from final output layer

    return final_outputs, hidden_outputs

  def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
    ''' Implement backpropagation
      
        Arguments
        ---------
        final_outputs: output from forward pass
        y: target (i.e. label) batch
        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers

    '''
    #### Implement the backward pass here ####
    ### Backward pass ###
    # Calculate the network's prediction error: y−ŷ
    error = y - final_outputs

    # Calculate error term for the output unit
    # δ^0 = (y−ŷ)f'(final_outputs)
    output_error_term =  error * self.activation_prime(final_outputs)

    # Propagate the errors to the hidden layer

    # Calculate the hidden layer's contribution to the error
    # δ^output * W_j
    hidden_error = output_error_term * self.weights_input_to_hidden

    # Calculate the error term for the hidden layer
    # # δ_j^hidden = ∑(δ^output * W_j * f'(h_j))
    prime = np.array(self.activation_prime(hidden_outputs))
    hidden_error_term = np.sum(hidden_error * prime)

    # Weight step (input to hidden)
    delta_weights_i_h += hidden_error_term * X[:, None]
    # Weight step (hidden to output)
    delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)
    return delta_weights_i_h, delta_weights_h_o

  def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
    ''' Update weights on gradient descent step
      
        Arguments
        ---------
        delta_weights_i_h: change in weights from input to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers
        n_records: number of records

    '''
    # update hidden-to-output weights with gradient descent step
    self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
     # update input-to-hidden weights with gradient descent step
    self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

  def run(self, features):
    ''' Run a forward pass through the network with input features 
    
        Arguments
        ---------
        features: 1D array of feature values
    '''
    print("run")
    #### Implement the forward pass here ####
    hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
    hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
    final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
    final_outputs = self.activation_function(final_inputs) # signals from final output layer
    return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 100
learning_rate = 0.1
hidden_nodes = 2
output_nodes = 1

if 0:
  # # Data pre-processing
  rides = pd.read_csv('Bike-Sharing-Dataset/hour.csv')
  # # drill down into the data
  # one_hot_data = ut.one_hot_encoder(rides, ['season', 'weathersit', 'mnth', 'hr', 'weekday'])
  # # drop data that might not be useful to the NN
  # one_hot_data = one_hot_data.drop(['instant', 'dteday', 'workingday', 'atemp'], axis=1)
  # # Scale the data
  # quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
  # scaled_data, scaled_features = ut.scale_data(one_hot_data, quant_features)
  # # Split the data
  # train_data, val_data, test_data = ut.test_train_split(scaled_data)
  # target_fields = ['cnt', 'casual', 'registered']
  # # Separate the train_data into features and targets
  # train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
  # # Separate the val_data into features and targets
  # val_features, val_targets = val_data.drop(target_fields, axis=1), val_data[target_fields]
  # # Separate the test_data into features and targets
  # test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
