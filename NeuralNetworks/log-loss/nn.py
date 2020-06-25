import numpy as np
import pandas as pd
from utilities import Utilities
from activation import Activation
act = Activation()
class NeuralNetwork:
  def __init__(self, W=np.array(0, dtype=float), b=0, epoch=10, learn_rate=0.01):
    """
    This is the constructor class
    Input:
    * W: an array of weights
    * b: the bias
    * epoch: the number of loops to fit model
    * learn_rate: the scale of gradient descent steps
    * cost: the array of regularization
    Ouput:
    * returns a log-loss neural network object
    """
    self.w_ = W
    self.b_ = b
    self.e_ = epoch
    self.l_ = learn_rate
    self.c_ = None
    np.random.seed(143)

  def l1_reg(self, loss, lambd=0.7):
    """
    This returns (loss + λ(|w_1| + |w_2| + ... + |w_n|))
    """
    return loss + lambd * np.sum(np.abs(self.w_))

  def l2_reg(self, loss, lambd=0.7):
    """
    This returns (loss + λ(w_1^2 + w_2^2 + ... + w_n^2))
    """
    return loss + lambd * np.sum(np.square(self.w_))

  def error_formula(self, m, y, output):
    """
    This calculates the standard cross-entropy loss function
    """
    return (-1.0/m) * np.sum( (1 - y) * np.log(1 - output) + y * np.log(output) )

  def error_term_formula(self, x, y, output):
    """
    This is the log-loss cross entropy function
    output = ŷ = W_i * x_i
    y = 1 or 0
    """
    # 1 - output is negative if y is 0 and output is high
    # 1 - output is positive if y is 1 and output is high
    return (y - output) * act.sigmoid_prime(x)

  def fit(self, features, targets, activation="relu"):
    """
    This calculates the new boundary line based on the weight, input, and learning rate

    Inputs:
    * X: array of features
    * y: array of targets
    """
    
    self.c_ = []
    # Get num_targets
    n_records, n_features = features.shape
    # Update weights to a sample of the normal distribution (1/(sqrt(n_features)))
    self.w_ = np.random.normal(scale=1 / n_features**.5, size=n_features)
    # Repeat until stopping criterion is satisfied
    for _ in range(self.e_):
      # Create a gradient array: ΔW = [w_1, w_2, ..., w_n]
      del_w = np.zeros(self.w_.shape)
      # Choose one sample from training set
      for x, y in zip(features.values, targets):
        # y = Wx+b
        h = np.dot(x, self.w_) + self.b_

        # Calculate the prediction sample
        if activation is "sig":
          # y_hat = sigmoid(h)
          sigmoid_output = act.sigmoid(h)
        if activation is "relu":
          # y_hat = relu(h)
          relu_output = act.relu(h)
        if activation is "tanh":
          # y_hat = tanh(h)
          tanh_output = act.tanh(h)

        # # Calculate error
        # if activation is "sig":
        #   self.c_.append(self.error_formula(n_features, y, sigmoid_output))
        # if activation is "relu":
        #   self.c_.append(self.error_formula(n_features, y, relu_output))
        # if activation is "tanh":
        #   self.c_.append(self.error_formula(n_features, y, tanh_output))
        
        # Calculate loss function for that prediction sample
        if activation is "sig":
          sigmoid_error_term = self.error_term_formula(x, y, sigmoid_output)
        if activation is "relu":
          relu_error_term = act.relu_prime(relu_output)
        if activation is "tanh":
          tanh_error_term = act.tanh_prime(tanh_output)

        # Calculate gradient from loss function
        if activation is "sig":
          del_w += sigmoid_error_term * x
        if activation is "relu":
          del_w += relu_error_term * x
        if activation is "tanh":
          del_w += tanh_error_term * x
        
      # Update model parameters based on gradient and learning rate
      self.w_ += self.l_ * del_w / n_records

  def predict(self, X):
    """
    This will provide the results. It calculates the Matrix
    Multiplication of the weights and the inputs. 
    Then it uses the sigmoid method to predict the results.
    Inputs:
    * X: this is the matrix of inputs
    Output:
    * returns 1 if calculation is > 0.5
    * returns 0 if calculation is < 0.5
    """
    # Returns 1 / (1 + np.exp(-(Wx)))
    return act.sigmoid(np.dot(X, self.w_))

  def score(self, X_t, y_t):
    """
    This will find out how accurate the fitted model is using the test data

    Input:
    * X_t: the training features
    * y_t: the training targets
    """
    # Make a prediction using the test data
    predictions = self.predict(X_t)
    
    # Calculate accuracy on test data
    accuracy = np.mean( (predictions > 0.5) == y_t)
    conf_matrix = pd.crosstab(y_t, predictions , rownames=['Actual'], colnames=['Predicted']) 
    return accuracy, conf_matrix, self.c_

# Test usage
if 1:
  """
  Sigmoid prediction accuracy: 0.720
  RelU prediction accuracy: 0.750
  TanH prediction accuracy: 0.710
  """
  ut = Utilities()

  # Pulling the data into a tableu
  data = pd.read_csv('student_data.csv')

  # Drill-down the rank column
  processed_data = ut.one_hot_encoder(data, "rank")

  # Scaling the columns
  processed_data['gre'] = processed_data['gre']/800
  processed_data['gpa'] = processed_data['gpa']/4.0

  # Split the data 2/3 train and 1/3 test
  train_data, test_data = ut.test_train_split(processed_data)
  
  # Splitting inputs and labels
  features = train_data.drop('admit', axis=1)
  targets = train_data['admit']
  features_test = test_data.drop('admit', axis=1)
  targets_test = test_data['admit']

  # Create a neural network for two activation functions
  sigmoid_clf = NeuralNetwork(np.array(0, dtype=float),0,1000,0.5)
  relu_clf = NeuralNetwork(np.array(0, dtype=float),0,1000,0.0001)
  tanh_clf = NeuralNetwork(np.array(0, dtype=float),0,100,0.001)

  # Train the model on the training data
  sigmoid_clf.fit(features, targets, "sig")
  relu_clf.fit(features, targets, "relu")
  tanh_clf.fit(features, targets, "tanh")

  # Test the model on the testing data
  sigmoid_accuracy, sigmoid_cm, sigmoid_cost = sigmoid_clf.score(features_test, targets_test)
  relu_accuracy, relu_cm, relu_cost = relu_clf.score(features_test, targets_test)
  tanh_accuracy, tanh_cm, tanh_cost = tanh_clf.score(features_test, targets_test)


  print("Sigmoid prediction accuracy: {:.3f}".format(sigmoid_accuracy))
  print(sigmoid_cm)
  print("RelU prediction accuracy: {:.3f}".format(relu_accuracy))  
  print(relu_cm)
  print("TanH prediction accuracy: {:.3f}".format(tanh_accuracy))  
  print(tanh_cm)