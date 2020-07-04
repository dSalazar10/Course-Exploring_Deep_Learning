import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


class Network(nn.Module):
  def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, epochs=4, drop_p=0.5, learn_rate=0.001):
    super(Network, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.output_size = output_size
    self.n_layers = n_layers
    self.hidden_dim = hidden_dim
    self.epochs = epochs
    self.model = nn.ModuleDict({
      'embedding': nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True),
      'lstm': nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True),
      'classifier': nn.Sequential(
        nn.Dropout(p=drop_p),
        nn.Linear(hidden_dim, output_size),
        nn.Sigmoid()
      )
    })
    self.model.to(self.device)
    # Binary Cross Entropy Loss function; 0=neg and 1=pos
    self.criterion = nn.BCELoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)

  def forward(self, x, hidden):
    """
    Perform a forward pass of our model on some input and hidden state
    Input:
    * x: some input words
    * hidden: some hidden states
    Output:
    * return last sigmoid output and hidden state
    """
    # Pass input into the Input Embedding Layer
    embeds = self.model['embedding'](x.long())
    # Pass the Embeddings and
    lstm_out, hidden = self.model['lstm'](embeds, hidden)
    lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
    out = self.model['classifier'](lstm_out)
    # reshape to be batch_size 
    sig_out = sig_out.view(batch_size, -1)
    # get last batch of labels
    sig_out = sig_out[:, -1]
    return sig_out, hidden
  
  def init_hidden(self, batch_size):
    """
    Initializes hidden state
    Input:
    * batch_size: number of words
    Output:
    * returns a LSTM hidden layer
    """
    # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    # initialized to zero, for hidden state and cell state of LSTM
    weight = next(self.parameters()).data
    hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
    return hidden

  def train(self, batch_size, train_loader):
    """
    Train a model
    """
    # display - counter for displaying stats
    steps = 0
    # display - display to stats everytime steps % print_every == 0    
    print_every = 100
    # gradient clipping
    clip = 5
    # turn on gradients
    self.model.train()
    # train for some number of epochs
    for e in range(self.epochs):
      # initialize hidden state
      h = self.init_hidden(batch_size)
      # batch loop
      for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        # zero accumulated gradients
        self.model.zero_grad()
        # get the output from the model
        output, h = self.forward(inputs, h)
        # calculate the loss and perform backprop
        loss = self.criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.optimizer.step()
        # loss stats
        if steps % print_every == 0:
          # Get validation loss
          val_h = self.init_hidden(batch_size)
          val_losses = []
          self.model.eval()
          for inputs, labels in valid_loader:
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            val_h = tuple([each.data for each in val_h])
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output, val_h = self.forward(inputs, val_h)
            val_loss = self.criterion(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())
          self.model.train()
          print("Epoch: {}/{}...".format(e+1, epochs),
                "Step: {}...".format(steps),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)))

  def test(self, batch_size, test_loader):
    # Get test data loss and accuracy
    test_losses = [] # track loss
    num_correct = 0
    # init hidden state
    h = self.init_hidden(batch_size)
    self.model.eval()
    # iterate over test data
    for inputs, labels in test_loader:
      # Creating new variables for the hidden state, otherwise
      # we'd backprop through the entire training history
      h = tuple([each.data for each in h])
      inputs, labels = inputs.to(self.device), labels.to(self.device)
      # get predicted outputs
      output, h = self.forward(inputs, h)
      # calculate loss
      test_loss = self.criterion(output.squeeze(), labels.float())
      test_losses.append(test_loss.item())
      # convert output probabilities to predicted class (0 or 1)
      pred = torch.round(output.squeeze())  # rounds to the nearest integer
      # compare predictions to true label
      correct_tensor = pred.eq(labels.float().view_as(pred))
      correct = np.squeeze(correct_tensor.to(self.device).numpy())
      num_correct += np.sum(correct)
    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

  def predict(self, test_review, sequence_length=200):
    self.model.eval()
    # tokenize review
    test_ints = tokenize_review(test_review)
    # pad tokenized sequence
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)
    # initialize hidden state
    h = net.init_hidden(batch_size)
    feature_tensor = feature_tensor.to(self.device)
    # get the output from the model
    output, h = self.forward(feature_tensor, h)
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
    # print custom response
    if(pred.item()==1):
      print("Positive review detected!")
    else:
      print("Negative review detected.")
         

def pad_features(reviews_ints, seq_length):
  ''' Return features of review_ints, where each review is padded with 0's 
      or truncated to the input seq_length.
  '''
  # getting the correct rows x cols shape
  features = np.zeros((len(reviews_ints), seq_length), dtype=int)
  # for each review, I grab that review and 
  for i, row in enumerate(reviews_ints):
    features[i, -len(row):] = np.array(row)[:seq_length]
  return features


def tokenize_review(test_review):
  test_review = test_review.lower() # lowercase
  # get rid of punctuation
  test_text = ''.join([c for c in test_review if c not in punctuation])
  # splitting by spaces
  test_words = test_text.split()
  # tokens
  test_ints = []
  test_ints.append([vocab_to_int[word] for word in test_words])
  return test_ints

if 1:
  # read data from text files
  with open('data/reviews.txt', 'r') as f:
    reviews = f.read()
  with open('data/labels.txt', 'r') as f:
    labels = f.read()

  # get rid of punctuation
  reviews = reviews.lower() # lowercase, standardize
  all_text = ''.join([c for c in reviews if c not in punctuation])

  # split by new lines and spaces
  reviews_split = all_text.split('\n')
  all_text = ' '.join(reviews_split)

  # create a list of words
  words = all_text.split()

  ## Build a dictionary that maps words to integers
  counts = Counter(words)
  vocab = sorted(counts, key=counts.get, reverse=True)
  vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

  ## use the dict to tokenize each review in reviews_split
  ## store the tokenized reviews in reviews_ints
  reviews_ints = []
  for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

  # 1=positive, 0=negative label conversion
  labels_split = labels.split('\n')
  encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

  review_lens = Counter([len(x) for x in reviews_ints])
  ## remove any reviews/labels with zero length from the reviews_ints list.

  # get indices of any reviews with length 0
  non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

  # remove 0-length reviews and their labels
  reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
  encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

  seq_length = 200

  features = pad_features(reviews_ints, seq_length=seq_length)

  ## test statements - do not change - ##
  assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
  assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

  split_frac = 0.8

  ## split data into training, validation, and test data (features and labels, x and y)

  split_idx = int(len(features)*split_frac)
  train_x, remaining_x = features[:split_idx], features[split_idx:]
  train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

  test_idx = int(len(remaining_x)*0.5)
  val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
  val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

  ## print out the shapes of your resultant feature data
  print("\t\t\tFeature Shapes:")
  print("Train set: \t\t{}".format(train_x.shape), 
        "\nValidation set: \t{}".format(val_x.shape),
        "\nTest set: \t\t{}".format(test_x.shape))

  # create Tensor datasets
  train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
  valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
  test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

  # dataloaders
  batch_size = 50

  # make sure the SHUFFLE your training data
  train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
  valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
  test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

  # Instantiate the model w/ hyperparams
  vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
  output_size = 1
  embedding_dim = 400
  hidden_dim = 256
  n_layers = 2

  n = Network(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
  n.train(batch_size, train_loader)
  n.test(batch_size, test_loader)

  # negative test review
  test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'
  # positive test review
  test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'

  # test sequence padding
  seq_length=200
  features = pad_features(test_ints, seq_length)
  # test conversion to tensor and pass into your model
  feature_tensor = torch.from_numpy(features)

  # call function
  seq_length=200 # good to use the length that was trained on

  n.predict(test_review_neg, seq_length)

  # test code and generate tokenized review
  test_ints = tokenize_review(test_review_neg)