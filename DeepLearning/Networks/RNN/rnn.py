import numpy as np
import torch
from torch import nn
import collections
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self, tokens, n_hidden=256, n_layers=2, drop_p=0.5, lr=0.001):
    super(Network, self).__init__()
    self.n_layers = n_layers
    self.n_hidden = n_hidden
    self.lr = lr
    # creating character dictionaries
    self.chars = tokens
    self.int2char = dict(enumerate(self.chars))
    self.char2int = {ch: ii for ii, ch in self.int2char.items()}
    self.model = nn.ModuleDict({
      'lstm': nn.LSTM(
        len(tokens), 
        n_hidden, 
        n_layers, 
        dropout=drop_p, 
        batch_first=True
      ),
      'dropout': nn.Dropout(drop_p),
      'fc': nn.Linear(n_hidden, len(tokens))
    })
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    self.criterion = nn.CrossEntropyLoss()
    self.train_on_gpu = torch.cuda.is_available()
    if self.train_on_gpu:
      self.model.cuda()

  def forward(self, x, hidden):
    ''' Forward pass through the network. 
        These inputs are x, and the hidden/cell state `hidden`. '''
    # Get the outputs and the new hidden state from the lstm
    r_output, hidden = self.model['lstm'](x, hidden)
    # Pass through a dropout layer
    out = self.model['dropout'](r_output)
    # Stack up LSTM outputs using view
    # you may need to use contiguous to reshape the output
    out = out.contiguous().view(-1, self.n_hidden)
    # Put x through the fully-connected layer
    out = self.model['fc'](out)
    # return the final output and the hidden state
    return out, hidden
  
  def train(self, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
  
      Arguments
      ---------
      
      net: CharRNN network
      data: text data to train the network
      epochs: Number of epochs to train
      batch_size: Number of mini-sequences per mini-batch, aka batch size
      seq_length: Number of character steps per mini-batch
      lr: learning rate
      clip: gradient clipping
      val_frac: Fraction of data to hold out for validation
      print_every: Number of steps for printing training and validation loss
  
    '''
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
      # initialize hidden state
      h = self.model.init_hidden(batch_size)
      
      for x, y in self.get_batches(data, batch_size, seq_length):
        counter += 1
        # One-hot encode our data and make them Torch tensors
        x = self.one_hot_encode(x, n_chars)
        inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
        if self.train_on_gpu:
          inputs, targets = inputs.cuda(), targets.cuda()
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        # zero accumulated gradients
        self.optimizer.zero_grad()
        # get the output from the model
        output, h = self.forward(inputs, h)
        # calculate the loss and perform backprop
        loss = self.criterion(output, targets.view(batch_size*seq_length).long())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.optimizer.step()
        # loss stats
        if counter % print_every == 0:
          # Get validation loss
          val_h = self.model.init_hidden(batch_size)
          val_losses = []
          self.model.eval()
          for x, y in self.get_batches(val_data, batch_size, seq_length):
            # One-hot encode our data and make them Torch tensors
            x = self.one_hot_encode(x, n_chars)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            val_h = tuple([each.data for each in val_h])
            
            inputs, targets = x, y
            if(self.train_on_gpu):
              inputs, targets = inputs.cuda(), targets.cuda()

            output, val_h = net(inputs, val_h)
            val_loss = self.criterion(output, targets.view(batch_size*seq_length).long())
        
            val_losses.append(val_loss.item())
          
          self.model.train() # reset to train mode after iterationg through validation data
          
          print("Epoch: {}/{}...".format(e+1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.4f}...".format(loss.item()),
                "Val Loss: {:.4f}".format(np.mean(val_losses)))
  
  def predict(self, net, char, h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''
    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = self.one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)
    if(self.train_on_gpu):
      inputs = inputs.cuda()
    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)
    # get the character probabilities
    p = F.softmax(out, dim=1).data
    if(self.train_on_gpu):
      p = p.cpu() # move to cpu
    # get top characters
    if top_k is None:
      top_ch = np.arange(len(net.chars))
    else:
      p, top_ch = p.topk(top_k)
      top_ch = top_ch.numpy().squeeze()
    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())
    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h

  def save_checkpoint(self, filepath='checkpoint.tar'):
    """
    This function is responsible for saving a copy of the model's data
    Input:
    * filepath: (Optional) PATH name using the convention *.tar
    """
    # Extract the data from the model: in_features and out_features for each linear layer,
    # the model's state dictionary, and the optimizer's state dictionary
    checkpoint = {'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_layers': np.append([each.out_channels for each in n.model[0:-7:3]], np.append(n.model[11].in_features, n.model[11].out_features)),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}
    # Save the model's data to a tar file
    torch.save(checkpoint, filepath)

  def load_checkpoint(self, filepath='checkpoint.tar'):
    """
    This function is responsible for loading a copy of a model's data
    that was previosuly stored to disk
    Input:
    * filepath: (Optional) PATH name using the convention *.tar
    """
    # Load the data
    checkpoint = torch.load(filepath)
    # Create an ordered dictionary to hold the modified model's state dictionary
    new_state_dict = collections.OrderedDict()
    # Append 'model.' to each of the model_state_dict elements
    for k, v in checkpoint['model_state_dict'].items():
      # prepend 'model.' to each of the keys
      name = ''.join(('model.', k))
      # copy the key-value pair into the new container
      new_state_dict[name] = v
    # create a new network based on the old network's parameters
    model = Network(checkpoint['input_size'],
                    checkpoint['output_size'],
                    checkpoint['hidden_layers'])
    # load the model's state dictionary of the old into the new
    model.load_state_dict(new_state_dict)
    # load the optimizer's state dictionary of the old into the new
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # turn training off
    model.eval()
    return model

    def init_hidden(self, batch_size):
      """
      Initializes hidden state
      """
      # Create two new tensors with sizes n_layers x batch_size x n_hidden,
      # initialized to zero, for hidden state and cell state of LSTM
      weight = next(self.parameters()).data
      
      if (self.train_on_gpu):
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
      else:
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                    weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
      
      return hidden

    def get_batches(self, arr, batch_size, seq_length):
      """
      Create a generator that returns batches of size
          batch_size x seq_length from arr.
          
          Arguments
          ---------
          arr: Array you want to make batches from
          batch_size: Batch size, the number of sequences per batch
          seq_length: Number of encoded chars in a sequence
      """
      batch_size_total = batch_size * seq_length
      # total number of batches we can make
      n_batches = len(arr)//batch_size_total
      # Keep only enough characters to make full batches
      arr = arr[:n_batches * batch_size_total]
      # Reshape into batch_size rows
      arr = arr.reshape((batch_size, -1))
      # iterate through the array, one sequence at a time
      for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
          # from the start to end of array -> shift x over by one
          y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
          # If at end of array, set last element to the fist element
          y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

  def sample(self, net, size, prime='The', top_k=None):
    if self.train_on_gpu:
      net.cuda()
    else:
      net.cpu()
    net.eval() # eval mode
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
      char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)
    # Now pass in the previous character and get a new one
    for ii in range(size):
      char, h = predict(net, chars[-1], h, top_k=top_k)
      chars.append(char)
    return ''.join(chars)

  def one_hot_encode(self, arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

  def load_text(self, path):
    with open(path, 'r') as f:
      return f.read()

  def encode_text(self, text):
    # encode the text and map each character to an integer and vice versa
    # Tokenization
    chars = tuple(set(text))
    # maps integers to characters
    int2char = dict(enumerate(chars))
    # maps characters to unique integers
    char2int = {ch: ii for ii, ch in int2char.items()}
    # encode the text
    return np.array([char2int[ch] for ch in text])

if 0:
  # Load Data
  text = load_text('data/anna.txt')
  # Encode Data
  encoded = encode_text(text)
  # Data Pre-processing

  # define and print the net
  n_hidden=512
  n_layers=2

  net = CharRNN(chars, n_hidden, n_layers)
  batch_size = 128
  seq_length = 100
  n_epochs = 20 # start smaller if you are just testing initial behavior

  # train the model
  train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)
