import re
from collections import Counter
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim

class NegativeSamplingLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input_vectors, output_vectors, noise_vectors):
    """
    Custom Loss function, only update a small subset of the weightsat once
    Input:
    * input_vectors: correct input words
    * output_vectors: correct output words
    * noise_vectors: incorrect words
    Output:
    * returns the average batch loss
    """
    batch_size, embed_size = input_vectors.shape
    # Input vectors should be a batch of column vectors
    input_vectors = input_vectors.view(batch_size, embed_size, 1)
    # Output vectors should be a batch of row vectors
    output_vectors = output_vectors.view(batch_size, 1, embed_size)
    # log-sigmoid of the inner product of the output word vector and
    # the input word vector. This is the correct target loss
    out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
    out_loss = out_loss.squeeze()
    # log-sigmoid of the inner product of the noise word vector and
    # the output word vector. This is the noisy target loss
    noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
    # sum the losses over the sample of noise vectors
    noise_loss = noise_loss.squeeze().sum(1)
    # negate and sum correct and noisy log-sigmoid losses
    return -(out_loss + noise_loss).mean()

class Network(nn.Module):
  def __init__(self, n_vocab, n_embed, noise_dist=None, embedding_dim=300, epochs=5, learn_rate=0.003):
    super(Network, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.embedding_dim = embedding_dim
    self.epochs = epochs
    self.n_vocab = n_vocab
    self.n_embed = n_embed
    self.noise_dist = noise_dist
    # Define the model
    self.model = nn.ModuleDict({
      'in_embed': nn.Embedding(n_vocab, n_embed),
      'out_embed': nn.Embedding(n_vocab, n_embed)
    })
    # Set the weights to improve conversion (negative sampling)
    self.model['in_embed'].weight.data.uniform_(-1, 1)
    self.model['out_embed'].weight.data.uniform_(-1, 1)
    # Set to GPU if available
    self.model.to(self.device)
    # Set the loss function to the custom funcion defined above
    self.criterion = NegativeSamplingLoss()
    # Set the optimization function
    self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
    
  def forward_input(self, input_words):
    """
    Pass input words into Input Embedding Layer
    Input:
    * input_words: correct input vector
    Output:
    * returns the input embeddings
    """
    input_vectors = self.model['in_embed'](input_words)
    return input_vectors
  
  def forward_output(self, output_words):
    """
    Pass output words into Output Embedding Layer
    Input:
    * output_words: correct output vector
    Output:
    * returns the output embeddings
    """
    output_vectors = self.model['out_embed'](output_words)
    return output_vectors
  
  def forward_noise(self, batch_size, n_samples):
    """
    Generate noise vectors with shape (batch_size, n_samples, n_embed)
    Input:
    * batch_size: size of words in a window
    * n_samples: count of noise samples (negative sampling)
    Output:
    * returns the incorrect noisy embeddings
    """
    # gets noisy samples from a noise distribution
    if self.noise_dist is None:
      # default to a uniform distribution
      noise_dist = torch.ones(self.n_vocab)
    else:
      noise_dist = self.noise_dist
    # sample words from our noise distribution and push to GPU if available
    noise_words = torch.multinomial(noise_dist, batch_size * n_samples, replacement=True).to(self.device)
    # passes the noisy words through the output embedding layer, then reshapes the view
    noise_vectors = self.model['out_embed'](noise_words).view(batch_size, n_samples, self.n_embed)
    return noise_vectors

  def train(self, freq, train_words, int_to_vocab):
    """
    Train the model on the data set and print stats ever 1500 loops
    Input:
    * freq: dict with key=word and val=frequency
    * train_words: the trimmed list of words
    * int_to_vocab: dict with key=emb_idx and val=word
    Output:
    * None
    """
    # calculate the frequency of each word in the dict
    word_freqs = np.array(sorted(freqs.values(), reverse=True))
    # calculate the unigram noise distribution
    unigram_dist = word_freqs/word_freqs.sum()
    # initialize the model's noise distribution
    self.noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))
    # display - display to stats everytime steps % print_every == 0
    print_every = 1500
    # display - counter for displaying stats
    steps = 0
    # train for some number of epochs
    for e in range(self.epochs):
      # get our input and target batches
      for input_words, target_words in get_batches(train_words, 512):
        steps += 1
        # convert the batches into longs and put on GPU if available
        inputs = torch.LongTensor(input_words).to(self.device)
        targets = torch.LongTensor(target_words).to(self.device)
        # input, output, and noise vectors
        input_vectors = self.forward_input(inputs)
        output_vectors = self.forward_output(targets)
        n_samples = 5
        noise_vectors = self.forward_noise(inputs.shape[0], n_samples)
        # negative sampling loss
        loss = self.criterion(input_vectors, output_vectors, noise_vectors)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # display - loss stats (epoch, loss, and validation similarities)
        if steps % print_every == 0:
          print("Epoch: {}/{}".format(e+1, self.epochs))
          print("Loss: ", loss.item()) # avg batch loss at this point in training
          valid_examples, valid_similarities = self.cosine_similarity()
          _, closest_idxs = valid_similarities.topk(6)
          valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
          for ii, valid_idx in enumerate(valid_examples):
            closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
            print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
          print("...\n")

  def cosine_similarity(self, valid_size=16, valid_window=100):
    """
    Returns the cosine similarity of validation words with words in the embedding matrix.
    Here we're calculating the cosine similarity between some random words and 
    our embedding vectors. With the similarities, we can look at what words are
    close to our random words.
    sim = (a . b) / |a||b|
    Input:
    * valid_size: (Optional)
    * valid_window: (Optional)
    Output:
    * returns the validation samples and the similarities
    """
    # get the embeddings from the model's Input Embedding Layer 
    embed_vectors = self.model['in_embed'].weight
    # calculate the magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    # pick N words from our ranges: common words = (0,window) and uncommon words = (1000,1000+window)
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(self.device)
    # pass the random words into the Input Embedding Layer
    valid_vectors = self.model['in_embed'](valid_examples)
    # calculate the similarities: (a . b) / |b|
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
    return valid_examples, similarities

def preprocess(text):
  """
  Converts any puctuation into tokens, creates a counter dictionary of the data,
  and trims the data dictionary of words with occurrances less than five
  Input:
  * text: a text file containing words
  Output:
  * returns a trimmed counter dictionary of tokenized words 
  """
  # Replace punctuation with tokens so we can use them in our model
  text = text.lower()
  text = text.replace('.', ' <PERIOD> ')
  text = text.replace(',', ' <COMMA> ')
  text = text.replace('"', ' <QUOTATION_MARK> ')
  text = text.replace(';', ' <SEMICOLON> ')
  text = text.replace('!', ' <EXCLAMATION_MARK> ')
  text = text.replace('?', ' <QUESTION_MARK> ')
  text = text.replace('(', ' <LEFT_PAREN> ')
  text = text.replace(')', ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?', ' <QUESTION_MARK> ')
  # text = text.replace('\n', ' <NEW_LINE> ') # Uncomment this if your text has newlines
  text = text.replace(':', ' <COLON> ')
  words = text.split()
  # Get a dictionary with the words and their occurances
  word_counts = Counter(words)
  # Remove all words with  5 or fewer occurences
  trimmed_words = [word for word in words if word_counts[word] > 5]
  return trimmed_words

def create_lookup_tables(words):
  """
  Create lookup tables for vocabulary
  Input:
  * words: list of words in a text
  Output:
  * returns Two dictionaries: vocab_to_int, int_to_vocab
  """
  # Get a dictionary with the words and their occurances
  word_counts = Counter(words)
  # sorting the words from most to least frequent in text occurrence
  sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
  # create a dictionary with integers as the key
  int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
  # create a dictionary with the words as the key
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab

def get_target(words, idx, window_size=5):
  """
  Get a list of words in a window around an index
  Input:
  * words: the dictionary of words
  * idx: the index for the word of interest
  * window_size: (Optional) the number of words to include
  Output:
  * returns the words of interest that fall into the window
  """
  # Define a random range: [1, C]
  R = np.random.randint(1, window_size+1)
  # words behind the words of interest; stopping at element 0
  start = idx - R if (idx - R) > 0 else 0
  # words in front of the words of interest
  stop = idx + R
  # words of interest
  target_words = words[start:idx] + words[idx+1:stop+1]
  return list(target_words)

def get_batches(words, batch_size, window_size=5):
  """
  Create a generator of word batches as a tuple (inputs, targets)
  Input:
  * words: the dictionary of words
  * batch_size: count of words
  * window_size: (Optional) the number of words to include
  Output:
  * returns batches of input and target data for the model
  """
  # calculate the total number of batches in the text data
  n_batches = len(words)//batch_size
  # only include complete batches of data
  words = words[:n_batches*batch_size]
  # iterate over the words, one batch at a time
  for idx in range(0, len(words), batch_size):
    x, y = [], []
    # get a batch of words
    batch = words[idx:idx+batch_size]
    # iterate over the batch, one words at a time
    for ii in range(len(batch)):
      # get a target group
      batch_x = batch[ii]
      # get the words of interest that fall into the window
      batch_y = get_target(batch, ii, window_size)
      # set y to a single row of values
      y.extend(batch_y)
      # set x to one row of values, equivalent in length as y
      x.extend([batch_x]*len(batch_y))
    yield x, y

if 1:
  # Load Data
  with open('data/text8') as f:
    text = f.read()
  # Tokenize the text file into a counter dictionary
  words = preprocess(text)
  # Create two dictionaries based on the preprocessed data
  vocab_to_int, int_to_vocab = create_lookup_tables(words)
  # Create a list of words
  int_words = [vocab_to_int[word] for word in words]
  # Subsampling of Data
  threshold = 1e-5
  word_counts = Counter(int_words)
  total_count = len(int_words)
  # Calculate the frequency for each word in the vocabulary
  freqs = {word: count/total_count for word, count in word_counts.items()}
  # Calculate the discard propability for each word in the vocabulary
  drop_p = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
  # discard words based on the subsampling equation
  train_words = [word for word in int_words if random.random() < (1 - drop_p[word])]
  # Instatiate the network
  n = Network(len(vocab_to_int), 300, epochs=1)
  # Train the network
  n.train(freqs, train_words, int_to_vocab)