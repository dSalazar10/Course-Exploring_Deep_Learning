import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import pickle as pkl

class Discriminator(nn.Module):
  def __init__(self, input_size, hidden_dim, output_size, learn_rate=0.002):
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, hidden_dim*4),
      nn.LeakyReLU(negative_slope=0.2),
      nn.Dropout(),
      nn.Linear(hidden_dim*4, hidden_dim*2),
      nn.LeakyReLU(negative_slope=0.2),
      nn.Dropout(),
      nn.Linear(hidden_dim*2, hidden_dim),
      nn.LeakyReLU(negative_slope=0.2),
      nn.Dropout(),
      nn.Linear(hidden_dim, output_size)
    )
    self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
    
  def forward(self, x):
    x = x.view(-1, 28*28)
    return self.model(x)

class Generator(nn.Module):
  def __init__(self, input_size, hidden_dim, output_size, learn_rate=0.002):
    super(Generator, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(input_size, hidden_dim),
      nn.LeakyReLU(negative_slope=0.2),
      nn.Dropout(),
      nn.Linear(hidden_dim, hidden_dim*2),
      nn.LeakyReLU(negative_slope=0.2),
      nn.Dropout(),
      nn.Linear(hidden_dim*2, hidden_dim*4),
      nn.LeakyReLU(negative_slope=0.2),
      nn.Dropout(),
      nn.Linear(hidden_dim*4, output_size),
      nn.Tanh()
    )
    self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)

  def forward(self, x):
    return self.model(x)

class Network(nn.Module):
  def __init__(self, g_params=[100, 32, 784], d_params=[784, 32, 1], epochs=100, learn_rate=0.002):
    super(Network, self).__init__()
    self.g_params = g_params
    self.d_params = d_params
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.epochs = epochs
    self.model = nn.ModuleDict({
      'D': Discriminator(d_params[0], d_params[1], d_params[2], learn_rate),
      'G': Generator(g_params[0], g_params[1], g_params[2], learn_rate)
    })
    self.model.to(self.device)
    self.criterion = nn.BCEWithLogitsLoss()
  def forward(self, x):
    return self.model(x)

  def train(self, train_loader):
    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []
    print_every = 1500
    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float().to(self.device)
    # train the network
    self.model.train()
    for e in range(self.epochs):
      for batch_i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        ## Important rescaling step ## 
        real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
        real_images = real_images.to(self.device)
        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================
        self.model['D'].optimizer.zero_grad()
        # 1. Train with real images
        # Compute the discriminator losses on real images 
        # smooth the real labels
        D_real = self.model['D'](real_images)
        d_real_loss = self.real_loss(D_real, smooth=True)
        # 2. Train with fake images
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float().to(self.device)
        fake_images = self.model['G'](z)
        fake_images = fake_images.to(self.device)
        # Compute the discriminator losses on fake images        
        D_fake = self.model['D'](fake_images)
        d_fake_loss = self.fake_loss(D_fake)
        # add up loss and perform backprop
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.model['D'].optimizer.step()
        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        self.model['G'].optimizer.zero_grad()
        # 1. Train with fake images and flipped labels
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float().to(self.device)
        fake_images = self.model['G'](z)
        fake_images = fake_images.to(self.device)
        # Compute the discriminator losses on fake images 
        # using flipped labels!
        D_fake = self.model['D'](fake_images)
        g_loss = self.real_loss(D_fake) # use real loss to flip labels
        # perform backprop
        g_loss.backward()
        self.model['G'].optimizer.step()
        # Print some loss stats
        if batch_i % print_every == 0:
          # print discriminator and generator loss
          print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                  e+1, self.epochs, d_loss.item(), g_loss.item()))
      ## AFTER EACH EPOCH##
      # append discriminator loss and generator loss
      losses.append((d_loss.item(), g_loss.item()))
      # generate and save sample, fake images
      self.model['G'].eval() # eval mode for generating samples
      samples_z = self.model['G'](fixed_z)
      samples.append(samples_z)
      self.model.train() # back to train mode
    # Save training generator samples
    save_samples(samples)
    # Save the trained model
    self.save_checkpoint()

  # Calculate losses
  def real_loss(self, D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
      # smooth, real labels = 0.9
      labels = torch.ones(batch_size)*0.9
    else:
      labels = torch.ones(batch_size) # real labels = 1
    # calculate loss
    D_out, labels = D_out.to(self.device), labels.to(self.device)
    loss = self.criterion(D_out.squeeze(), labels)
    return loss

  def fake_loss(self, D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    # calculate loss
    D_out, labels = D_out.to(self.device), labels.to(self.device)
    loss = self.criterion(D_out.squeeze(), labels)
    return loss

  def save_checkpoint(self, filepath='checkpoint.tar'):
    """
    This function is responsible for saving a copy of the model's data
    Input:
    * filepath: (Optional) PATH name using the convention *.tar
    """
    # Extract the data from the model: in_features and out_features for each linear layer,
    # the model's state dictionary, and the optimizer's state dictionary
    checkpoint = {'g_params': self.g_params,
                'd_params': self.d_params,
                'g_model_state_dict': self.model['G'].state_dict(),
                'd_model_state_dict': self.model['D'].state_dict(),
                'g_optimizer_state_dict': self.model['G'].optimizer.state_dict(),
                'd_optimizer_state_dict': self.model['D'].optimizer.state_dict()
    }
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
    # create a new network based on the old network's parameters
    n = Network(checkpoint['g_params'], checkpoint['d_params'])
    # load the model's state dicts
    n.model['G'].load_state_dict(checkpoint['g_model_state_dict'])
    n.model['D'].load_state_dict(checkpoint['d_model_state_dict'])
    # load the optimizer's state dicts
    n.model['G'].optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    n.model['D'].optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    # turn training off
    n.model.eval()
    return n

def view_samples(epoch, samples):
  """
  helper function for viewing a list of passed in sample images
  """
  fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
  for ax, img in zip(axes.flatten(), samples[epoch]):
    img = img.detach()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    img.cpu()
    im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')

def load_samples(filepath='data/samples.pkl'):
  # Load samples from generator, taken while training
  with open(path, 'rb') as f:
    return pkl.load(f)

def save_samples(samples, filepath='data/samples.pkl'):
  # Save training generator samples
  with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

if 0:
  %matplotlib inline
  import matplotlib.pyplot as plt
  # number of subprocesses to use for data loading
  num_workers = 0
  # how many samples per batch to load
  batch_size = 64
  # convert data to torch.FloatTensor
  transform = transforms.ToTensor()
  # get the training datasets
  train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
  # prepare data loader
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

  # Discriminator hyperparams
  # Size of input image to discriminator (28*28)
  input_size = 784
  # Size of discriminator output (real or fake)
  d_output_size = 1
  # Size of last hidden layer in the discriminator
  d_hidden_size = 32

  # Generator hyperparams
  # Size of latent vector to give to generator
  z_size = 100
  # Size of discriminator output (generated image)
  g_output_size = 784
  # Size of first hidden layer in the generator
  g_hidden_size = 32

  n = Network([z_size, g_hidden_size, g_output_size], [input_size, d_hidden_size, d_output_size], epochs=1)
  n.train(train_loader)

  # Load samples from generator, taken while training
  samples = load_samples()
  # -1 indicates final epoch's samples (the last in the list)
  view_samples(-1, samples)
  rows = 10 # split epochs into 10, so 100/10 = every 10 epochs
  cols = 6
  fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

  for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
      for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
          img = img.detach()
          ax.imshow(img.reshape((28,28)), cmap='Greys_r')
          ax.xaxis.set_visible(False)
          ax.yaxis.set_visible(False)
  # randomly generated, new latent vectors
  sample_size=16
  rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
  rand_z = torch.from_numpy(rand_z).float()

  n.model['G'].eval() # eval mode
  # generated samples
  rand_images = G(rand_z)

  # 0 indicates the first set of samples in the passed in list
  # and we only have one batch of samples, here
  view_samples(0, [rand_images])