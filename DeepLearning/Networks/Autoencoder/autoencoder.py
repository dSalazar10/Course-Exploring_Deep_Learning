import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),     
                                    nn.Conv2d(16, 4, 3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2)
                                    )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(4, 16, 2, stride=2),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(16, 1, 2, stride=2),
                                    nn.Sigmoid()
                                    )
        # specify loss function
        self.criterion = nn.MSELoss()
        # specify loss function
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def forward(self, x):
        x = self.encoder(x)        
        return self.decoder(x)

    def fit(self, train_loader):
        # number of epochs to train the model
        n_epochs = 30

        for epoch in range(1, n_epochs+1):
            # monitor training loss
            train_loss = 0.0
            
            ###################
            # train the model #
            ###################
            for data in train_loader:
                # _ stands in for labels, here
                # no need to flatten images
                images, _ = data
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = self.forward(images)
                # calculate the loss
                loss = self.criterion(outputs, images)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item()*images.size(0)
                    
            # print avg training statistics 
            train_loss = train_loss/len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, 
                train_loss
                ))



# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 20

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# initialize the NN
model = Network()
