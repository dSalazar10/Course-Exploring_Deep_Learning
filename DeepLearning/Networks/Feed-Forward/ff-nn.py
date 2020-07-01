import torch
from torch import optim, nn
import numpy as np
import time
import collections

class Network(nn.Module):
  """
  This class is a Feed-Forward Neural network. It has three hidden layers and one output layer
  It uses ReLU for its activation function and the Negative Log Liklihood for its loss function.
  This was designed as a classification system, and as such, uses Log Softmax for the output layer.
  The dropout probability is set to 20%. The epoch is set to 5 and the learning rate is set to
  0.003 for testing purposes. The optimizer used is the first-order gradient-based optimization
  of Stochastic Gradient Descent
  The data it was designed with was the Fashion MNIST data set
  Take a look at the models.ipynb for graphs and explantions of why thing work the way they do.
  """
  def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2, epochs=5, learn_rate=0.003):
    super(Network, self).__init__()
    # The in_features for the first hidden layer
    self.input_size = input_size
    # The out_feature for the last hidden layer
    self.output_size = output_size
    # The number of training cycles
    self.epochs = epochs
    # Linear model with Dropout, ReLU, and LogSoftmax
    self.model = nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                        nn.ReLU(),
                        nn.Dropout(p=drop_p),
                        nn.Linear(hidden_layers[0], hidden_layers[1]),
                        nn.ReLU(),
                        nn.Dropout(p=drop_p),
                        nn.Linear(hidden_layers[1], hidden_layers[2]),
                        nn.ReLU(),
                        nn.Dropout(p=drop_p),
                        nn.Linear(hidden_layers[2],output_size),
                        nn.LogSoftmax(dim=1))
    # Negative Log Liklihood pairs with LogSoftmax
    self.criterion = nn.NLLLoss()
    # Define the first-order gradient-based optimization of Stochastic Gradient Descent
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)

  def forward(self, x):
    """
    Forward pass sends x through the sequential model, which has the
    weights and bias built into them
    Input:
    * x: the collection of elements in a record
    """
    return self.model(x)
      
  def fit(self, trainloader, valloader):
    """
    This function is responsible for training the model
    Input:
    * trainloader: The MINST training data loader
    * valloader: The MINST validation data loader
    """
    # Initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    # Debug - number of images trained
    trained_so_far = 0
    # Debug - time stamp when function was executed
    start = time.time()
    # Debug - total loss for training and validation data
    train_losses, val_losses = [], []
    # count controlled loop based on model's epoch
    for e in range(self.epochs):
      # Debug - training loss for each inner loop
      running_loss = 0
      # count controlled loop based on n_records of training data
      for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        # Clear out the parameters of the optimizer
        self.optimizer.zero_grad()
        # Forward pass
        output = self.forward(images)
        # Calculate the loss with the output and the labels
        loss = self.criterion(output, labels)
        # Backward pass
        loss.backward()
        # Gradient Step
        self.optimizer.step()
        # Debug - update number of images trained
        trained_so_far += len(labels)
        # Debug - update the training loss
        running_loss += loss.item() * images.size(0)
      else: # once inner loop completes, print debug data
        # Debug - validation loss
        val_loss = 0
        # Turn off gradient for CPU/GPU speed boost
        with torch.no_grad():
          # Set model to evaluation mode
          self.model.eval()
          # count controlled loop based on n_records in validation data
          for images, labels in valloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            # Forward pass
            log_ps = self.forward(images)
            # Calculate the loss with the output and the labels
            loss = self.criterion(log_ps, labels)
            # Calculate the accuracy of the loss
            ps = torch.exp(log_ps)
            # Get the largest probability values per record
            _, top_class = ps.topk(1, dim=1)
            # Compare to the list of Ground Truth
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the validation loss
            val_loss += loss * images.size(0)
        # Set the model back to train mode
        self.model.train()
        # Debug - append the training loss for each epoch to the list
        running_loss = running_loss/len(trainloader.sampler)
        # Debug - append the validation loss for each epoch to the list
        val_loss = val_loss/len(valloader.sampler)
        # Debug - calculate the time it took to complete one epoch
        elapsed_time = float(time.time() - start)
        # Debug - calculate the number of images trainer per second in one epoch
        images_per_second = trained_so_far / elapsed_time if elapsed_time > 0 else 0
        # Debug - display debug information
        print("Epoch: {}/{}.. ".format(e+1, self.epochs),
            "Training loss: {:.3f}..".format( running_loss ),
            "Validation Loss: {:.3f}..".format( val_loss ),
            "Speed(images/sec): {:.2f}..".format( images_per_second ))
        # save model if validation loss has decreased
        if val_loss <= valid_loss_min:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            val_loss))
          self.save_checkpoint()
          valid_loss_min = val_loss
    # Debug - display the time it took for the total training to complete
    print("Total elapse time: {:.2f} seconds".format(float(time.time() - start)))
    # Load the best model back into the current state
    self.load_checkpoint()

  def test(self, testloader):
    """
    This function is responsible for testing the trained model on data that it has not
    seen before. 
    Input:
    * testloader: MNIST training data loader
    """
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)

    # Debug - testing loss for each inner loop
    running_loss = 0
    # Debug - number of images trained
    trained_so_far = 0
    # Debug - time stamp when function was executed
    start = time.time()
    # Turn off gradient for CPU/GPU speed boost
    with torch.no_grad():
      # Set model to evaluation mode
      self.model.eval()
      # count controlled loop based on n_records of testing data
      for images, labels in testloader:
        # Debug - update number of images trained
        trained_so_far += len(labels)
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        # Clear out the parameters of the optimizer
        self.optimizer.zero_grad()
        # Forward pass
        output = self.forward(images)
        # Calculate the loss with the output and the labels
        loss = self.criterion(output, labels)
        # Calculate the accuracy of the loss
        ps = torch.exp(output)
        # Get the largest probability values per record
        _, top_class = ps.topk(1, dim=1)
        # Compare to the list of Ground Truth
        equals = top_class == labels.view(*top_class.shape)
        # Debug - update the testing loss
        running_loss += loss.item() * images.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(labels)):
          target = labels.data[i]
          class_correct[target] += correct[i].item()
          class_total[target] += 1
    # Debug - calculate avg test loss
    running_loss = running_loss/len(testloader.sampler)
    print("Test Loss: {:.6f}\n".format(running_loss))
    for i in range(10):
      if class_total[i] > 0:
        print("Test Accuracy of {}: {:.2f}% ({:}/{:})".format(i, 
                    100 * class_correct[i] / class_total[i],
                    torch.sum(class_correct[i]).int(), 
                    torch.sum(class_total[i]).int() ))
      else:
          print("Test Accuracy of {:.2f}: N/A (no training examples)".format(classes[i]))
    print("\nTest Accuracy (Overall): {:.2f}% ({}/{})".format(100. * torch.sum(class_correct) / torch.sum(class_total),
                        torch.sum(class_correct).int(), 
                        torch.sum(class_total).int() ))
    # Set the model back to train mode
    self.model.train()

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
                'hidden_layers': [each.out_features for each in self.model[0:-2:3]],
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

if 0:
  from torchvision import datasets, transforms

  # Define a transform to normalize the data
  transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))])
  # Download and load the training data
  trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

  # Download the test data
  data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

  # Split the test data in half
  testset, valset = torch.utils.data.random_split(data, [5000, 5000])

  # Load the test and validation data
  testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
  valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

  # Create a new network
  n = Network(784, 10, [256, 128, 64])

  # Train your network
  n.fit(trainloader, valloader)

  # Test your network
  n.test(testloader)