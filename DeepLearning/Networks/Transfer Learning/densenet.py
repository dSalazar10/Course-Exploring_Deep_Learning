import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import time
import collections
import numpy as np

class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.model = models.densenet121(pretrained=True)
    for param in self.model.features.parameters():
      param.requires_grad = False
    self.model.classifier = nn.Sequential(nn.Dropout(p=0.2),
                            nn.Linear(1024, 500),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(500, 2),
                            nn.LogSoftmax(dim=1))
    if self.train_on_gpu:
      self.model.cuda()
    self.criterion = nn.NLLLoss()
    self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)
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
    # count controlled loop based on model's epoch
    for e in range(self.epochs):
      # Debug - training loss for each inner loop
      running_loss = 0
      # count controlled loop based on n_records of training data
      for images, labels in trainloader:
        # move tensors to GPU if CUDA is available
        if self.train_on_gpu:
            images, labels = images.cuda(), labels.cuda()
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
      else: # once inner loop completes, evaluate
        # Debug - validation loss
        val_loss = 0
        # Turn off gradient for CPU/GPU speed boost
        with torch.no_grad():
          # Set model to evaluation mode
          self.model.eval()
          # count controlled loop based on n_records in validation data
          for images, labels in valloader:
            # move tensors to GPU if CUDA is available
            if self.train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            # Forward pass
            log_ps = self.forward(images)
            # Calculate the loss with the output and the labels
            loss = self.criterion(log_ps, labels)
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
    running_loss = 0.0
    # Debug - number of images trained
    trained_so_far = 0
    # Turn off gradient for CPU/GPU speed boost
    with torch.no_grad():
      # Set model to evaluation mode
      self.model.eval()
      # count controlled loop based on n_records of testing data
      for images, labels in testloader:
        # move tensors to GPU if CUDA is available
        if self.train_on_gpu:
            images, labels = images.cuda(), labels.cuda()
        # Debug - update number of images trained
        trained_so_far += len(labels)
        # Clear out the parameters of the optimizer
        self.optimizer.zero_grad()
        # Forward pass
        output = self.forward(images)
        # Calculate the loss with the output and the labels
        loss = self.criterion(output, labels)
        # Debug - update the testing loss
        running_loss += loss.item() * images.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(labels.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not self.train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(len(labels)):
          target = labels.data[i]
          class_correct[target] += correct[i].item()
          class_total[target] += 1
    # Debug - calculate avg test loss
    running_loss = running_loss/len(testloader.sampler)
    print("Test Loss: {:.6f}\n".format(running_loss))
    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
      if class_total[i] > 0:
        print("Test Accuracy of {}: {:.2f}% ({:}/{:})".format(classes[i], 
                    100 * class_correct[i] / class_total[i],
                    torch.sum(class_correct[i]), 
                    torch.sum(class_total[i])))
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

if 1:
  data_dir = 'Cat_Dog_data'
  train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

  test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

  # Pass transforms in here, then run the next cell to see how the transforms look
  train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
  data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
  
  split_size = len(data) / 2
  testset, valset = torch.utils.data.random_split(data, [split_size, split_size])

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
  valid_loader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

  # Create a new network
  n = Network()

  # Train your network
  n.fit(train_loader, valid_loader)

  # Test your network
  n.test(test_loader)