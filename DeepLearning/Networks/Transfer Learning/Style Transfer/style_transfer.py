from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import requests
from torchvision import transforms, models

class Network(nn.Module):
  def __init__(self, content_path, style_path, target_path, epoch=2000, alpha=1, beta=1e6):
    super(Network, self).__init__()
    # path where the output file will be saved
    self.target_path = target_path
    # number of updates to image
    self.epoch = epoch
    # It's recommended that you leave the content_weight = 1
    self.content_weight = alpha
    # set the style_weight to achieve the ratio you want
    self.style_weight = beta
    # get the "features" portion of VGG19 (we will not need the "classifier" portion)
    self.model = models.vgg19(pretrained=True).features
    # freeze all VGG parameters since we're only optimizing the target image
    for param in self.model.parameters():
      param.requires_grad_(False)
    # check if CUDA is available
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set model to cuda or cpu
    self.model.to(self.device)
    # load content image and set to cuda or cpu
    self.content = self.load_image(content_path).to(self.device)
    # load stylee image and set to cuda or cpu
    self.style = self.load_image(style_path, shape=self.content.shape[-2:]).to(self.device)

  def save_image(self, image):
    """
    This saves the output image to disk
    Input:
    * image: the image data to be saved
    """
    # creating a image object
    img = Image.open(self.target_path)
    # save a image using extension 
    return img.save('out.jpg')

  def load_image(self, img_path, max_size=400, shape=None):
    """
    Load in and transform an image, making sure the image
    is <= 400 pixels in the x-y dims.
    Input:
    * img_path: the path to the input image
    * max_size: (Optional) maximum image size
    * shape: shape of image; used to make style image same shape as content image
    Output:
    * returns a trasnformed PIL image
    """
    # if path is a url
    if "http" in img_path:
      response = requests.get(img_path)
      image = Image.open(BytesIO(response.content)).convert('RGB')
    else: # path is local
      image = Image.open(img_path).convert('RGB')
    # large images will slow down processing
    if max(image.size) > max_size:
      size = max_size
    else:
      size = max(image.size)
    if shape is not None:
      size = shape
    # resize the image, convert to a tensor, and normalize
    in_transform = transforms.Compose([transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

  
  def im_convert(self, tensor):
    """
    Helper function for un-normalizing an image and converting
    it from a Tensor image to a NumPy image for display
    Input:
    * tensor: the image tensor (content, style, target)
    Output:
    * returns a converted image
    """
    # convert to cpu, clone the image, and detach
    image = tensor.to("cpu").clone().detach()
    # convert to numpy array and remove single-dimensional 
    # entries from the shape of the array
    image = image.numpy().squeeze()
    # get a new view of the image
    image = image.transpose(1,2,0)
    # scale the image
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    # limit the values to a min of 0 and max of 1
    image = image.clip(0, 1)
    return image

  def get_features(self, image, layers=None):
    """ 
    Run an image forward through a model and get the features for 
    a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    Input:
    * image: PIL image
    * layers: (Optional) specify a layer
    Output:
    * returns the features of the input image
    """
    ## Mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in self.model._modules.items():
        # forward pass
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

  def gram_matrix(self, tensor):
    """ 
    Calculate the Gram Matrix of a given tensor 
    Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    Input:
    * tensor: feature map matrix (style, target)
    Output:
    * returns the Gram Matrix of the feature map
    """
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    return gram 

  def transfer(self, vis=False):
    """
    Get features of content and style images, calculates the gram matricies, creates
    a target image based off of the content image, calculates to Content Loss, Calculates
    the Style loss, calculates the Total Loss, and updates the image based on the results.
    This will optionally display the image every 400 iterations
    This will save the file to disk
    Input:
    * vis: (Optional) this controls the diplay of images
    Output:
    * None
    """
    # get content and style features only once before training
    content_features = self.get_features(self.content)
    style_features = self.get_features(self.style)
    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}
    # create a third "target" image and prep it for change
    # it is a good idea to start off with the target as a copy of our *content* image
    # then iteratively change its style
    target = self.content.clone().requires_grad_(True).to(self.device)
    # weights for each style layer 
    # weighting earlier layers more will result in *larger* style artifacts
    # notice we are excluding `conv4_2` our content representation
    style_weights = {'conv1_1': 1.,
                    'conv2_1': 0.75,
                    'conv3_1': 0.2,
                    'conv4_1': 0.2,
                    'conv5_1': 0.2}
    # Add target image parameters to optimizer
    optimizer = optim.Adam([target], lr=0.003)
    # For displaying the target image, intermittently
    if vis is True:
      show_every = 400
    # Update image e number of times (default = 2000)
    for e in range(self.epoch):
        # get the features from your target image
        target_features = self.get_features(target)
        # calculate the content loss: ((∑(T_c - C_c)^2) / 2) == (mean((T_c - C_c)^2))
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        # calculate the style loss
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            # calculate the Gram Matrix of the target image
            target_gram = self.gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            # get the "style" style representation
            style_gram = style_grams[layer]
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # normalize to the style loss
            style_loss += layer_style_loss / (d * h * w)
        # calculate the total loss: α(L_content) + β(L_style)
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        # Zero the existing gradients
        optimizer.zero_grad()
        # Backward pass
        total_loss.backward()
        # Gradient Descent Step
        optimizer.step()
        # Display intermediate images and print the loss
        if vis is True and e % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(self.im_convert(target))
            plt.show()
    # Display content and final, target image
    if vis is True:
      _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
      ax1.imshow(self.im_convert(self.content))
      ax2.imshow(self.im_convert(target))
    # save the image to disk
    self.save_image(self.im_convert(target))

if 0:
  n = Network('images/octopus.jpg', 'images/hockney.jpg', 'images/out.jpg')
  n.transfer()
  