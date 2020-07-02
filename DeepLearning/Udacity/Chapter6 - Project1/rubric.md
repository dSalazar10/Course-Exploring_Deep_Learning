# PROJECT SPECIFICATION: Dog Breed Classifier

## Step 1: Detect Humans
1) Assess the Human Face Detector
- The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected, human face.

## Step 2: Detect Dogs
1) Implement a Dog Detector

- Use a pre-trained VGG16 Net to find the predicted class for a given image. Use this to complete a dog_detector function below that returns True if a dog is detected in an image (and False if not).

2) Assess the Dog Detector

- The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected dog.

## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
1) Specify DataLoaders for the Dog Dataset

- Write three separate data loaders for the training, validation, and test datasets of dog images. 
- These images should be pre-processed to be of the correct size.

2) Describe your chosen procedure for preprocessing the data.

- Answer describes how the images were pre-processed and/or augmented.

3) Model Architecture

- The submission specifies a CNN architecture.

4) Outline the steps you took to get to your final CNN architecture and your reasoning at each step.

- Answer describes the reasoning behind the selection of layer types.

5) Train the Model

- Choose appropriate loss and optimization functions for this classification task. Train the model for a number of epochs and save the "best" result.

6) Test the Model

- The trained model attains at least 10% accuracy on the test set.

## Step 4: Create a CNN Using Transfer Learning
1) Model Architecture

- The submission specifies a model architecture that uses part of a pre-trained model.

2) Model Architecture

- The submission details why the chosen architecture is suitable for this classification task.

3) Train and Validate the Model

- Train your model for a number of epochs and save the result wth the lowest validation loss.

4) Test the Model

- Accuracy on the test set is 60% or greater.

5) Predict Dog Breed with the Model

- The submission includes a function that takes a file path to an image as input and returns the dog breed that is predicted by the CNN.

## Step 5: Write Your Algorithm
1) Write your Algorithm

- The submission uses the CNN from the previous step to detect dog breed. 
- The submission has different output for each detected image type (dog, human, other) and provides either predicted actual (or resembling) dog breed.

## Step 6: Test Your Algorithm
1) Test Your Algorithm on Sample Images!

- The submission tests at least 6 images, including at least two human and two dog images.

2) Question 6: Weaknesses and Improvements

- Submission provides at least three possible points of improvement for the classification algorithm.

## Suggestions to Make Your Project Stand Out!

(1) AUGMENT THE TRAINING DATA

Augmenting the training and/or validation set might help improve model performance.

(2) TURN YOUR ALGORITHM INTO A WEB APP

Turn your code into a web app using Flask!

(3) OVERLAY DOG EARS ON DETECTED HUMAN HEADS

Overlay a Snapchat-like filter with dog ears on detected human heads. You can determine where to place the ears through the use of the OpenCV face detector, which returns a bounding box for the face. If you would also like to overlay a dog nose filter, some nice tutorials for facial keypoints detection exist here.

(4) ADD FUNCTIONALITY FOR DOG MUTTS

Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned. The algorithm is currently guaranteed to fail for every mixed breed dog. Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, you will have to find a nice balance.

(5) EXPERIMENT WITH MULTIPLE DOG/HUMAN DETECTORS

Perform a systematic evaluation of various methods for detecting humans and dogs in images. Provide improved methodology for the face_detector and dog_detector functions.