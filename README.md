# Transfer Learning Classifier & Anime GAN Generator

Transfer Learning Approach for Image Classification


Overview
This project uses transfer learning with MobileNetV2 to classify images into two categories.
Dataset Structure

Training Data: Subfolders represent class labels, each containing corresponding images.
Test Data: Contains images named sequentially (e.g., image_1.jpg, image_2.jpg).
Approach

1. Data Preprocessing
Verified and resized images to 236x236, normalizing pixel values.
Split training data into training and validation sets (70%/30%).

2. Model Architecture
Base Model: MobileNetV2 pre-trained on ImageNet.
Custom Layers: Added pooling, dense layers, and dropout for binary classification.
Fine-tuned the last 20 layers of MobileNetV2.
3. Training and Testing
Used ImageDataGenerator for data splitting.
Computed class weights to handle imbalance.
Compiled the model with Adam optimizer and categorical crossentropy loss.
Saved the best model using ModelCheckpoint.
Made predictions on the test set and saved results in tl_submission.csv.


GAN Training for Anime Face Generation


Overview
This project implements a Generative Adversarial Network (GAN) to generate anime-style face images. The GAN consists of a generator and discriminator network trained on a dataset of anime face images.
Dataset Structure
Dataset: Folder containing images of anime faces.
Images are resized to 64x64 pixels during preprocessing.

Approach

1. Data Preprocessing
Applied transformations:
Resized images to 64x64 pixels.
Center-cropped images.
Normalized pixel values to the range [-1, 1] for compatibility with the Tanh activation function in the generator.
Used torchvision.transforms for preprocessing and DataLoader for batching.

2. Model Architecture
Generator:
Takes a latent vector of size 128 as input.
Uses transpose convolution layers to generate 64x64x3 images.
Activation functions:
ReLU for intermediate layers.
Tanh for the output layer.
Discriminator:
Uses convolution layers to classify images as real or fake.
Activation functions:
LeakyReLU for intermediate layers.
Sigmoid for the output layer.

3. Training
Alternately trained the generator and discriminator:
Discriminator:
Trained to maximize the accuracy of distinguishing real and fake images.
Loss function: Binary Cross-Entropy Loss.
Generator:
Trained to generate images that can fool the discriminator.
Loss function: Binary Cross-Entropy Loss.
Optimizers:
Adam optimizer for both networks with learning rates:
Generator: 0.0005
Discriminator: 0.0001
Beta values: (0.5, 0.999).
Saved generated samples during training for visualization.

4. Visualization and Output
Generated images are saved at each epoch in a generated folder.
Sampled and displayed generated images using fixed latent vectors to monitor progress.
Files
gan_training.py: Script for training the GAN.
generated/: Folder containing images generated during training.
G.pth: Saved generator model weights.
D.pth: Saved discriminator model weights.


