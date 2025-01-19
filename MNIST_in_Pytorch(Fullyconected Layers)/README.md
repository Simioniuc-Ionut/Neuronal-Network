# MNIST in PyTorch (Fully Connected Layers)

This project implements a fully connected neural network for classifying the MNIST dataset using PyTorch.

## Overview

The notebook `MNIST_in_Pytorch(Fullyconected Layers)/MNIST.ipynb` contains code to:
- Load and preprocess the MNIST dataset
- Define and train a neural network with fully connected layers
- Evaluate the model's performance
- Save and load the model

## Dataset

We use the MNIST dataset which consists of 60,000 training images and 10,000 test images of handwritten digits.

## Data Preprocessing

### Transformations:
- **Training Transformations**:
  - Random Rotation
  - Random Affine Transformation
  - Conversion to Tensor
  - Normalization
- **Testing Transformations**:
  - Conversion to Tensor
  - Normalization

## Model Architecture

The model is defined in the `NN` and `NN_2` classes with the following architecture:
- **Input Layer**: 784 neurons (28x28 flattened image)
- **Hidden Layer 1**: 128 neurons, ReLU activation
- **Hidden Layer 2**: 64 neurons, ReLU activation
- **Output Layer**: 10 neurons (one for each digit 0-9)

The `NN_2` class also includes dropout layers for regularization.

## Training

The model is trained using the Adam optimizer and CrossEntropyLoss. Training is performed for 10 epochs with a batch size of 64.

## Evaluation

The accuracy of the model is evaluated on both the training and test datasets.

## Usage

```python
# Load dataset
train_dataset = MNISTDataset('data/train', True, False)
test_dataset = MNISTDataset('data/test', False, False)

# Initialize model
model = NN(input_size=784, hidden1=128, hidden2=64, num_classes=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.evalTraining()
model.train(criterion, optimizer, num_epochs=10)

# Check accuracy
check_accuracy(model)
```

## Saving and Loading Model

```python
# Save model weights
torch.save(model.state_dict(), 'model_weights.pth')

# Load model weights
model.load_state_dict(torch.load('model_weights.pth'))

# Save the entire model
torch.save(model, 'complete_model.pth')

# Load the entire model
model = torch.load('complete_model.pth')
```
