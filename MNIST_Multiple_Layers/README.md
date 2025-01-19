# MNIST Multiple Layers

This project demonstrates a multi-layer neural network for classifying the MNIST dataset, implemented from scratch without using frameworks like PyTorch. The goal is to understand the core concepts of neural networks by manually coding various components.

## Architecture
The neural network architecture includes:
- Input Layer: 784 neurons (28x28 pixels flattened)
- Hidden Layer: 100 neurons, using the Tanh activation function
- Output Layer: 10 neurons, using the Softmax activation function

## Data Processing
### Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

### Preprocessing
- **Loading Data**: The MNIST dataset is loaded and converted to numpy arrays.
- **Normalization**: Pixel values are normalized by dividing by 255.
- **One-Hot Encoding**: Labels are one-hot encoded to create a binary matrix representation.

## Model Initialization
Weights and biases are initialized using Xavier initialization to ensure the gradients do not vanish or explode during training.

## Forward Propagation
Forward propagation is implemented to calculate the activations of each layer with and without dropout to prevent overfitting.

## Activation Functions
- **Tanh**: Used in the hidden layer for non-linearity.
- **Softmax**: Used in the output layer to convert logits to probabilities.

## Loss Function
The cross-entropy loss function is used to measure the difference between the predicted probabilities and the true labels.

## Training
The training process involves multiple epochs where the model is trained using mini-batches of data. Both L1 and L2 regularization are applied to prevent overfitting.

## Evaluation
The model's accuracy is evaluated on the test dataset to measure its performance.

## Saving and Loading Model
Model parameters (weights and biases) are saved and loaded using pickle for persistence.

## Usage
1. **Load and preprocess data**:
   ```python
   train_X, train_Y = download_mnist(True)
   test_x, test_y = download_mnist(False)
   process_train_X, process_train_Y = process_data(train_X, train_Y)
   process_test_x, process_test_y = process_data(test_x, test_y)
   ```

2. **Initialize model**:
   ```python
   w1, b1, w2, b2 = xavier_init()
   ```

3. **Train model**:
   ```python
   for epoch in range(50):
       train_epoch(process_train_X, process_train_Y, w1, b1, w2, b2)
   ```

4. **Evaluate model**:
   ```python
   print("Accuracy: ", accuracy(process_test_x, process_test_y, w1, b1, w2, b2))
   ```

5. **Save model**:
   ```python
   save_model_parameters(w1, b1, w2, b2, "model_parameters.pkl")
   ```

6. **Load model**:
   ```python
   w1, b1, w2, b2 = load_model_parameters("model_parameters.pkl")
   ```
