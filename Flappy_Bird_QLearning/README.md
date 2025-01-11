
## Resources
### On-Policy vs Off-Policy
-https://www.geeksforgeeks.org/on-policy-vs-off-policy-methods-reinforcement-learning/
### CNN
-https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
-https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns
-https://medium.com/analytics-vidhya/understanding-convolution-operations-in-cnn-1914045816d4
-https://towardsdatascience.com/exploring-feature-extraction-with-cnns-345125cefc9a
### Actor-Critic
-https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/

![Flappy Bird CNN QLearning Diagram](Flappy_Bird_CNN_QLearning.drawio.png)rawio)

# CNN Architecture Details
## Preprocessing/Normalization
The preprocessing steps applied to the input images are as follows:  
Conversion to PIL Image: transforms.ToPILImage()
Grayscale Conversion: transforms.Grayscale(num_output_channels=1)
Resizing: transforms.Resize((84, 84))
Conversion to Tensor: transforms.ToTensor()

# CNN Architecture
The CNN architecture is defined in the CNN class. Here are the details:  
1. First Convolutional Layer:  
Input Channels: 1 (grayscale image)
Output Channels: 8
Kernel Size: 5x5
Stride: 1
Padding: 0
2. First Max Pooling Layer:  
Kernel Size: 2x2
Stride: 2
3. Second Convolutional Layer:  
Input Channels: 8
Output Channels: 16
Kernel Size: 5x5
Stride: 1
Padding: 0
4. Second Max Pooling Layer:  
Kernel Size: 2x2
Stride: 2
5. Third Convolutional Layer:  
Input Channels: 16
Output Channels: 32
Kernel Size: 5x5
Stride: 1
Padding: 0
6. Third Max Pooling Layer:  
Kernel Size: 2x2
Stride: 2
7. Fully Connected Layer:  
Input Features: 32 * 7 * 7 (after three 2x2 poolings)
Output Features: 12 (number of classes, represents state for Q-Learning Input)

# Hyperparameters
The hyperparameters used in the training process are:  
Q-Learning Hyperparameters:  
Learning Rate (alpha): 0.1
Discount Factor (gamma): 0.99
Exploration Rate (epsilon): 1.0
Exploration Decay (epsilon_decay): 0.995
Number of Episodes (num_episodes): 100
CNN Training Hyperparameters:  
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 10