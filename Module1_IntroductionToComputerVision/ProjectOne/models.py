## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel: after pool: 32-5 + 1 = 28 -> so (32, 14, 14)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # 2. Second convolutional Layer
        # 32 inputs, 64 outputs, kernel size 3.
        # after pool: 14-3 + 1 = 12 -> so (64, 6, 6)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # 3. MaxPool Layer
        self.pool = nn.MaxPool2d(2,2)
        
        # 4. Linear Layer One --> 
        self.f1 = nn.Linear(387200, 42)
        
        # 5. A drop out layer between Dense Layers
        self.df1 = nn.Dropout(p=0.4)
        
        # 6. Linear Layer Two
        self.f2 = nn.Linear(42, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Two Linear layers with dropout in between
        x = F.relu(self.f1(x))
        x = self.df1(x)
        x = self.f2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
