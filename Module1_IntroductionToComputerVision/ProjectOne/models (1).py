# CNN Architecture for Facial Keypoints Detection

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I # can use the below import should you choose to initialize the weights of your Net


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__() # Initialize the Pytorch NN module, parent class of Net class.
        
        # Define a CNN architecture; Initialize custom layers
        
        # Convolution Layer 1: Input channel: 224x224 images. Output channel: 32. Filter: 5x5. Output Size: (224-5)/1 + 1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Pooling Layer 1: F:(2,2). Output Size: 220/2 = 110. Output Size: (32, 110, 110)
        self.pool1 = nn.MaxPool2d(2,2)
        
        # Convolution Layer 2: Input channel: 32. Output channel: 64. Filter: 3x3. Output Size: (110-3)/1 + 1 = 108
        self.conv2 = nn.Conv2d(32, 64, 5)
        # Pooling Layer 1: F:(2,2). Output Size: 108/2 = 54. Output Size: (64, 54, 54)
        self.pool2 = nn.MaxPool2d(2,2)# Already initialized above.
        
        # Fully Connected Layer 1: Input: 64*54*54 Ouput: 1024
        self.f1 = nn.Linear(64*54*54, 1024)
        
        # Drop Out Layer 1: prob: 0.4
        self.drop1 = nn.Dropout(p = 0.4)
        
        # Fully Connected Layer 2: Input: 1024 Ouput: 136 (68 keyoints in (x,y) format)
        self.f2 = nn.Linear(1024, 136)
        

        
    def forward(self, x):
        # Feed Forward Behavior where x is the input image
        
        x = self.pool1(F.relu(self.conv1(x))) # C1 -> P1
        x = self.pool2(F.relu(self.conv2(x))) # C2 -> P2
        
        x = x.view(x.size(0), -1) # Flatten
        
        x = F.relu(self.f1(x)) # LL 1
        x = self.drop1(x) # Dropout in between
        x = self.f2(x) # LL 2
        
        return x  # a modified x, having gone through all the layers of your model, should be returned
