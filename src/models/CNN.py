import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import numpy as np

class CNN_classifier(nn.Module):

    def __init__(self , num_classes = 10):
        super(CNN_classifier , self).__init__()
        # input size = (3,32,32)
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=16 , kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16 , out_channels=32 , kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32 , out_channels=48 , kernel_size=3)  

        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(in_features=48*2*2 , out_features=128 )
        self.fc2 = nn.Linear(in_features=128 , out_features=10)

        self.dropout = nn.Dropout(0.2)


    def forward(self , x):

        x = self.pool(F.relu(self.conv1(x)))     #(16 , 15 , 15)
        x = self.pool(F.relu(self.conv2(x)))     #(32 , 6 , 6)
        x = self.pool(F.relu(self.conv3(x)))     #(16 ,  2, 2)

        # flattrn
        # print(x.shape)
        x = x.view(x.size(0) , -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
    


input_size = (3,32,32)

ip = arr = np.random.rand(3, 32, 32)
cnn = CNN_classifier(num_classes=10)
