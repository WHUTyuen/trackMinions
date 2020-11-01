import torch.nn as nn
import torch.nn.functional as F

import torch

class CNNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.ReLU(True),
            nn.Conv2d(16,32,3,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,128,3,1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,64,3,1),
            nn.ReLU(True)
        )


        self.fc_layer = nn.Sequential(
            nn.Linear(10*10*64,128),
            nn.ReLU(True),
            nn.Linear(128,5)
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = torch.reshape(x, (-1,10*10*64))
        x = self.fc_layer(x)
        position = F.relu(x[:,0:4])
        flag = torch.sigmoid(x[:,4])
        return position , flag

