'''
Creating the model, will be a deep Q network, either Dueling deep Q network or deep Q network

Simple linear neural network used

This neural network predicts the Q value for each possible action

'''

import torch
from torch import nn
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim_traffic_light, output_dim_on_off, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim,64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512) 
        self.layer5 = nn.Linear(512, 512)
        # # Dueling DQN predicts both the value of the state and the advantage of each possible action
        # # Best action should have advantage of 0
        ## Outputs are combined to generate the Q values
        self.output_1 = nn.Linear(512, output_dim_traffic_light)
        self.output_2 = nn.Linear(512, output_dim_on_off)

    def forward(self, x):
        x = F.relu6(self.layer1(x))
        x = F.relu6(self.layer2(x))
        x = F.relu6(self.layer3(x))
        x = F.relu6(self.layer4(x))
        x = F.relu6(self.layer5(x))
        output_1 = self.output_1(x)
        output_2 = self.output_2(x)
        return output_1, output_2
        
