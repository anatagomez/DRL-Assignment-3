import torch
import torch.nn as nn
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the correct output size from convolutional layers
        conv_out_size = self._get_conv_out(input_shape)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _get_conv_out(self, shape):
        # Pass a dummy tensor through conv layers to get output shape
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        # Make sure input has the right shape
        if len(x.size()) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        conv_out = self.conv(x).view(x.size()[0], -1)
        
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # Combine value and advantage to get Q values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))