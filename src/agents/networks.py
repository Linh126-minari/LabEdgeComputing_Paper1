import torch
import torch.nn as nn

class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNetwork, self).__init__()
        
        # Feature Extraction Layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Kết hợp theo Eq 29 trong 
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals