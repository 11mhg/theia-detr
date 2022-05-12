import torch
from torch import nn
from torch.nn import functional as F



class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation = F.relu):
        super().__init__()
        
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        layers = []
        
        if num_layers == 1:
            layers.append(
                nn.Linear(in_dim, out_dim)
            )
        else:
            for i in range(0, self.num_layers):
                layer_in = in_dim if i == 0 else hidden_dim
                layer_out = out_dim if i == self.num_layers-1 else hidden_dim
                layers.append(nn.Linear(layer_in, layer_out))
        
        self.layers = nn.ModuleList(layers)
        self.activation = activation
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x