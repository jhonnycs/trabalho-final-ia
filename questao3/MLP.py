import torch
import matplotlib.pyplot as plt

class MLP(torch.nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        
        self.hidden = torch.nn.Linear(n_inputs, n_hidden)
        self.output = torch.nn.Linear(n_hidden, 1)
        
    
    def forward(self, x):
        x = self.hidden(x)
        x = torch.relu(x)

        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x
