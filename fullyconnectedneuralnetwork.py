import torch
import torch.nn as nn

class feedforward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().init()
        self.net=nn.Sequential(
            nn.Linear(d_model,mult*d_model),
            nn.Gelu(),
            nn.Linear(mult*d_model,d_model),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
