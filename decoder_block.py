import torch
import torch.nn as nn
from multihead_attention import multiheadattention
from fullyconnectedneuralnetwork import feedforward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        self.ln1=nn.LayerNorm(d_model)
        self.attn=multiheadattention(d_model, n_head, dropout)
        self.ln2=nn.LayerNorm(d_model)
        self.fnn=feedforward(d_model, mult=4, dropout=dropout)
    
    def forward(self,x):
        x=x+self.attn(self.ln1(x))[0]
        x=x+self.fnn(self.ln2(x))
        return x
