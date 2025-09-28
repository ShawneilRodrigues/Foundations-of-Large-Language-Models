import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-8):
        self.eps=eps
        self.weight=nn.Parameter(torch,Ones(dim))

    def forward(x: torch.Tensor):
        rms=x.pow(2).mean(dim=-1,keepdim=True).add(self.eps).sqrt()
        return (x/rms)*self.weight