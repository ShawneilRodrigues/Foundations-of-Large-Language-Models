import torch
import torch.nn as nn
import torch.optim as optim

class multiheadattention(nn.Module):
    def __init__(self,d_model: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv=nn.linear(d_model,3*d_model,bias=False)
        self.proj=nn.linear(d_model,d_model,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.trace_shapes = trace_shapes
    
    def forward(self,x: torch.Tensor):
        B,T,C=x.shape(-1)
        qkv=qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)  # (B,T,3,heads,dim)
        if self.trace_shapes:
            print("qkv view:", qkv.shape)
        q, k, v = qkv.unbind(dim=2)    
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)
        if self.trace_shapes:
            print("q:", q.shape, "k:", k.shape, "v:", v.shape)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,heads,T,T)
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask, float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        ctx = torch.matmul(w, v)
        ctx = torch.matmul(w, v)                  # (B,heads,T,dim)
        if self.trace_shapes:
            print("weights:", w.shape, "ctx:", ctx.shape)
        out = ctx.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,d_model)
        out = self.proj(out)
        if self.trace_shapes:
            print("out:", out.shape)
        return out, w
        