import torch
import torch.nn as nn
import torch.nn.functional as F








class attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 896
        self.head_num = 7
        self.head_dim = 64

        self.q_w = nn.Linear(self.hidden_size, self.head_num * self.head_dim, dtype=torch.float16)
        self.k_w = nn.Linear(self.hidden_size, self.head_num * self.head_dim, dtype=torch.float16)
        self.v_w = nn.Linear(self.hidden_size, self.head_num * self.head_dim, dtype=torch.float16)
        self.o_w = nn.Linear(self.head_num * self.head_dim, self.hidden_size, dtype=torch.float16)

    def forward(
        self,
        x:torch.Tensor,
        cos_sin : tuple[torch.Tensor, torch.Tensor]      
    ):
        shape1 = x.shape[:-1]
        shape1 = (*shape1, -1, self.head_dim)
        q = self.q_w(x).view(shape1).transpose(1,2)
        k = self.k_w(x).view(shape1).transpose(1,2)
        v = self.v_w(x).view(shape1).transpose(1,2)
        cos,sin = cos_sin
        q, k = RoPE_function(q, k, cos, sin)
        attention = F.softmax(q @ k.transpose(-1, -2)/self.head_dim ** 0.5, dim = -1);
        out = (attention @ v).transpose(1,2).contiguous.view(x.size(0),x.size(1),-1)
        return slef.o_w(out)
