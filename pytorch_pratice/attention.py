import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F



class attention(nn.Module):
    def __init__(self) -> None :
        super().__init__()
        self.head_dim = 64
        self.q_head_num = 14
        self.kv_head_num = 2
        self.hidden_size = self.head_dim * self.q_head_num
        self.max_batch_size = 3
        self.max_seq_len = 512

        self.q_weight = nn.Linear(self.hidden_size, self.head_dim * self.q_head_num, bias = True, dtype = torch.float16)
        self.k_weight = nn.Linear(self.hidden_size, self.head_dim * self.kv_head_num, bias = True, dtype = torch.float16)
        self.v_weight = nn.Linear(self.hidden_size, self.head_dim * self.kv_head_num, bias = True, dtype = torch.float16)
        self.o_weight = nn.Linear(self.head_dim * self.q_head_num, self.hidden_size, bias = True, dtype = torch.float16)

        k_cache = torch.zeros(self.max_batch_size, self.kv_head_num, self.max_seq_len, self.head_dim, dtype= torch.float16)
        v_cache = torch.zeros(self.max_batch_size, self.kv_head_num, self.max_seq_len, self.head_dim, dtype= torch.float16)
        self.register_buffer("k_cache", k_cache, persistent=False)
        self.register_buffer("v_cache", v_cache, persistent=False)

    def forward(self,
                input_data : torch.Tensor,
                cache_pos : torch.Tensor,
                seq_len : int
                ) -> torch.Tensor:
        input_dim = input_data.shape[:-1]
        B = input_data.shape[0]
        divide_dim = (*input_dim, -1, self.head_dim)
        q_status = self.q_weight(input_data).view(divide_dim).transpose(-2, -3)
        k_status = self.k_weight(input_data).view(divide_dim).transpose(-2, -3)
        v_status = self.v_weight(input_data).view(divide_dim).transpose(-2, -3)

        self.k_cache[:B,:,cache_pos,:] = k_status
        self.v_cache[:B,:,cache_pos,:] = v_status

        k_extend = self.k_cache[:B,:,:seq_len,:]
        v_extend = self.v_cache[:B,:,:seq_len,:]

        attention = F.scaled_dot_product_attention(
            q_status,
            k_extend,
            v_extend,
            enable_gqa= True,
            is_causal = True if k_extend.shape[-2] == q_status.shape[-2] else False
            )
        output = self.o_weight(attention.transpose(-3,-2).reshape(input_data.shape))
        return output






        