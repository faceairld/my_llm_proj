import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch.utils.cpp_extension import load
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
cu_file_path = os.path.join(current_dir, "csrc", "rmsnorm_kernel.cu")
my_attention_path = os.path.join(current_dir,"csrc", "my_decode_attention.cu")

custom_rmsnorm_cuda = load(
    name = "my_rmsnorm_cuda",
    sources=[cu_file_path],
    verbose=True
)

custom_attention_cuda = load(
    name = "my_attention_cuda",
    sources=[my_attention_path],
    verbose=True
)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMSnormal(nn.Module):
    def __init__(self, dim : int, eps : float = 1e-6, add_unit_offset : bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16)) 
        self.add_unit_offset = add_unit_offset

    # def _norm(self, x : torch.Tensor):
    #     return x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps)
    
    # def forward(self, x : torch.Tensor):
    #     nvtx.range_push("RMSNorm")
    #     output = self._norm(x.float())
    #     if self.add_unit_offset:
    #         output = output * (1 + self.weight)
    #     else:
    #         output = output * self.weight
    #     nvtx.range_pop()
    #     return output.type_as(x)
    def forward(self, x: torch.Tensor):
        nvtx.range_push("RMSNorm")
        out = custom_rmsnorm_cuda.forward(x, self.weight, self.eps) # type: ignore
        nvtx.range_pop()
        return out
    



#generate RoPE cos sin
class QwenRoPEfactorgenerate(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 64
        self.base = 1000000
        i = torch.arange(0, 64, 2, dtype=torch.float32)
        rotary_emb = 1 / (self.base ** (i/self.emb_dim))
        self.register_buffer("inv_freq", rotary_emb, persistent=False)

    def forward(self, position_num : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nvtx.range_push("RoPEfactorgenerate")
        rotary_emb = self.inv_freq[None,:,None].float()
        addr = position_num[:,None,:].to(dtype=torch.float32,device=position_num.device)
        addr_emb = (rotary_emb @ addr).transpose(1,2)
        emb = torch.cat((addr_emb,addr_emb), dim=-1)
        cos = emb.cos()[:,None].to(dtype=torch.float16)
        sin = emb.sin()[:,None].to(dtype=torch.float16)
        nvtx.range_pop()
        return cos, sin

#calculate q and k rotary embedding
# class QwenRotaryEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, q : torch.Tensor, k : torch.Tensor, sin : torch.Tensor, cos : torch.Tensor) \
#         -> tuple[torch.Tensor, torch.Tensor]:
#         xq1 = q[...,:q.size(-1)//2]
#         xq2 = q[...,q.size(-1)//2:]
#         x_q = torch.cat((-xq2, xq1), dim=-1)
#         xk1 = k[...,:k.size(-1)//2]
#         xk2 = k[...,k.size(-1)//2:]
#         x_k = torch.cat((-xk2, xk1), dim=-1)
#         q_emb = q * cos + x_q * sin
#         k_emb = k * cos + x_k * sin
#         return q_emb, k_emb

def QwenRotaryEmbedding(q : torch.Tensor, k : torch.Tensor, sin : torch.Tensor, cos : torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    xq1 = q[...,:q.size(-1)//2]
    xq2 = q[...,q.size(-1)//2:]
    x_q = torch.cat((-xq2, xq1), dim=-1)
    xk1 = k[...,:k.size(-1)//2]
    xk2 = k[...,k.size(-1)//2:]
    x_k = torch.cat((-xk2, xk1), dim=-1)
    q_emb = (q * cos + x_q * sin).to(q.dtype)
    k_emb = (k * cos + x_k * sin).to(k.dtype)
    return q_emb, k_emb

    


#class QwenRotaryEmbedding(nn.Module):
#    def __init__(self) -> None:
#        super().__init__()
#        self.embedding_num = 32
#        self.emb_dim = 64
#        self.base = 1000000
#        #self.rotary_emb = torch.Tensor(pow(self.base, i/64) for i in range(0,62,2))
#        i = torch.arange(0, 64, 2, dtype=torch.float32)
#        rotary_emb = (self.base ** (i * self.emb_dim ** -1)) ** -1
#        self.register_buffer("inv_freq", rotary_emb, persistent=False)
#    
#    def forward(self, q : torch.Tensor, k : torch.Tensor, position_num : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#        rotary_emb = self.inv_freq[None,:,None]             #(1,32,1)
#        addr = torch.arange(0, position_num, 1, dtype = torch.float32, device=q.device)[None, None, :] #(1,1,seq_len)
#        addr_emb = (rotary_emb @ addr).transpose(1, 2)      #(1,seq_len,32)
#        emb = torch.cat((addr_emb, addr_emb), dim=-1)         #(1,seq_len,64)
#        cos = emb.cos()[:,None].to(q.dtype)                             #(1,1,seq_len,64)
#        sin = emb.sin()[:,None].to(q.dtype)                             
#        xq1 = q[...,: q.size(-1)//2]                         #(batch_size, head_nums, seq_len, head_dim)
#        xq2 = q[..., q.size(-1)//2: ]                        #(n,14/2,seq_len,64)
#        xk1 = k[...,: k.size(-1)//2]                         
#        xk2 = k[..., k.size(-1)//2: ]                        
#        x_q = torch.cat((-xq2, xq1), dim=-1)
#        x_k = torch.cat((-xk2, xk1), dim=-1)
#        q_embed = q * cos + x_q * sin
#        k_embed = k * cos + x_k * sin
#        return q_embed, k_embed






# group query attention
class MyQwenAttention(nn.Module):
    def __init__(
            self,
            layer_idx : int = None
        ) -> None:
        super().__init__()
        self.q_head_num = 14
        self.kv_head_num = 2
        self.hidden_size = 896
        self.head_dim = 64
       # self.layer = layer_idx

        self.max_seq_len = 512
        self.max_batch_size = 1

        k_cache = torch.zeros(self.max_batch_size, self.kv_head_num, self.max_seq_len, self.head_dim, dtype=torch.float16)
        v_cache = torch.zeros(self.max_batch_size, self.kv_head_num, self.max_seq_len, self.head_dim, dtype=torch.float16)
        self.register_buffer("k_cache", k_cache, persistent=False)
        self.register_buffer("v_cache", v_cache, persistent=False)

        self.q_weight = nn.Linear(self.hidden_size, self.head_dim * self.q_head_num  , bias = True, dtype = torch.float16)
        self.k_weight = nn.Linear(self.hidden_size, self.head_dim * self.kv_head_num , bias = True, dtype = torch.float16)
        self.v_weight = nn.Linear(self.hidden_size, self.head_dim * self.kv_head_num , bias = True, dtype = torch.float16)       
        self.o_weight = nn.Linear(self.head_dim * self.q_head_num, self.hidden_size,  bias = False, dtype = torch.float16)


    def forward(self, 
                hidden_status : torch.Tensor,
                embedding_factor: tuple[torch.Tensor, torch.Tensor],
                attention_mask : torch.Tensor | None,
                cache_position : torch.LongTensor,
                current_seq_len : int,
                seq_len_t : torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        nvtx.range_push("Attention_generate")
        input_dim = hidden_status.shape[:-1]  

        divide_dim = (*input_dim, -1, self.head_dim)
        q_states = self.q_weight(hidden_status).view(divide_dim).transpose(1, 2)      #(batch_size, head_num, seq_len, head_dim)   
        k_states = self.k_weight(hidden_status).view(divide_dim).transpose(1, 2)      #(batch_size, head_num, seq_len, head_dim)   
        v_states = self.v_weight(hidden_status).view(divide_dim).transpose(1, 2)      #(batch_size, head_num, seq_len, head_dim)   
        cos, sin = embedding_factor
        q_after_emb, k_after_emb = QwenRotaryEmbedding(q_states, k_states, sin, cos)
        self.k_cache[:,:,cache_position,:] = k_after_emb
        self.v_cache[:,:,cache_position,:] = v_states
        #cache_position_len = cache_position[-1].item() + 1

        # k_calculate = self.k_cache[:,:,:current_seq_len,:]
        # v_calculate = self.v_cache[:,:,:current_seq_len,:]

        #k_extend = torch.cat((k_calculate[:,0:1].expand(-1,7,-1,-1), k_calculate[:,1].expand(-1,7,-1,-1)), dim=1)
        #v_extend = torch.cat((v_calculate[:,1:2].expand(-1,7,-1,-1), v_calculate[:,1].expand(-1,7,-1,-1)), dim=1)
        """
        k_extend = k_calculate[:,:,None,:,:]
        v_extend = v_calculate[:,:,None,:,:]
        q_extend = q_after_emb.view(q_after_emb.size(0), 2, 7, q_after_emb.size(2), self.head_dim)
        q_kT = (q_extend @ k_extend.transpose(-1,-2))/(self.head_dim ** 0.5)
        if attention_mask is not None:
            q_kT = q_kT + attention_mask
        attention = F.softmax(q_kT, dim = -1) @ v_extend   #(batch_size,head_nums,seq_len,head_dim)
        """
        # k_extend = k_calculate.repeat_interleave(repeats=7, dim=1)
        # v_extend = v_calculate.repeat_interleave(repeats=7, dim=1)
        """
        q_extend = q_after_emb.reshape([q_after_emb.size(0) * 2, 7, q_after_emb.size(-2), self.head_dim])
        k_extend = k_calculate.reshape([k_calculate.size(0) * 2, 1, -1, self.head_dim])
        v_extend = v_calculate.reshape([v_calculate.size(0) * 2, 1, -1, self.head_dim])
        attention = F.scaled_dot_product_attention(
            # query= q_after_emb,
            query=q_extend,
            key= k_extend,
            value= v_extend,
            # attn_mask= attention_mask,
            is_causal= True if q_extend.size(-2) == k_calculate.size(-2) else False
        
        )

        """
        if q_after_emb.size(2) > 1:
            k_calculate = self.k_cache[:,:,:current_seq_len,:]
            v_calculate = self.v_cache[:,:,:current_seq_len,:]
            k_extend = k_calculate.repeat_interleave(repeats=7, dim=1)
            v_extend = v_calculate.repeat_interleave(repeats=7, dim=1)
            attention = F.scaled_dot_product_attention(
                query= q_after_emb,
                key= k_extend,
                value= v_extend,
                # attn_mask= attention_mask,
                # is_causal= attention_mask is not None 
                is_causal= True 

            )
        else:
            attention = custom_attention_cuda.forward(self.k_cache, self.v_cache, q_after_emb.contiguous(), seq_len_t)



        out_data = self.o_weight(attention.view(q_after_emb.size(0), 14, q_after_emb.size(2), self.head_dim).transpose(1,2).contiguous().view(*input_dim,-1))
        nvtx.range_pop()
        return out_data,None
        


class myQwenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 896
        self.updim = 4864
        self.gate_proj = nn.Linear(self.hidden_size, self.updim, bias=False, 
                                    dtype= torch.float16)
        self.up_proj   = nn.Linear(self.hidden_size, self.updim, bias=False, 
                                    dtype= torch.float16)
        self.down_proj = nn.Linear(self.updim, self.hidden_size, bias=False, 
                                   dtype= torch.float16)
            
    def forward(self, x:torch.Tensor) -> torch.Tensor:
            nvtx.range_push("MLP")
            out_data = self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
            nvtx.range_pop()
            return out_data
            
        



class myQwenDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 896

        self.pre_Normal = RMSnormal(self.hidden_size)
        self.post_Normal = RMSnormal(self.hidden_size)
        #self.RoPE_gen = QwenRoPEfactorgenerate()
        #self.Rotary_emb = QwenRotaryEmbedding()
        self.attention = MyQwenAttention()
        self.mlplayer = myQwenMLP()
    
    def forward(self, x : torch.Tensor, 
                #position_num : torch.Tensor, 
                attention_mask : torch.Tensor | None,
                cache_position : torch.LongTensor,
                current_seq_len : int,
                cos : torch.Tensor,
                sin : torch.Tensor,
                seq_len_t: torch.Tensor
                ) -> torch.Tensor:
        #cos, sin = self.RoPE_gen(position_num)
        nvtx.range_push("DecoderLayer")
        original_x = x
        x = self.pre_Normal(x)
        attention, _ = self.attention(hidden_status = x,
                                        embedding_factor = (cos, sin),
                                        attention_mask = attention_mask,
                                        cache_position = cache_position,
                                        current_seq_len = current_seq_len,
                                        seq_len_t = seq_len_t
                                        )
        x = original_x + attention

        original_x = x
        x = self.post_Normal(x)
        x = self.mlplayer(x)
        out_data = x + original_x
        nvtx.range_pop()
        return out_data


class Qwen2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 151936
        self.hidden_dim = 896
        self.padding_idx = 151643

        self.emb_weight = nn.Embedding(self.vocab_size, self.hidden_dim, self.padding_idx, dtype= torch.float16)
        self.model_list = nn.ModuleList([myQwenDecoderLayer() for _ in range(24)])
        self.RoPE_factor_gen = QwenRoPEfactorgenerate()
        self.final_normal = RMSnormal(self.hidden_dim)
        self.register_buffer("seq_len_t", torch.zeros(1,dtype=torch.int32))

        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size,
                                dtype=torch.float16,
                                bias=False)

    def forward(self,
                x : torch.Tensor,
                position_num : torch.Tensor,
                attention_mask : torch.Tensor | None,
                cache_position : torch.LongTensor,
                current_seq_len: int | None
                )->torch.Tensor:
        nvtx.range_push("top_Model")
        # self.seq_len_t.fill_(current_seq_len)
        emb_data = self.emb_weight(x)
        cos, sin = self.RoPE_factor_gen(position_num)
        for decoder_layer in self.model_list[:24]:
            emb_data = decoder_layer(
                x = emb_data,
                cos = cos,
                sin = sin,
                attention_mask = attention_mask,
                cache_position = cache_position,
                current_seq_len = current_seq_len,
                seq_len_t = self.seq_len_t
            )
        final_out = self.lm_head(self.final_normal(emb_data)[:,-1:])
        nvtx.range_pop()
        return final_out





        









