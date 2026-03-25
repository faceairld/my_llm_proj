import torch
import torch.nn.functional as F
from my_qwen2 import Qwen2Model 
from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM
import os
from safetensors.torch import load_file



def generate(module : Qwen2Model,
             input_data : torch.LongTensor,
             max_seq_len : int = 40,
             temperature : float = 0.7
             )->torch.LongTensor:
    
    device = input_data.device
    input_len = input_data.size(1)

    mask = torch.tril(torch.ones(input_len, input_len, device = device, dtype = torch.float16))
    attention_mask = torch.zeros(input_len, input_len, device = device, dtype = torch.float16).masked_fill(mask== 0, float('-inf'))

    generate_data = input_data
    past_len = 0
    
    with torch.no_grad():
        for _ in range(max_seq_len):
            input_len = input_data.size(1)
            cache_position = torch.arange(past_len, past_len + input_len, step=1, dtype=torch.long, device=device)
            position_sum = cache_position[None,:]

            data_out = module(
                        x = input_data,
                        position_num = position_sum,
                        attention_mask = attention_mask,
                        cache_position = cache_position,
                        current_seq_len = past_len + input_len,
                        )
            logits = data_out[:,-1,:]
            if temperature > 0:
                final_data = torch.multinomial(F.softmax((logits / temperature), dim= -1), num_samples= 1)
            else:
                final_data = torch.argmax(logits, dim= -1, keepdim= True)
            generate_data = torch.cat((generate_data, final_data), dim= 1)
            if final_data[0, 0].item() == 151643:
                break

            past_len += input_len
            input_data = final_data
            attention_mask = None
    
    return generate_data

def load_weights_from_safetensors(my_model, model_dir):
    print("loading weight...")
    
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    hf_state_dict = load_file(safetensors_path)
    my_state_dict = my_model.state_dict()
    
    my_state_dict["emb_weight.weight"] = hf_state_dict["model.embed_tokens.weight"]
    my_state_dict["final_normal.weight"] = hf_state_dict["model.norm.weight"]
    
    if "lm_head.weight" in hf_state_dict:
        my_state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"]
    else:
        my_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]
        
    for i in range(24):
        hf_prefix = f"model.layers.{i}."
        my_prefix = f"model_list.{i}."
        
        my_state_dict[f"{my_prefix}pre_Normal.weight"] = hf_state_dict[f"{hf_prefix}input_layernorm.weight"]
        my_state_dict[f"{my_prefix}post_Normal.weight"] = hf_state_dict[f"{hf_prefix}post_attention_layernorm.weight"]
        
        # Attention
        my_state_dict[f"{my_prefix}attention.q_weight.weight"] = hf_state_dict[f"{hf_prefix}self_attn.q_proj.weight"]
        my_state_dict[f"{my_prefix}attention.q_weight.bias"] = hf_state_dict[f"{hf_prefix}self_attn.q_proj.bias"]
        my_state_dict[f"{my_prefix}attention.k_weight.weight"] = hf_state_dict[f"{hf_prefix}self_attn.k_proj.weight"]
        my_state_dict[f"{my_prefix}attention.k_weight.bias"] = hf_state_dict[f"{hf_prefix}self_attn.k_proj.bias"]
        my_state_dict[f"{my_prefix}attention.v_weight.weight"] = hf_state_dict[f"{hf_prefix}self_attn.v_proj.weight"]
        my_state_dict[f"{my_prefix}attention.v_weight.bias"] = hf_state_dict[f"{hf_prefix}self_attn.v_proj.bias"]
        my_state_dict[f"{my_prefix}attention.o_weight.weight"] = hf_state_dict[f"{hf_prefix}self_attn.o_proj.weight"]
        
        # MLP
        my_state_dict[f"{my_prefix}mlplayer.gate_proj.weight"] = hf_state_dict[f"{hf_prefix}mlp.gate_proj.weight"]
        my_state_dict[f"{my_prefix}mlplayer.up_proj.weight"] = hf_state_dict[f"{hf_prefix}mlp.up_proj.weight"]
        my_state_dict[f"{my_prefix}mlplayer.down_proj.weight"] = hf_state_dict[f"{hf_prefix}mlp.down_proj.weight"]
        
    my_model.load_state_dict(my_state_dict, strict=False)
    print("loading finish")
        

if __name__ == "__main__":
    print("start MyQwen Engine...")
    local_model_path = r"E:\vscode\cuda_proj\SNN_proj1\Qwen2_5_0_5B\modeel_dir"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2Model().to(device)
    model.eval()
    load_weights_from_safetensors(model, local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    while True:
        user_input = input("\nuser:")
        if user_input.lower() in ['quit', 'exit']:
            print("stop model")
            break
        if not user_input.strip():
            continue
        #inputs = tokenizer(user_input, return_tensors="pt")
        #input_ids = inputs.input_ids.to(device)
        messages = [
            {"role": "user", "content": user_input}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)

        print("think...", end="", flush=True)
        output_ids = generate(module=model, input_data=input_ids, max_seq_len=100)
        input_len = input_ids.shape[1]
        new_token_ids = output_ids[0][input_len:]
        response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        print(f"\r MyQwen: {response}")