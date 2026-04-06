import time
import torch
from torch.autograd.profiler import emit_nvtx
from torch.profiler import record_function
from my_qwen2 import Qwen2Model
from transformers import AutoTokenizer
from safetensors.torch import load_file

# 👑 零拷贝极速权重注入 (专为你手搓的空壳机甲准备)
def load_weights_from_safetensors(my_model, model_dir):
    safetensors_path = f"{model_dir}\\model.safetensors"
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
        my_state_dict[f"{my_prefix}attention.q_weight.weight"] = hf_state_dict[f"{hf_prefix}self_attn.q_proj.weight"]
        my_state_dict[f"{my_prefix}attention.q_weight.bias"] = hf_state_dict[f"{hf_prefix}self_attn.q_proj.bias"]
        my_state_dict[f"{my_prefix}attention.k_weight.weight"] = hf_state_dict[f"{hf_prefix}self_attn.k_proj.weight"]
        my_state_dict[f"{my_prefix}attention.k_weight.bias"] = hf_state_dict[f"{hf_prefix}self_attn.k_proj.bias"]
        my_state_dict[f"{my_prefix}attention.v_weight.weight"] = hf_state_dict[f"{hf_prefix}self_attn.v_proj.weight"]
        my_state_dict[f"{my_prefix}attention.v_weight.bias"] = hf_state_dict[f"{hf_prefix}self_attn.v_proj.bias"]
        my_state_dict[f"{my_prefix}attention.o_weight.weight"] = hf_state_dict[f"{hf_prefix}self_attn.o_proj.weight"]
        my_state_dict[f"{my_prefix}mlplayer.gate_proj.weight"] = hf_state_dict[f"{hf_prefix}mlp.gate_proj.weight"]
        my_state_dict[f"{my_prefix}mlplayer.up_proj.weight"] = hf_state_dict[f"{hf_prefix}mlp.up_proj.weight"]
        my_state_dict[f"{my_prefix}mlplayer.down_proj.weight"] = hf_state_dict[f"{hf_prefix}mlp.down_proj.weight"]
        
    my_model.load_state_dict(my_state_dict, strict=False)

# 👑 带有自动打标核武器的推理解码器
def benchmark_generate(module, input_data, max_seq_len, phase_name, graph_use = False):
    device = input_data.device
    generate_data = input_data
    past_len = 0
    static_input = torch.zeros([1, 1], device=device, dtype= torch.long)
    static_pos_num = torch.zeros([1, 1], device=device, dtype= torch.long)
    static_cache_pos = torch.zeros([1], device=device, dtype= torch.long)

    
    with torch.no_grad():
        for step in range(max_seq_len):
            step_type = "Prefill" if step == 0 else "Decode"
            
            # 👑 record_function 替代了之前没用的 nvtx.range_push
            with record_function(f"{phase_name}_Step_{step}_{step_type}"):
                input_len = input_data.size(1)
                cache_position = torch.arange(past_len, past_len + input_len, step=1, dtype=torch.long, device=device)
                position_sum = cache_position[None,:]

                attention_mask = None
                if input_len > 1:
                    mask = torch.tril(torch.ones(input_len, input_len, device=device, dtype=torch.float16))
                    attention_mask = torch.zeros(input_len, input_len, device=device, dtype=torch.float16).masked_fill(mask == 0, float('-inf'))
                
                
                
                if step == 2 and graph_use:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                            data_out = module(
                                x = static_input,
                                position_num = static_pos_num,
                                attention_mask = None,
                                cache_position = static_cache_pos,
                                current_seq_len = None
                            )
                if step < 2 or not graph_use:
                    module.seq_len_t.fill_(past_len + input_len) 
                    data_out = module(
                        x = input_data,
                        position_num = position_sum,
                        attention_mask = attention_mask,
                        cache_position = cache_position,
                        current_seq_len = past_len + input_len
                    )
                else:
                    static_input.copy_(input_data)
                    static_pos_num.copy_(position_sum)
                    static_cache_pos.fill_(past_len)
                    module.seq_len_t.fill_(past_len + 1)
                    graph.replay()
                
                
                logits = data_out[:,-1,:]
                final_data = torch.argmax(logits, dim=-1, keepdim=True)
                
                generate_data = torch.cat((generate_data, final_data), dim=1)
                # if final_data[0, 0].item() == 151643:
                if (final_data == 151643).any():
                    break

                past_len += input_len
                input_data = final_data
            
    return generate_data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model_path = r"E:\vscode\cuda_proj\SNN_proj1\Qwen2_5_0_5B\modeel_dir"
    
    print("准备进入测速跑道...")
    # 注意：这里实例化的是你亲手写的 my_qwen2.py 里的 Qwen2Model！
    model = Qwen2Model().to(device)
    model.eval()
    # model = torch.compile(model)
    load_weights_from_safetensors(model, local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    
    prompt = "请详细分析一下人工智能在未来十年的发展趋势，并给出三个具体的应用场景。" * 15
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    # ==========================================
    # 阶段 1：预热
    # ==========================================
    print("阶段 1：预热 GPU...")
    with record_function("Warmup_Phase"):
        _ = benchmark_generate(model, input_ids, max_seq_len=10, phase_name="Warmup", graph_use = False)
        torch.cuda.synchronize() 
    print("预热完毕！")
    
    # ==========================================
    # 阶段 2：吞吐量测速 (Tokens/s)
    # ==========================================
    print("阶段 2：吞吐量压测 (只算纯解码耗时)...")
    TEST_TOKENS = 50
    
    with record_function("Throughput_Phase"):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        _ = benchmark_generate(model, input_ids, max_seq_len=TEST_TOKENS, phase_name="Throughput", graph_use = True)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    time_taken = end_time - start_time
    tps = TEST_TOKENS / time_taken
    print(f"吞吐量基线: {tps:.2f} tokens/s (耗时: {time_taken:.4f}s)")

    # ==========================================
    # 阶段 3：给 nsys 留的抓取窗口
    # ==========================================
    print("阶段 3：执行供 nsys 抓取的微观循环...")
    
    # 👑 启动 PyTorch 自动 NVTX 铺路机，强制暴露所有底层算子！
    # with emit_nvtx():
    #     with record_function("Profiler_Target_Region"):
    #         _ = benchmark_generate(model, input_ids, max_seq_len=5, phase_name="Profile", graph_use = True)
    #         torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()

    with emit_nvtx():
        with record_function("Profile_CUDAGraph"):
            _ = benchmark_generate(model, input_ids, max_seq_len=5,
                                   phase_name="Profile", graph_use=True)
            torch.cuda.synchronize()
    
    torch.cuda.cudart().cudaProfilerStop()
    print("测试结束！")