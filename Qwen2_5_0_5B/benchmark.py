import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

# 1. 定位本地模型路径 (如果本地已有，瞬间返回路径，不会消耗流量)
#model_id = "E:/vscode/cuda_proj/SNN_proj1/Qwen2_5_0_5B/modeel_dir"
print(f"========== 1. 寻找本地模型 ==========")
model_dir = r"E:/vscode/cuda_proj/SNN_proj1/Qwen2_5_0_5B/modeel_dir"
print(f"成功找到本地模型路径: {model_dir}\n")

# 2. 加载“翻译字典”和“灵魂肉体”
print("========== 2. 开始把模型搬进 3060 的显存 ==========")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16, # 半精度，省显存
    device_map="cuda"          # 直接塞进 GPU
)
print("搬运完成！\n")

# 3. 跑通基线：测试生成能力
print("========== 3. 验证模型智商 ==========")
# Instruct 版本必须用标准对话格式封装
messages = [
    {"role": "system", "content": "你是一个资深的 AI Infra 工程师。"},
    {"role": "user", "content": "在 6GB 显存的显卡上做大模型推理，最需要注意什么？请用一句话回答。"}
]
# 将对话转换成模型认识的格式
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

print("正在努力生成中...\n")
outputs = model.generate(
    **model_inputs, 
    max_new_tokens=50, # 限制字数，快点出结果
    temperature=0.7
)

# 截取新生成的部分并解码成人类文字
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"千问的回答: \n{response}\n")

# 4. 透视架构：寻找靶子
print("========== 4. 透视底层骨架 ==========")
print(model)

config = model.config
print("\n========== 核心架构参数 (请死死记住这几个数字) ==========")
print(f"隐藏层总维度 (hidden_size): {config.hidden_size}")
print(f"Query 头数量 (num_attention_heads): {config.num_attention_heads}")
print(f"KV 头数量 (num_key_value_heads): {config.num_key_value_heads}")