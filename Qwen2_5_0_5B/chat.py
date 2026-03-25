import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型（替换成你的真实本地路径）
model_dir = r"E:/vscode/cuda_proj/SNN_proj1/Qwen2_5_0_5B/modeel_dir"  # <--- 注意改这里
print("正在把千问装进 3060 显存，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="cuda"
)
print("装载完毕！\n")

# 2. 初始化对话历史 (System Prompt 设定它的人设)
messages = [
    {"role": "system", "content": "你是一个资深的 AI Infra 工程师，说话简明扼要。"}
]

print("==================================================")
print("聊天终端已启动！(输入 'quit' 或 'exit' 结束对话，输入 'clear' 清空记忆)")
print("==================================================\n")

# 3. 开启无限聊天循环
while True:
    # 获取你在终端敲下的字
    user_input = input("你: ")
    
    # 设置退出和清空指令
    if user_input.lower() in ['quit', 'exit']:
        print("对话结束，拜拜！")
        break
    if user_input.lower() == 'clear':
        messages = [{"role": "system", "content": "你是一个资深的 AI Infra 工程师，说话简明扼要。"}]
        print("（记忆已清空）\n")
        continue
    if not user_input.strip():
        continue

    # 将你的新问题加入对话历史
    messages.append({"role": "user", "content": user_input})

    # 将完整的对话历史翻译成模型认识的格式并丢进 GPU
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    # 模型开始思考并生成回答
    outputs = model.generate(
        **model_inputs, 
        max_new_tokens=512,  # 允许它多说一点
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id # 消除一个小警告
    )

    # 截断并解码出最新生成的回答
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"千问: {response}\n")

    # 把千问的回答也加入对话历史，这样它就能记住上下文了！
    messages.append({"role": "assistant", "content": response})

# 退出循环后，打印架构数据供我们明天手写引擎参考
print("\n========== 核心架构参数 ==========")
config = model.config
print(f"Query 头数量 (num_attention_heads): {config.num_attention_heads}")
print(f"KV 头数量 (num_key_value_heads): {config.num_key_value_heads}")