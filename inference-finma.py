from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型名称
model_name = "models/finma"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择模型支持的最佳数据类型
    device_map="auto"    # 自动分配模型到设备（如 GPU/CPU）
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 对话提示（Prompt 和上下文）
prompt = "Explain the concept of reinforcement learning in simple terms."
messages = [
    {"role": "system", "content": "You are finma, an AI assistant created by Alibaba Cloud. You are helpful and friendly."},
    {"role": "user", "content": prompt}
]

# 格式化消息为聊天输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # 自动添加生成提示（如特殊标记）
)

# 将输入编码为模型可处理的格式
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 推理生成（生成最大 512 个新 token）
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# 提取生成结果（排除输入部分，只保留生成的新内容）
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 解码生成的响应
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 打印生成结果
print("Response:")
print(response)
