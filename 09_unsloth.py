from unsloth import FastLanguageModel
from datasets import load_dataset
from typing import Dict, List
import os
from trl.trainer.sft_config import SFTConfig
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/09_Unsloth_DEMO"
from trl.trainer.sft_trainer import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

# 1、加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./model/Qwen3-8B",
    load_in_4bit=True,          # 开启 4-bit 量化加载
    use_exact_model_name=True,  # 强制使用精准的模型名称/路径，不进行内部改写
    local_files_only=True       # 只允许从本地读取文件，断开网络连接
)

# 2. 冻结原始模型全部底层权重的前提下，线性层旁路中注入可训练的低秩矩阵
quantized_peft_model = FastLanguageModel.get_peft_model(
    model=model,
    target_modules=["q_proj", "v_proj"],# 线性层
    bias="none",        # 偏置项
    r=8,                # 低秩矩阵的秩
    lora_alpha=8,       # 放缩比例 此处为 8/8 = 1不放缩
    lora_dropout=0.05   # 随机失活 防止过拟合
)


# 3. 处理数据
# 3.1 读取本地数据
psychology_data = load_dataset("json", data_files={"train": r"./data/psychology_data.jsonl"})
# 单条数据:
"""
{
    "conversation_id": 1,
    "category": "dialogue",
    "conversation": [
        {
            "human": "不想回家 宁愿呆在吵闹的学校宿舍也不愿意回家 家庭经济变故三年了还处在悬崖边的境地 一回家面对的就是一张张哭丧幽怨的脸 每时每刻在耳边的只有钱字 富人爱命穷人爱钱? 我感受不到自己存在的价值和意义 我想逃离这个社会可是还是这个家让我逃不掉 我才16岁可是我感觉我活过的每一秒比800年还漫长 是不是只有在根本上也就是等经济状况变好了我才能变得正常。。。我该怎么办。。",
            "assistant": "你现在遇到的困难是很现实的，但是也不要放弃希望。你可以和家人沟通，了解他们的想法和感受，同时也让他们了解你的想法和需求。也可以找到一些适合自己的方式来放松自己的心情，比如去看电影、读书、运动等等。同时，也可以尝试去寻求帮助和支持，比如和老师、朋友、心理咨询师等等沟通，这样有助于缓解你的情绪和压力。记得不要放弃对未来的希望和信心，每个人都有自己的节奏和发展方式，相信自己会慢慢找到自己的出路。"
        }
    ],
    "dataset": "psychology"
}
"""
# 3.2 划分训练集和数据集
psychology_data = psychology_data["train"].train_test_split(test_size=0.1)


# 3.3 数据格式转化函数
def data_convert(examples: Dict[str, List]):
    """
    详情见 unsloth.txt
    """
    conversation_example_list = examples["conversation"]
    message_text_list = []
    for example in conversation_example_list:
        message_list = []
        conversation = example[0]
        message_list.append({"role": "user", "content": conversation["human"]})
        message_list.append({"role": "assistant", "content": conversation["assistant"]})

        message_text = tokenizer.apply_chat_template(
            message_list,                   # [{'role':'user','content':'内容'}, {'role':'assistant','content':'内容'}]
            tokenize=False,                 # 不需要分词器转为 Token IDs 后续使用unsloth 专用的 train_on_responses_only
            add_generation_prompt=False     # 训练时设置为否
        )
        message_text_list.append(message_text)

    return {"text": message_text_list}

# 3.4 调用函数批处理
mapped_psychology_data = psychology_data.map(
    data_convert,                   # 函数
    batched=True,                   # 批处理 默认1000
    remove_columns=psychology_data["train"].column_names    # 移除原始列名
)

"""
[
    {
        "text": "<|im_start|>user\n不想回家 宁愿呆在吵闹的学校宿舍...我该怎么办。。<|im_end|>\n<|im_start|>assistant\n你现在遇到的困难是很现实的...找到自己的出路。<|im_end|>\n"
    },
    {
        "text": "<|im_start|>user\n马上要期末考试了，我特别焦虑，静不下心复习怎么办？<|im_end|>\n<|im_start|>assistant\n考试焦虑是很常见的反应。尝试把复习任务拆解成小目标，每看到30分钟就休息5分钟...<|im_end|>\n"
    }
]
"""


# 4. 构造SFTConfig实例
training_args = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=1000,
    num_train_epochs=1,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=100,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    output_dir="finetuned/09_Unsloth_DEMO",
    bf16=True,
    max_length=710,
    # assistant_only_loss=True, # 原始用于 trl库中 只计算答案部分损失 unsloth 已重写 不用配置
    # chat_template_path="./chat_template.jinja"    # data_convert 函数里调用了 tokenizer.apply_chat_template，把数据转为纯文本（包含了 <|im_start|> 等标签）并存入了 "text" 字段
)

# 5. 构造trainer
trainer = SFTTrainer(
    args=training_args,
    model=quantized_peft_model,
    train_dataset=mapped_psychology_data["train"],
    eval_dataset=mapped_psychology_data["test"],
    processing_class=tokenizer
)


# 6. Unsloth 专有的 train_on_responses_only 补丁函数: 只计算assistant回复部分的损失
trainer = train_on_responses_only(
    trainer=trainer,
    instruction_part="<|im_start|>user\n",  # 用户指令（Prompt）的起始文本锚点  统一修改为 -100
    response_part="<|im_start|>assistant\n" # 模型回答（Response）的起始文本锚点
)

# 7. 训练
trainer.train()


# 8. 将训练好的 LoRA 适配器（Adapter）权重与大模型的原始基座（Base）权重进行物理合并（Merge），输出一个可以直接用于生产环境独立部署的完整模型文件
quantized_peft_model.save_pretrained_merged(
    "./finetuned/Qwen3-8B-SFT-unsloth-merged",  # 保存路径
    tokenizer,                                  # 分词器实例
    save_method="merged_16bit"                  # 权重的合并与精度的保存策略
)