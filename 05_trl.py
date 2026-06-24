"""
使用trl库进行SFT全参微调
"""

import os
from typing import Dict, List
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

# 配置环境变量
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/05_TRL_DEMO"


# ==================== 1、数据格式转化函数 ====================
def data_convert(examples: Dict[str, List]):
    """
    将原始数据转换成带messages的标准对话格式
    """
    conversation_example_list = examples["conversation"]
    examples_message_list = []
    for example in conversation_example_list:
        message_list = []
        conversation = example[0]
        message_list.append({"role": "user", "content": conversation["human"]})
        message_list.append(
            {"role": "assistant", "content": conversation["assistant"]}
        )
        examples_message_list.append(message_list)

    return {"messages": examples_message_list}


# ==================== 2、数据加载与预处理 ====================
# 加载原始数据
keyword_data = load_dataset(
    "json",
    data_files={
        "train": r"./data/keywords_data_train.jsonl",
        "test": r"./data/keywords_data_test.jsonl",
    },
)

# 转换数据格式并移除原始列
mapped_keyword_data = keyword_data.map(
    data_convert,
    batched=True,
    remove_columns=keyword_data["train"].column_names,
)


# ==================== 3、加载模型与分词器 ====================
model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B/")


# ==================== 4、构造SFTConfig实例 ====================
training_args = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=1000,
    num_train_epochs=1,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    learning_rate=3e-5,
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
    output_dir="finetuned/05_TRL_DEMO",
    bf16=True,
    max_length=710,
    assistant_only_loss=True,
    chat_template_path="./chat_template.jinja",
)


# ==================== 5、构造SFTTrainer并开始训练 ====================
trainer = SFTTrainer(
    args=training_args,
    model=model,
    train_dataset=mapped_keyword_data["train"],
    eval_dataset=mapped_keyword_data["test"],
    processing_class=tokenizer,
)

# 开始微调训练
trainer.train()

# 保存微调后的模型
trainer.save_model("finetuned/05_TRL_DEMO")