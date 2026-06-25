"""
使用PEFT (LoRA) 结合 TRL 库进行高效参数微调
"""

import os
from typing import Dict, List
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

# 配置环境变量
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/06_PEFT_DEMO"


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


# ==================== 2、构造LoRAConfig实例对象 ====================
lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)


# ==================== 3、加载基座模型与分词器 ====================
model = AutoModelForCausalLM.from_pretrained("model/qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B/")


# ==================== 4、通过get_peft_model获取peftmodel ====================
peft_model = get_peft_model(model, lora_config)


# ==================== 5、数据加载与预处理 ====================
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


# ==================== 6、构造SFTConfig实例 ====================
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
    output_dir="finetuned/06_PEFT_DEMO",
    bf16=True,
    max_length=710,
    assistant_only_loss=True,
    chat_template_path="./chat_template.jinja",
)


# ==================== 7、构造trainer并开始训练 ====================
trainer = SFTTrainer(
    args=training_args,
    model=peft_model,  # 💡 这里传入的是注入了 LoRA 的 peft_model
    train_dataset=mapped_keyword_data["train"],
    eval_dataset=mapped_keyword_data["test"],
    processing_class=tokenizer,
)

# 开始高效微调训练
trainer.train()

# 保存微调后的 LoRA 权重及配置文件
trainer.save_model("finetuned/06_PEFT_DEMO")