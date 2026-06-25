"""
将基座模型和适配器进行合并并保存
"""

import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==================== 1、接收终端命令行参数 ====================
parser = argparse.ArgumentParser(description="Merge Lora Model")
parser.add_argument("--base_model", type=str, default="./model/qwen3-0.6B")  # 传入：基座模型路径
parser.add_argument("--peft_model", type=str, default="./finetuned/06_PEFT_DEMO")  # 传入：训练好的 LoRA 文件夹路径
parser.add_argument("--merge_model_path", type=str,default="./models_merged/qwen3_keyword_model",)  # 传入：合并后的新模型保存路径
args = parser.parse_args()


# ==================== 2、加载基座模型与分词器 ====================
# 加载未微调的原始大模型
base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
# 从 LoRA 文件夹加载分词器（保证特殊 Token 词表是最新的）
tokenizer = AutoTokenizer.from_pretrained(args.peft_model)


# ==================== 3、将 LoRA 挂载到基座模型 ====================
peft_model = PeftModel.from_pretrained(base_model, model_id=args.peft_model)


# ==================== 4、核心步骤：矩阵合并与外壳卸载 ====================
# 1. 把 LoRA 旁路的参数直接加进基座模型的参数矩阵里，融为一体
# 2. 把已经没用的 LoRA 旁路外壳结构从内存里移除
# 最终返回一个内部吸纳了微调知识、但结构与普通原生模型完全一致的 merged_model
merged_model = peft_model.merge_and_unload()


# ==================== 5、保存合并后的完整大模型 ====================
# 保存融为一体后的新完整模型权重
merged_model.save_pretrained(args.merge_model_path)
# 并在同目录下保存配套的分词器文件，方便后续直接用于部署推理
tokenizer.save_pretrained(args.merge_model_path)

# ==================== 6. 建议命令行启动文件  ====================
# python merge_model.py
# --base_model ./model/qwen3-0.6B
# --peft_model ./finetuned/06_PEFT_DEMO
# --merge_model_path ./models_merged/qwen3_keyword_model
# 分别指定各自路径