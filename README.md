# Large Model Architecture and Fine-tuning

学习 **Qwen3** 模型架构, 以**Qwen3** 为基座学习大语言模型**微调技术**，主要包括从现代 Transformer 改进组件、SFT/DPO 训练流程，到 TRL/PEFT/QLoRA/Unsloth 等工业级微调工具链的全流程实践。

---

## 项目结构与文件说明

| 文件 | 说明 |
|------|------|
| `01_Qwen3.py` | 学习现代大模型架构改进：GQA、RoPE、RMSNorm、SwiGLU、KV Cache |
| `02_sft.py` | 监督微调（SFT）手写训练循环 —— 基座模型：Qwen3-0.6B-Base |
| `03_dpo.py` | 偏好对齐（DPO）手写训练循环 —— 基座模型：SFT 后模型 |
| `04_model_generate.py` | 模型推理脚本，支持思考内容分离与命令行参数 |
| `05_trl.py` | TRL 全参微调 —— 基座模型：Qwen3-0.6B |
| `06_peft.py` | PEFT (LoRA) 高效微调 —— 基座模型：Qwen3-0.6B |
| `07_model_merge.py` | LoRA 适配器权重合并到基座模型 |
| `08_QLoRA.py` | 4-bit 量化 LoRA 微调 —— 基座模型：Qwen3-8B |
| `09_unsloth.py` | Unsloth 加速 QLoRA 微调 —— 基座模型：Qwen3-8B |

---

## 环境依赖

- Python 3.10+
- PyTorch 2.x
- Transformers
- Datasets
- TRL
- PEFT
- Unsloth
- TensorBoard
- tqdm

完整依赖见 [`requirements.txt`](./requirements.txt)

---

## 快速开始

### 1. 学习现代大模型架构

```bash
python 01_Qwen3.py
```

参照 Qwen3-0.6B 配置，实现现代大模型核心组件，用于对比学习传统 Transformer 的改进：RMSNorm、RoPE、GQA、SwiGLU、KV Cache。**仅用于理解架构，未加载预训练权重。**

### 2. 模型推理

```bash
# 基础用法
python 04_model_generate.py --model_name model/Qwen3-0.6B-Base --prompt "你好"
```

### 3. 微调实践

```bash
# 手写 SFT 微调
python 02_sft.py

# TRL 全参微调
python 05_trl.py

# LoRA 高效微调
python 06_peft.py

# QLoRA 量化微调（需要 Qwen3-8B）
python 08_QLoRA.py

# Unsloth 加速微调（需要 Qwen3-8B）
python 09_unsloth.py
```

### 4. DPO 偏好对齐

```bash
python 03_dpo.py
```

### 5. LoRA 模型合并

```bash
python 07_model_merge.py --base_model ./model/qwen3-0.6B --peft_model ./finetuned/06_PEFT_DEMO --merge_model_path ./models_merged/qwen3_keyword_model
```

---

## 使用的基座模型

| 模型 | 参数量 | 用途 |
|------|--------|------|
| Qwen3-0.6B-Base | 0.6B | SFT、DPO、TRL 全参微调 |
| Qwen3-0.6B | 0.6B | TRL、PEFT LoRA 微调 |
| Qwen3-8B | 8B | QLoRA、Unsloth 微调 |

---

## 从零学习的架构组件

`01_Qwen3.py` 中学习实现的现代大模型改进：

- **RMSNorm**：相比 LayerNorm 去除了均值中心化，计算更高效
- **RoPE**：旋转位置编码，通过旋转变换注入位置信息
- **GQA**：分组查询注意力，多个 Query 头共享一组 KV 头
- **SwiGLU**：使用 SiLU 激活的门控线性单元
- **KV Cache**：prefill/decode 两阶段推理，缓存历史 Key/Value

---

## 训练策略

| 阶段 | 学习率 | 调度器 | 预热比例 |
|------|--------|--------|----------|
| SFT | 2e-5 | 余弦衰减 | 10% |
| DPO | 1e-6 | 余弦衰减 | 10% |
| TRL SFT | 3e-5 | 余弦衰减 | 10% |
| QLoRA | 3e-4 | 余弦衰减 | 10% |

所有训练均仅在 **assistant 回答部分** 计算损失。

---

## 日志与监控

```bash
tensorboard --logdir logs/
```

| 日志目录 | 对应脚本 |
|----------|----------|
| `logs/02_SFT_TRAIN` | 手写 SFT |
| `logs/03_DPO_TRAIN` | 手写 DPO |
| `logs/05_TRL_DEMO` | TRL 全参微调 |
| `logs/06_PEFT_DEMO` | PEFT LoRA |
| `logs/08_QLoRA_DEMO` | QLoRA |
| `logs/09_Unsloth_DEMO` | Unsloth |

---

## 模型输出目录

| 目录 | 内容 |
|------|------|
| `finetuned/02_SFT_TRAIN/` | 手写 SFT 微调模型 |
| `finetuned/03_DPO_TRAIN/` | DPO 对齐后模型 |
| `finetuned/05_TRL_DEMO/` | TRL 全参微调模型 |
| `finetuned/06_PEFT_DEMO/` | LoRA 适配器权重 |
| `finetuned/08_QLoRA_DEMO/` | QLoRA 适配器权重 |
| `finetuned/09_Unsloth_DEMO/` | Unsloth 适配器权重 |
| `finetuned/Qwen3-8B-SFT-unsloth-merged/` | Unsloth 合并后完整模型 |

---

## 数据集说明

| 数据集 | 用途 | 路径 |
|--------|------|------|
| UltraChat 200k | SFT 训练 | `data/ultrachat_200k/` |
| UltraFeedback Binarized | DPO 训练 | `data/ultrafeedback_binarized/` |
| 自定义关键词数据 | TRL/PEFT 微调 | `data/keywords_data_train.jsonl` |
| 心理咨询数据 | QLoRA/Unsloth 微调 | `data/psychology_data.jsonl` |

---

## 许可

仅用于学习目的
```