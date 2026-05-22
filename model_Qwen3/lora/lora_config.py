from pathlib import Path


# LoRA 目录根路径
LORA_DIR = Path(__file__).resolve().parent

# 项目根目录
BASE_DIR = LORA_DIR.parents[1]


# -----------------------------
# LoRA / QLoRA 路径配置
# -----------------------------

# LoRA 训练时使用的基础模型目录
# RTX 3050 Ti 4GB 显存下优先使用 0.5B 模型，避免 4B 训练显存不足
QWEN_LORA_TRAIN_BASE_MODEL_DIR = BASE_DIR / "model_Qwen3" / "Qwen2.5-0.5B-Instruct"

# LoRA 推理时使用的基础模型目录
# 必须与训练 adapter 时的基础模型保持一致
QWEN_LORA_INFERENCE_BASE_MODEL_DIR = BASE_DIR / "model_Qwen3" / "Qwen2.5-0.5B-Instruct"

# LoRA 训练数据目录
QWEN_LORA_DATA_DIR = LORA_DIR / "data"

# LoRA 训练集路径
QWEN_LORA_TRAIN_FILE = QWEN_LORA_DATA_DIR / "train.jsonl"

# LoRA 验证集路径
QWEN_LORA_VALID_FILE = QWEN_LORA_DATA_DIR / "valid.jsonl"

# LoRA 适配器输出目录
QWEN_LORA_ADAPTER_DIR = LORA_DIR / "output" / "campus_classifier_real_posts"


# -----------------------------
# QLoRA 默认训练超参数
# -----------------------------

# 单条样本最大 token 长度
# 分类 prompt 本身包含完整 LABEL_SCHEMA，短帖样本也会超过 512 token。
# 0.5B 基座在 RTX 3050 Ti 4GB 上用 batch size 1 训练时，768 是更稳妥的默认值。
TRAIN_MAX_LENGTH = 768

# 训练轮数
TRAIN_EPOCHS = 3

# 学习率
TRAIN_LEARNING_RATE = 2e-4

# 单卡训练 batch size
TRAIN_BATCH_SIZE = 1

# 单卡验证 batch size
EVAL_BATCH_SIZE = 1

# 梯度累积步数
GRADIENT_ACCUMULATION_STEPS = 8

# 预热比例
WARMUP_RATIO = 0.03

# 权重衰减
WEIGHT_DECAY = 0.01

# 日志间隔
LOGGING_STEPS = 5

# 保存策略，可选 epoch / steps
SAVE_STRATEGY = "epoch"

# steps 模式下的保存间隔
SAVE_STEPS = 50

# 验证策略，可选 epoch / steps
EVAL_STRATEGY = "epoch"

# steps 模式下的验证间隔
EVAL_STEPS = 50

# LoRA rank
LORA_R = 16

# LoRA alpha
LORA_ALPHA = 32

# LoRA dropout
LORA_DROPOUT = 0.05
