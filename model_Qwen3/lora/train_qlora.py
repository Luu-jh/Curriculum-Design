r"""
Train a QLoRA adapter for the campus-wall classification task.

Features:
1. Reads comment-friendly JSONL files. Lines starting with "#" are ignored.
2. Uses the local Qwen model only; no online download is required.
3. Saves only the LoRA adapter, which can later be loaded by the LoRA classifier.

Typical usage from project root:
    .\.venv\Scripts\python.exe model_Qwen3\lora\train_qlora.py

If needed, install dependencies first:
    .\.venv\Scripts\python.exe -m pip install -r model_Qwen3\lora\requirements-qwen-lora.txt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lora_config
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = Path(lora_config.QWEN_LORA_TRAIN_BASE_MODEL_DIR)
DEFAULT_TRAIN_FILE = Path(lora_config.QWEN_LORA_TRAIN_FILE)
DEFAULT_VALID_FILE = Path(lora_config.QWEN_LORA_VALID_FILE)
DEFAULT_OUTPUT_DIR = Path(lora_config.QWEN_LORA_ADAPTER_DIR)


LABEL_SCHEMA = {
    "求助": ["代办", "资源获取", "问题咨询"],
    "交易": ["出售", "求购", "转让"],
    "找人找物": ["找人", "找物", "拿错归还"],
    "吐槽": ["吐槽环境", "吐槽生活"],
    "日常": ["生活分享", "交友扩列", "日常闲聊"],
    "其他": ["其他"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a campus-wall QLoRA adapter.")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL, help="Base Qwen model directory.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE, help="Training data file.")
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID_FILE, help="Validation data file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="LoRA adapter output directory.")
    parser.add_argument("--max-length", type=int, default=lora_config.TRAIN_MAX_LENGTH, help="Maximum token length.")
    parser.add_argument("--epochs", type=int, default=lora_config.TRAIN_EPOCHS, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=lora_config.TRAIN_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=lora_config.TRAIN_BATCH_SIZE, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=lora_config.EVAL_BATCH_SIZE, help="Per-device eval batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=lora_config.GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=lora_config.WARMUP_RATIO, help="Warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=lora_config.WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--logging-steps", type=int, default=lora_config.LOGGING_STEPS, help="Logging interval.")
    parser.add_argument("--save-strategy", choices=("epoch", "steps"), default=lora_config.SAVE_STRATEGY, help="Save strategy.")
    parser.add_argument("--save-steps", type=int, default=lora_config.SAVE_STEPS, help="Save interval when save-strategy=steps.")
    parser.add_argument("--eval-strategy", choices=("epoch", "steps"), default=lora_config.EVAL_STRATEGY, help="Eval strategy.")
    parser.add_argument("--eval-steps", type=int, default=lora_config.EVAL_STEPS, help="Eval interval when eval-strategy=steps.")
    parser.add_argument("--lora-r", type=int, default=lora_config.LORA_R, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=lora_config.LORA_ALPHA, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=lora_config.LORA_DROPOUT, help="LoRA dropout.")
    return parser.parse_args()


def import_dependencies():
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}\n"
            "Please install dependencies first:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install -r model_Qwen3\\lora\\requirements-qwen-lora.txt"
        ) from exc

    return (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        Trainer,
        TrainingArguments,
    )


def load_jsonl_with_comments(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    samples = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc
        validate_sample(item, path, line_no)
        samples.append(item)

    if not samples:
        raise ValueError(f"{path} 中没有可用训练样本。")
    return samples


def validate_sample(sample: dict, path: Path, line_no: int):
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{path}:{line_no} 缺少合法的 messages 列表。")

    roles = [message.get("role") for message in messages]
    if "assistant" not in roles:
        raise ValueError(f"{path}:{line_no} 必须包含 assistant 消息。")

    assistant_message = messages[-1]
    if assistant_message.get("role") != "assistant":
        raise ValueError(f"{path}:{line_no} 最后一条消息必须是 assistant。")

    try:
        answer = json.loads(assistant_message.get("content", ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no} assistant 内容必须是严格 JSON。") from exc

    category_1 = answer.get("llm_category_1")
    category_2 = answer.get("llm_category_2")
    if category_1 not in LABEL_SCHEMA:
        raise ValueError(f"{path}:{line_no} 一级分类不在标签体系中: {category_1}")
    if category_2 not in LABEL_SCHEMA[category_1]:
        raise ValueError(f"{path}:{line_no} 二级分类不在一级分类允许范围内: {category_2}")


@dataclass
class SampleEncoding:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class CommentJsonlSFTDataset:
    def __init__(self, samples, tokenizer, max_length: int):
        self.features = [build_feature(sample, tokenizer, max_length) for sample in samples]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def build_feature(sample: dict, tokenizer, max_length: int):
    messages = sample["messages"]
    prompt_messages = messages[:-1]
    assistant_message = messages[-1]

    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    answer_text = assistant_message["content"]
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + answer_text + eos

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("样本被截断后 assistant 内容全部丢失，请增大 --max-length 或缩短输入。")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_model_and_tokenizer(args, deps):
    (
        torch,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        _Trainer,
        _TrainingArguments,
    ) = deps

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    args = parse_args()
    args.base_model = args.base_model.resolve()
    args.train_file = args.train_file.resolve()
    args.valid_file = args.valid_file.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.base_model.exists():
        raise SystemExit(
            f"基础模型目录不存在: {args.base_model}\n"
            "说明：QLoRA 训练通常需要保留原始 Hugging Face 模型目录。"
        )

    deps = import_dependencies()
    (
        torch,
        _AutoModelForCausalLM,
        _AutoTokenizer,
        _BitsAndBytesConfig,
        _LoraConfig,
        _get_peft_model,
        _prepare_model_for_kbit_training,
        Trainer,
        TrainingArguments,
    ) = deps

    print(f"Base model : {args.base_model}")
    print(f"Train file : {args.train_file}")
    print(f"Valid file : {args.valid_file}")
    print(f"Output dir : {args.output_dir}")
    print(f"CUDA ready : {torch.cuda.is_available()}")

    train_samples = load_jsonl_with_comments(args.train_file)
    valid_samples = load_jsonl_with_comments(args.valid_file)
    print(f"Train samples: {len(train_samples)}")
    print(f"Valid samples: {len(valid_samples)}")
    if len(train_samples) < 20:
        print("Warning: 训练集少于 20 条，只适合验证流程，不足以得到可用分类器。")
    if len(valid_samples) < 10:
        print("Warning: 验证集少于 10 条，验证结果波动会比较大。")

    model, tokenizer = build_model_and_tokenizer(args, deps)
    train_dataset = CommentJsonlSFTDataset(train_samples, tokenizer, args.max_length)
    valid_dataset = CommentJsonlSFTDataset(valid_samples, tokenizer, args.max_length)
    data_collator = DataCollatorForCausalLM(tokenizer)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": str(args.base_model),
        "train_file": str(args.train_file),
        "valid_file": str(args.valid_file),
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    (args.output_dir / "training_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Training finished. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
