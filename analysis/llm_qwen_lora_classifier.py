# analysis/llm_qwen_lora_classifier.py
from pathlib import Path

from analysis.llm_qwen_classifier import QwenPostClassifier
from model_Qwen3.lora import lora_config


class QwenLoraPostClassifier(QwenPostClassifier):
    """Load a base Qwen model plus a LoRA adapter for classification."""

    def __init__(self, app):
        super().__init__(app)
        self.base_model_dir = Path(lora_config.QWEN_LORA_INFERENCE_BASE_MODEL_DIR)
        self.adapter_dir = Path(lora_config.QWEN_LORA_ADAPTER_DIR)
        self.model_dir = self.base_model_dir
        self.backend_name = "lora"

    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return
        if not self.base_model_dir.exists():
            raise FileNotFoundError(f"LoRA基础模型目录不存在: {self.base_model_dir}")
        if not self.adapter_dir.exists():
            raise FileNotFoundError(f"LoRA适配器目录不存在: {self.adapter_dir}")

        print(f"Qwen LoRA分类：正在加载基础模型 {self.base_model_dir} ...")
        print(f"Qwen LoRA分类：正在加载适配器 {self.adapter_dir} ...")

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_dir,
            device_map="auto",
            torch_dtype="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_dir,
            local_files_only=True,
        )
        self.model.eval()
        print(f"Qwen LoRA分类：模型加载完成，CUDA可用={torch.cuda.is_available()}。")
