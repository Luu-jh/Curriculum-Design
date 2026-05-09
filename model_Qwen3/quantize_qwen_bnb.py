"""
Quantize the local Qwen3 model with bitsandbytes 4-bit or 8-bit loading.

Default input:
    model_Qwen3/Qwen3-4B-Instruct-2507
Default output:
    model_Qwen3/Qwen3-4B-Instruct-2507-bnb-4bit

This script is written for local/offline use. It does not download a model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = BASE_DIR / "Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = BASE_DIR / "Qwen3-4B-Instruct-2507-bnb-4bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize local Qwen3 model with bitsandbytes.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Path to the original Hugging Face model directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Path where the quantized model will be saved.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=(4, 8),
        default=4,
        help="Quantization bits. Use 4 for QLoRA/inference memory saving, or 8 for better accuracy.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Computation dtype used by bitsandbytes 4-bit layers.",
    )
    parser.add_argument(
        "--quant-type",
        choices=("nf4", "fp4"),
        default="nf4",
        help="4-bit quantization type. nf4 is usually preferred for QLoRA.",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        default=True,
        help="Use nested/double quantization for 4-bit weights.",
    )
    parser.add_argument(
        "--no-double-quant",
        action="store_false",
        dest="double_quant",
        help="Disable nested/double quantization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="",
        help="Optional prompt used to test the saved model after quantization.",
    )
    return parser.parse_args()


def import_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing dependency: {missing}\n"
            "Install the Qwen dependencies first, for example:\n"
            "  pip install -r model_Qwen3/requirements-qwen-quant.txt\n"
            "If you want GPU 4-bit quantization, install the CUDA build of PyTorch that matches your machine."
        ) from exc

    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def dtype_from_name(torch, dtype_name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def build_quant_config(args: argparse.Namespace, torch, BitsAndBytesConfig):
    if args.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_compute_dtype=dtype_from_name(torch, args.compute_dtype),
        bnb_4bit_use_double_quant=args.double_quant,
    )


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(
                f"Output directory already exists: {output_dir}\n"
                "Use --overwrite if you want to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_extra_files(source_dir: Path, output_dir: Path):
    for file_name in ("README.md", "LICENSE", "generation_config.json"):
        source_file = source_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, output_dir / file_name)


def write_quantization_info(args: argparse.Namespace, output_dir: Path):
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "bits": args.bits,
        "compute_dtype": args.compute_dtype if args.bits == 4 else None,
        "quant_type": args.quant_type if args.bits == 4 else None,
        "double_quant": args.double_quant if args.bits == 4 else None,
        "note": "Saved from a bitsandbytes-quantized Transformers model. Use mainly for inference or QLoRA-style loading.",
    }
    (output_dir / "quantization_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_generation(model, tokenizer, prompt: str):
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs.input_ids.shape[-1] :]
    print("\n===== Test output =====")
    print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


def main():
    args = parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()

    if not args.source.exists():
        raise SystemExit(f"Source model directory does not exist: {args.source}")

    torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = import_dependencies()

    print(f"Source model : {args.source}")
    print(f"Output model : {args.output}")
    print(f"Quantization : {args.bits}-bit")
    print(f"CUDA visible : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Warning: bitsandbytes 4-bit/8-bit quantization usually needs a CUDA GPU.")

    prepare_output_dir(args.output, args.overwrite)

    tokenizer = AutoTokenizer.from_pretrained(
        args.source,
        trust_remote_code=True,
        local_files_only=True,
    )

    quant_config = build_quant_config(args, torch, BitsAndBytesConfig)
    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant_config,
    )

    print("Saving quantized model and tokenizer...")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    copy_extra_files(args.source, args.output)
    write_quantization_info(args, args.output)

    print(f"Done. Quantized model saved to: {args.output}")

    if args.test_prompt:
        test_generation(model, tokenizer, args.test_prompt)


if __name__ == "__main__":
    main()
