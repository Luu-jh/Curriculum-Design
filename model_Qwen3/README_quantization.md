# Qwen3 模型量化说明

这个目录现在有这些和量化相关的东西：

- `Qwen3-4B-Instruct-2507/`：原始 Hugging Face 格式模型。
- `Qwen3-4B-Instruct-2507-bnb-4bit/`：已经生成好的 4bit 量化模型。
- `quantize_qwen_bnb.py`：量化脚本。
- `requirements-qwen-quant.txt`：除 PyTorch 外的量化/推理依赖。

## 1. 量化是什么

原始模型权重大多是 `bfloat16/float16`，每个参数大约 2 字节。4bit 量化会把权重压到约 0.5 字节一个参数，显存占用会明显下降。

粗略理解：

```text
原始模型：更大，更吃显存，精度更稳
4bit 模型：更小，更省显存，适合本地推理和 QLoRA
8bit 模型：介于两者之间
```

## 2. 安装依赖

当前项目 `.venv` 已经装好了 CUDA 版 PyTorch 和量化依赖。如果你以后换电脑或重建虚拟环境，需要重新安装。

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r model_Qwen3\requirements-qwen-quant.txt
```

注意：`bitsandbytes` 的 4bit/8bit 量化通常需要 NVIDIA CUDA GPU。没有 CUDA 时，脚本可能无法真正执行 4bit 加载。

## 3. 执行 4bit 量化

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe model_Qwen3\quantize_qwen_bnb.py --bits 4 --overwrite
```

默认输入：

```text
model_Qwen3/Qwen3-4B-Instruct-2507
```

默认输出：

```text
model_Qwen3/Qwen3-4B-Instruct-2507-bnb-4bit
```

可以加一个测试 prompt：

```powershell
.\.venv\Scripts\python.exe model_Qwen3\quantize_qwen_bnb.py --bits 4 --overwrite --test-prompt "请把这条校园墙帖子分成一级分类和二级分类：找4月30号下午前两节的代课，女生。"
```

## 4. 执行 8bit 量化

```powershell
.\.venv\Scripts\python.exe model_Qwen3\quantize_qwen_bnb.py --bits 8 --output model_Qwen3\Qwen3-4B-Instruct-2507-bnb-8bit --overwrite
```

## 5. 和后面 LoRA 的关系

后面做 LoRA 时，通常不是直接改原始模型全部权重，而是：

```text
原始/量化 base model + 很小的 LoRA adapter
```

如果做 QLoRA，就是把 base model 用 4bit 加载，然后只训练 LoRA adapter。这样显存压力小很多。

## 6. 分类功能应该加在哪里

建议新增：

```text
analysis/llm_qwen_classifier.py
```

它负责：

```text
读取 posts 表里 llm_category_1/llm_category_2 还是“未分类”的帖子
调用 Qwen 模型生成一级分类和二级分类
写回 core.models.Post 的 llm_category_1 和 llm_category_2 字段
```

然后在 `main.py` 里像现在调用情感分析一样，增加一个 `run_llm_classification_job()`。

前端图表目前 `visual/charts.py` 已经在用 `llm_category_1` 做饼图，所以分类结果写进数据库后，饼图就能直接更新。
