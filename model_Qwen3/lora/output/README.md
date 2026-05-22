# LoRA 输出目录说明

本目录保存校园墙帖子分类任务的 LoRA/QLoRA 训练产物、检查点和模型效果对比结果。这里的文件不是完整 Qwen 基座模型，而是基于 `model_Qwen3/Qwen2.5-0.5B-Instruct` 训练出来的 LoRA adapter。推理时必须同时准备基础模型目录和 adapter 目录。

## 当前推荐使用的输出

当前项目配置文件 `model_Qwen3/lora/lora_config.py` 中：

```python
QWEN_LORA_ADAPTER_DIR = LORA_DIR / "output" / "campus_classifier_real_posts"
```

因此正式分类推理默认使用：

```text
model_Qwen3/lora/output/campus_classifier_real_posts
```

该目录是基于真实校园墙标注数据训练的 adapter，相比 smoke check 和 demo adapter 更适合作为当前项目的默认分类模型。

## 目录结构

```text
output/
├─ campus_classifier/              # demo/小样本训练得到的 adapter
├─ campus_classifier_real_posts/   # 真实校园墙帖子训练得到的正式 adapter
├─ smoke_check/                    # 命令行快速冒烟训练产物
├─ ide_smoke_check/                # IDE 运行快速冒烟训练产物
├─ last_comparison.json            # demo adapter 与 base 模型对比结果
├─ ide_last_comparison.json        # IDE smoke adapter 与 base 模型对比结果
└─ real_posts_comparison.json      # 真实帖子 adapter 与 base 模型对比结果
```

各 adapter 子目录通常包含：

```text
adapter_config.json          # LoRA adapter 配置
adapter_model.safetensors    # LoRA adapter 权重
tokenizer.json               # tokenizer 文件
tokenizer_config.json        # tokenizer 配置
chat_template.jinja          # 对话模板
training_meta.json           # 本次训练使用的数据规模和超参数
checkpoint-*/                # 训练中间检查点，含 optimizer/scheduler/trainer_state 等
```

## 各输出目录说明

| 目录 | 用途 | 训练数据 | 训练轮数 | 备注 |
| --- | --- | --- | --- | --- |
| `campus_classifier_real_posts/` | 当前正式推荐 adapter | 训练集 147 条，验证集 56 条 | 3 | 真实校园墙帖子标注数据训练，当前推理配置默认指向它。 |
| `campus_classifier/` | 早期 demo adapter | 训练集 30 条，验证集 15 条 | 3 | 用于验证 LoRA 流程，数据量较小，不建议作为最终模型。 |
| `smoke_check/` | 命令行冒烟检查 | 训练集 30 条，验证集 15 条 | 1 | 只用于确认训练脚本、CUDA、bitsandbytes、PEFT 等环境能跑通。 |
| `ide_smoke_check/` | IDE 冒烟检查 | 训练集 30 条，验证集 15 条 | 1 | 用于确认在 PyCharm/IDE 解释器中也能跑通训练流程。 |

## 当前效果对比

对比文件记录了 base Qwen 与 LoRA adapter 在验证集上的分类命中情况：

| 文件 | 后端 | 样本数 | 一级分类准确率 | 二级分类准确率 | 完全命中率 |
| --- | --- | ---: | ---: | ---: | ---: |
| `last_comparison.json` | base | 15 | 20.0% | 6.7% | 6.7% |
| `last_comparison.json` | lora | 15 | 60.0% | 46.7% | 46.7% |
| `ide_last_comparison.json` | base | 15 | 26.7% | 6.7% | 6.7% |
| `ide_last_comparison.json` | lora | 15 | 60.0% | 46.7% | 46.7% |
| `real_posts_comparison.json` | base | 56 | 19.6% | 16.1% | 16.1% |
| `real_posts_comparison.json` | lora | 56 | 53.6% | 41.1% | 41.1% |

其中 `real_posts_comparison.json` 最能代表当前正式 adapter 的效果。可以看到 LoRA 后一级分类、二级分类和完全命中率都有明显提升。

## 如何在项目中使用

1. 确认基础模型存在：

```text
model_Qwen3/Qwen2.5-0.5B-Instruct
```

2. 确认 `model_Qwen3/lora/lora_config.py` 中的推理路径指向正式 adapter：

```python
QWEN_LORA_INFERENCE_BASE_MODEL_DIR = BASE_DIR / "model_Qwen3" / "Qwen2.5-0.5B-Instruct"
QWEN_LORA_ADAPTER_DIR = LORA_DIR / "output" / "campus_classifier_real_posts"
```

3. 在 `config.py` 中启用 LoRA 分类后端：

```python
ENABLE_LLM_CLASSIFICATION = True
LLM_CLASSIFIER_BACKEND = "lora"
```

4. 启动项目：

```powershell
python main.py
```

运行后，`analysis/llm_qwen_lora_classifier.py` 会先加载基础模型，再通过 `PeftModel.from_pretrained` 加载本目录下的 LoRA adapter，并把分类结果写入 `posts.llm_category_1` 与 `posts.llm_category_2`。

## 重新训练与重新评估

从项目根目录执行训练：

```powershell
python model_Qwen3\lora\train_qlora.py --base-model model_Qwen3\Qwen2.5-0.5B-Instruct --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8 --max-length 768 --epochs 3 --output-dir model_Qwen3\lora\output\campus_classifier_real_posts
```

训练完成后，可运行对比评估：

```powershell
python model_Qwen3\lora\evaluate_lora_comparison.py --json-output model_Qwen3\lora\output\real_posts_comparison.json
```

如果要先检查环境是否可用，可以把 `--epochs` 调成 1，并输出到 `smoke_check` 或新的临时目录。

## 清理建议

- `adapter_config.json`、`adapter_model.safetensors`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`training_meta.json` 是 adapter 推理需要重点保留的文件。
- `checkpoint-*` 目录主要用于中断恢复训练，里面的 `optimizer.pt` 通常很大。如果已经确认最终 adapter 可用，且不需要继续从中间状态恢复训练，可以备份后删除旧 checkpoint 以节省空间。
- `real_posts_comparison.json` 建议保留，它是说明当前 adapter 效果最直接的评估记录。
- 本目录包含模型训练产物，默认被 `.gitignore` 忽略，不建议直接提交到 GitHub。需要迁移到其他机器时，建议整体复制基础模型目录和当前正式 adapter 目录。

## 常见问题

**1. 只有这个 output 目录能不能推理？**

不能。LoRA adapter 只保存增量权重，推理时还需要同训练时一致的 Qwen 基础模型目录。

**2. 为什么有多个 output 子目录？**

这些目录对应不同训练目的：`smoke_check` 和 `ide_smoke_check` 用于环境验证，`campus_classifier` 是早期小样本训练结果，`campus_classifier_real_posts` 是当前正式推荐结果。

**3. 为什么 checkpoint 目录比最终 adapter 大很多？**

checkpoint 里除了 adapter 权重，还保存 optimizer、scheduler、trainer_state、training_args 等训练恢复信息，所以体积更大。最终推理通常不需要这些训练状态文件。

**4. 如果重新训练覆盖了正式 adapter，旧结果还能恢复吗？**

如果没有提前备份，覆盖后无法从最终 adapter 文件自动恢复旧版本。重新训练前建议把 `campus_classifier_real_posts` 复制成带日期的目录，例如 `campus_classifier_real_posts_20260512`。
