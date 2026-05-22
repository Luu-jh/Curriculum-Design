# 校园墙数据中台与智能体分析系统

这是一个基于 Flask、MySQL、Pyecharts、SnowNLP 和本地 Qwen 大模型的校园墙数据分析课设项目。系统会采集校园墙帖子，写入数据库，并提供数据看板、热词分析、分类占比、情感分析、智能体深度研判和大模型问答功能。

## 主要功能

- 数据采集：通过接口分页采集校园墙帖子，支持启动回填和定时增量爬取。
- 数据入库：使用 MySQL 保存帖子内容、互动量、发布时间、情感结果和分类结果。
- 数据看板：展示近 7 天/近 30 天发帖趋势、发帖类型占比、二级分类分布、本周热词和热门帖子。
- 情感分析：使用 SnowNLP 对帖子文本做正向、负向和中性判断。
- 大模型分类：可使用本地 Qwen2.5-0.5B-Instruct 对帖子做一级/二级分类。
- 智能体分析：根据数据库实时生成总体研判、风险提醒、热点话题、建议和代表性帖子。
- 大模型问答：基于当前周期帖子和智能体报告构建上下文，回答用户关于校园墙数据的问题。

## 技术栈

- 后端框架：Flask
- ORM：Flask-SQLAlchemy
- 数据库：MySQL + PyMySQL
- 定时任务：APScheduler
- 数据处理：pandas
- 可视化：Pyecharts
- 中文分词：jieba
- 情感分析：SnowNLP
- 本地大模型：Qwen2.5-0.5B-Instruct + Transformers + PyTorch
- 可选微调：PEFT/LoRA/QLoRA + bitsandbytes

## 目录结构

```text
analysis/       情感分析、大模型分类、智能体分析、智能体问答
core/           数据模型与校园墙爬虫
visual/         Flask 路由与图表数据生成
templates/      前端页面模板
temolates/      历史模板目录
model_Qwen3/    轻量模型说明、量化脚本、LoRA/QLoRA 微调配置，模型权重不提交
test/           诊断脚本
main.py         项目入口
config.example.py 示例配置文件
```

## 本地运行

1. 创建并激活虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. 安装依赖。

```powershell
pip install -r requirements.txt
```

3. 复制配置文件。

```powershell
Copy-Item config.example.py config.py
```

4. 修改 `config.py`。

- 填写本地 MySQL 用户名、密码和数据库名。
- 填写校园墙接口地址、Cookie、Referer 等爬虫请求信息。
- 如果使用本地大模型，确认 `QWEN_CLASSIFIER_MODEL_DIR` 指向模型目录。

5. 准备数据库。

启动 MySQL 服务即可，项目启动时会自动创建配置中的数据库和表。

6. 启动项目。

```powershell
python main.py
```

浏览器访问：

```text
http://127.0.0.1:5000
```

## 本地 Qwen 模型

推荐使用轻量模型：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

默认本地目录：

```text
model_Qwen3/Qwen2.5-0.5B-Instruct
```

模型权重文件较大，已通过 `.gitignore` 排除，不会上传到 GitHub。下载后放入上述目录，或在环境变量中指定：

```powershell
$env:QWEN_CLASSIFIER_MODEL_DIR="D:\path\to\Qwen2.5-0.5B-Instruct"
```

如果只想运行规则分析和数据看板，可以在 `config.py` 中关闭大模型相关开关。

## LoRA 微调

项目已经支持基于本地 `Qwen2.5-0.5B-Instruct` 的校园墙分类 LoRA/QLoRA 微调。当前在 Windows + Python 3.13 + RTX 3050 Ti Laptop GPU 环境下验证通过。

当前已验证的 IDE 解释器：

```text
C:\Users\罗加海\AppData\Local\Programs\Python\Python313\python.exe
```

当前已验证的核心依赖：

```text
torch 2.12.0+cu126
peft 0.19.1
transformers 5.8.0
accelerate 1.13.0
bitsandbytes 0.49.2
```

LoRA 相关文件集中在：

```text
model_Qwen3/lora/
```

核心文件：

```text
model_Qwen3/lora/lora_config.py              LoRA 路径与默认训练超参数
model_Qwen3/lora/train_qlora.py              QLoRA 微调入口
model_Qwen3/lora/evaluate_lora_comparison.py 微调前后效果对比
model_Qwen3/lora/build_real_lora_data.py      从真实校园墙 CSV 生成人工标注数据
model_Qwen3/lora/build_demo_lora_data.py      示例训练/验证数据生成
model_Qwen3/lora/data/train.jsonl             训练集
model_Qwen3/lora/data/valid.jsonl             验证集
model_Qwen3/lora/data/real_label_manifest.json 真实标注数据统计
model_Qwen3/lora/output/campus_classifier_real_posts/ 当前 LoRA adapter 输出目录
```

推荐从项目根目录执行：

```powershell
cd D:\python\大三下课设
python model_Qwen3\lora\build_real_lora_data.py
python model_Qwen3\lora\train_qlora.py --base-model model_Qwen3\Qwen2.5-0.5B-Instruct --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 8 --max-length 768 --epochs 3 --output-dir model_Qwen3\lora\output\campus_classifier_real_posts
```

微调前后对比：

```powershell
python model_Qwen3\lora\evaluate_lora_comparison.py --json-output model_Qwen3\lora\output\real_posts_comparison.json
```

当前真实标注数据规模为训练集 147 条、验证集 56 条；本地验证结果为 base 完全命中率 16.1%，LoRA 完全命中率 41.1%。

在 IDE 中直接运行时，建议配置：

```text
Script path: D:\python\大三下课设\model_Qwen3\lora\train_qlora.py
Working directory: D:\python\大三下课设
Python interpreter: C:\Users\罗加海\AppData\Local\Programs\Python\Python313\python.exe
```

更多说明见：

```text
model_Qwen3/lora/README_lora_finetuning.md
model_Qwen3/lora/LORA_STRUCTURE_AND_FLOW.md
```

## 安全说明

`config.py`、本地数据库导出、爬虫原始数据、模型权重、LoRA 输出权重和运行日志不会提交到仓库。提交到 GitHub 的是可复现的项目代码和示例配置，真实密码、Cookie、模型文件需要在本地自行配置。
