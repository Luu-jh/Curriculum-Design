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
- 可选微调：PEFT/LoRA

## 目录结构

```text
analysis/       情感分析、大模型分类、智能体分析、智能体问答
core/           数据模型与校园墙爬虫
visual/         Flask 路由与图表数据生成
templates/      前端页面模板
temolates/      历史模板目录
model_Qwen3/    轻量模型说明、依赖与 LoRA 配置，模型权重不提交
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

## 安全说明

`config.py`、本地数据库导出、爬虫原始数据、模型权重和运行日志不会提交到仓库。提交到 GitHub 的是可复现的项目代码和示例配置，真实密码、Cookie、模型文件需要在本地自行配置。
