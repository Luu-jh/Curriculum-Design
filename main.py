# main.py
# 项目主入口文件：
# 1. 检查并初始化 MySQL 数据库。
# 2. 创建 Flask 应用并注册数据库模型、页面路由。
# 3. 定义爬虫、情感分析、LLM 分类、智能体分析等后台任务。
# 4. 使用 APScheduler 定时调度后台任务。
# 5. 启动 Web 服务，供前端页面和 API 访问。
import pymysql
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask

import config
from analysis.agent_analyzer import AgentAnalyzer
from analysis.llm_qwen_classifier import QwenPostClassifier
from analysis.llm_qwen_lora_classifier import QwenLoraPostClassifier
from analysis.rule_based_snowNLP import SentimentAnalyzer
from core.crawler import CampusWallCrawler
from core.models import db
from visual.dashboard import dashboard_bp


def init_database():
    """启动 Flask 前，先确保配置中的 MySQL 数据库实例存在。"""
    print("正在检查数据库实例...")
    try:
        # 这里先不指定 database，只连接 MySQL 服务本身，用于执行 CREATE DATABASE。
        conn = pymysql.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASS,
            charset="utf8mb4",
        )
        with conn.cursor() as cursor:
            # 如果数据库不存在，就自动创建；字符集使用 utf8mb4，支持中文和 emoji。
            sql = (
                f"CREATE DATABASE IF NOT EXISTS `{config.DB_NAME}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            cursor.execute(sql)
        conn.commit()
        conn.close()
        print(f"数据库实例 [{config.DB_NAME}] 准备就绪")
    except Exception as e:
        print(f"自动建库失败，请检查 MySQL 是否启动或密码是否正确: {e}")
        raise SystemExit(1)


# -----------------------------
# Flask 应用与数据库模型初始化
# -----------------------------

# 先确保数据库存在，再创建 Flask 应用。
init_database()

app = Flask(__name__)

# 从 config.py 读取 SQLAlchemy 数据库连接配置。
app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS

# 把 SQLAlchemy 实例绑定到当前 Flask 应用。
db.init_app(app)

with app.app_context():
    # 根据 core/models.py 中的 Post、AgentReport 等模型自动检查/创建表结构。
    db.create_all()
    print("数据库表结构检查/创建完毕")

# 注册前端页面和 API 路由，具体接口在 visual/dashboard.py 中定义。
app.register_blueprint(dashboard_bp)

# 全局缓存对象：
# _qwen_classifier 用于复用 Qwen/LoRA 分类模型，避免每轮任务重复加载模型。
# _agent_analyzer 用于复用智能体分析对象。
# _scheduler 用于记录后台调度器是否已经启动，避免重复启动多个调度器。
_qwen_classifier = None
_agent_analyzer = None
_scheduler = None


# -----------------------------
# LLM 分类器构建
# -----------------------------

def build_llm_classifier(app):
    """根据配置选择普通 Qwen 分类器或 LoRA 分类器。"""
    backend = getattr(config, "LLM_CLASSIFIER_BACKEND", "base").lower()
    if backend == "lora":
        return QwenLoraPostClassifier(app)
    return QwenPostClassifier(app)


# -----------------------------
# 后台任务：情感分析
# -----------------------------

def run_sentiment_job():
    """对 posts 表中 sentiment 为“未分类”的帖子执行 SnowNLP 情感分析。"""
    if not getattr(config, "ENABLE_SENTIMENT_ANALYSIS", True):
        return
    print("触发情感分析任务...")
    analyzer = SentimentAnalyzer(app)
    analyzer.run_analysis()
    print("情感分析任务执行完毕")


# -----------------------------
# 后台任务：LLM/LoRA 两级分类
# -----------------------------

def run_llm_classification_job():
    """对 llm_category_1/2 未分类的帖子执行 Qwen 或 LoRA 两级分类。"""
    if not getattr(config, "ENABLE_LLM_CLASSIFICATION", False):
        return

    print("触发LLM分类任务...")
    try:
        global _qwen_classifier
        # 模型加载成本高，所以第一次任务创建后复用同一个分类器对象。
        if _qwen_classifier is None:
            _qwen_classifier = build_llm_classifier(app)
        _qwen_classifier.run_classification()
        print("LLM分类任务执行完毕")
    except Exception as exc:
        _qwen_classifier = None
        print(f"LLM分类任务失败，已跳过本轮，不阻塞后续智能体分析: {exc}")


# -----------------------------
# 后台任务：智能体舆情分析
# -----------------------------

def run_agent_analysis_job():
    """生成今日、近 7 天、近 30 天等周期的智能体分析报告。"""
    if not getattr(config, "ENABLE_AGENT_ANALYSIS", True):
        return

    print("触发智能体分析任务...")
    global _agent_analyzer, _qwen_classifier

    # 智能体报告默认可以只用规则分析；如果配置允许且分类模型已加载，则可复用模型做 LLM 报告增强。
    llm_generator = None
    if getattr(config, "ENABLE_AGENT_LLM_ANALYSIS", True) and _qwen_classifier is not None:
        llm_generator = _qwen_classifier

    # 复用 AgentAnalyzer，避免频繁创建对象；每次任务前同步最新 llm_generator。
    if _agent_analyzer is None:
        _agent_analyzer = AgentAnalyzer(app, llm_generator=llm_generator)
    else:
        _agent_analyzer.llm_generator = llm_generator

    _agent_analyzer.run()
    print("智能体分析任务执行完毕")


# -----------------------------
# 后台任务：校园墙爬虫
# -----------------------------

def run_crawler_job():
    """定时增量采集校园墙数据，并写入/更新 posts 表。"""
    print("触发定时爬虫任务...")
    crawler = CampusWallCrawler(app)
    crawler.run_incremental_crawl()
    print("定时爬虫任务执行完毕")


# -----------------------------
# 启动任务：历史数据回填与首轮分析
# -----------------------------

def run_startup_backfill_job():
    """服务启动后先回填最近一段时间的数据，再立即跑一轮分析链路。"""
    print(f"启动数据对齐：开始回填最近 {config.DAYS_TO_KEEP} 天的数据...")
    crawler = CampusWallCrawler(app)
    crawler.run_incremental_crawl(backfill_days=config.DAYS_TO_KEEP)
    print("启动数据对齐完成")
    # 回填完成后立即补齐情感、LLM 分类和智能体报告，减少页面首次打开时的空数据。
    run_sentiment_job()
    run_llm_classification_job()
    run_agent_analysis_job()


# -----------------------------
# APScheduler 后台调度器
# -----------------------------

def start_background_tasks():
    """启动后台调度器，周期性执行爬虫、分析和报告生成任务。"""
    global _scheduler
    # 防止 Flask 或外部调用重复启动多个调度器，造成任务重复执行。
    if _scheduler is not None:
        return

    scheduler = BackgroundScheduler()

    # 启动后异步做一次历史回填，不阻塞 Web 服务启动。
    scheduler.add_job(
        func=run_startup_backfill_job,
        trigger="date",
        run_date=datetime.now() + timedelta(seconds=1),
        id="startup_backfill",
        max_instances=1,
        coalesce=True,
    )

    # 增量爬虫保持较高频率。
    scheduler.add_job(
        func=run_crawler_job,
        trigger="interval",
        seconds=config.REFRESH_INTERVAL,
        id="crawler_job",
        max_instances=1,
        coalesce=True,
    )

    # 情感分析和 LLM 分类通常比爬虫慢，单独拆出来并放宽频率。
    analysis_interval = max(config.REFRESH_INTERVAL, 60)
    scheduler.add_job(
        func=run_sentiment_job,
        trigger="interval",
        seconds=analysis_interval,
        id="sentiment_job",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        func=run_llm_classification_job,
        trigger="interval",
        seconds=analysis_interval,
        id="llm_classification_job",
        max_instances=1,
        coalesce=True,
    )

    agent_interval = max(getattr(config, "AGENT_ANALYSIS_INTERVAL", 600), analysis_interval)
    scheduler.add_job(
        func=run_agent_analysis_job,
        trigger="interval",
        seconds=agent_interval,
        id="agent_analysis_job",
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    _scheduler = scheduler
    print(f"后台调度器已启动，爬虫频率：每 {config.REFRESH_INTERVAL} 秒一次")
    print(f"后台调度器已启动，分析频率：每 {analysis_interval} 秒一次")
    print(f"后台调度器已启动，智能体分析频率：每 {agent_interval} 秒一次")


# -----------------------------
# 程序启动入口
# -----------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  系统初始化：启动后台服务...")
    print("=" * 50)

    # 先启动后台调度器，再启动 Flask Web 服务。
    start_background_tasks()
    print(f"Web 服务已启动，可访问：http://127.0.0.1:{config.WEB_PORT}")
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        debug=config.DEBUG_MODE,
        use_reloader=False,
    )
