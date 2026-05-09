# main.py
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
    print("正在检查数据库实例...")
    try:
        conn = pymysql.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASS,
            charset="utf8mb4",
        )
        with conn.cursor() as cursor:
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


init_database()

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS

db.init_app(app)

with app.app_context():
    db.create_all()
    print("数据库表结构检查/创建完毕")

app.register_blueprint(dashboard_bp)

_qwen_classifier = None
_agent_analyzer = None
_scheduler = None


def build_llm_classifier(app):
    backend = getattr(config, "LLM_CLASSIFIER_BACKEND", "base").lower()
    if backend == "lora":
        return QwenLoraPostClassifier(app)
    return QwenPostClassifier(app)


def run_sentiment_job():
    if not getattr(config, "ENABLE_SENTIMENT_ANALYSIS", True):
        return
    print("触发情感分析任务...")
    analyzer = SentimentAnalyzer(app)
    analyzer.run_analysis()
    print("情感分析任务执行完毕")


def run_llm_classification_job():
    if not getattr(config, "ENABLE_LLM_CLASSIFICATION", False):
        return

    print("触发LLM分类任务...")
    try:
        global _qwen_classifier
        if _qwen_classifier is None:
            _qwen_classifier = build_llm_classifier(app)
        _qwen_classifier.run_classification()
        print("LLM分类任务执行完毕")
    except Exception as exc:
        _qwen_classifier = None
        print(f"LLM分类任务失败，已跳过本轮，不阻塞后续智能体分析: {exc}")


def run_agent_analysis_job():
    if not getattr(config, "ENABLE_AGENT_ANALYSIS", True):
        return

    print("触发智能体分析任务...")
    global _agent_analyzer, _qwen_classifier

    llm_generator = None
    if getattr(config, "ENABLE_AGENT_LLM_ANALYSIS", True) and _qwen_classifier is not None:
        llm_generator = _qwen_classifier

    if _agent_analyzer is None:
        _agent_analyzer = AgentAnalyzer(app, llm_generator=llm_generator)
    else:
        _agent_analyzer.llm_generator = llm_generator

    _agent_analyzer.run()
    print("智能体分析任务执行完毕")


def run_crawler_job():
    print("触发定时爬虫任务...")
    crawler = CampusWallCrawler(app)
    crawler.run_incremental_crawl()
    print("定时爬虫任务执行完毕")


def run_startup_backfill_job():
    print(f"启动数据对齐：开始回填最近 {config.DAYS_TO_KEEP} 天的数据...")
    crawler = CampusWallCrawler(app)
    crawler.run_incremental_crawl(backfill_days=config.DAYS_TO_KEEP)
    print("启动数据对齐完成")
    run_sentiment_job()
    run_llm_classification_job()
    run_agent_analysis_job()


def start_background_tasks():
    global _scheduler
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


if __name__ == "__main__":
    print("=" * 50)
    print("  系统初始化：启动后台服务...")
    print("=" * 50)

    start_background_tasks()
    print(f"Web 服务已启动，可访问：http://127.0.0.1:{config.WEB_PORT}")
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        debug=config.DEBUG_MODE,
        use_reloader=False,
    )
