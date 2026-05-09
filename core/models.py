# core/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# 实例化 SQLAlchemy 对象
db = SQLAlchemy()


class Post(db.Model):
    """
    数据表：posts
    """
    __tablename__ = 'posts'

    # 基础信息
    id = db.Column(db.String(50), primary_key=True, comment='帖子唯一ID')
    category = db.Column(db.String(50), comment='原生分类：体测/日常等')
    content = db.Column(db.Text, comment='帖子详细内容')
    author = db.Column(db.String(100), comment='发帖人昵称')
    create_time = db.Column(db.DateTime, comment='发帖时间')

    # 动态数据
    views = db.Column(db.Integer, default=0, comment='浏览量')
    comments = db.Column(db.Integer, default=0, comment='评论数')
    likes = db.Column(db.Integer, default=0, comment='点赞数')

    # --- 分析与算法写入字段 ---
    sentiment = db.Column(db.String(20), default='未分类', comment='机器学习: 情感分析(正/中/负)')

    # 你刚才补充的两个字段：LLM 详细分类
    llm_category_1 = db.Column(db.String(50), default='未分类', comment='LLM: 一级分类')
    llm_category_2 = db.Column(db.String(50), default='未分类', comment='LLM: 二级分类')

    def __repr__(self):
        return f"<Post {self.id} - {self.author}>"


class AgentReport(db.Model):
    """
    数据表：agent_reports
    保存智能体分析模块生成的结构化舆情报告。
    """
    __tablename__ = 'agent_reports'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    period = db.Column(db.String(20), index=True, nullable=False, comment='分析周期：today/week/month')
    post_count = db.Column(db.Integer, default=0, comment='本次分析覆盖的帖子数量')
    report_json = db.Column(db.Text, nullable=False, comment='智能体报告JSON')
    status = db.Column(db.String(20), default='success', comment='生成状态：success/fallback/failed')
    error_message = db.Column(db.Text, comment='失败或降级原因')
    created_at = db.Column(db.DateTime, default=datetime.now, index=True, comment='报告生成时间')

    def __repr__(self):
        return f"<AgentReport {self.period} - {self.created_at}>"
