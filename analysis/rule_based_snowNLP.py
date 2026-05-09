# analysis/rule_based_snowNLP.py
import re
from collections import Counter

from snownlp import SnowNLP

import config
from core.models import Post, db


class SentimentAnalyzer:
    def __init__(self, app):
        self.app = app
        self.positive_threshold = getattr(config, "SENTIMENT_POSITIVE_THRESHOLD", 0.65)
        self.negative_threshold = getattr(config, "SENTIMENT_NEGATIVE_THRESHOLD", 0.35)
        self.batch_size = getattr(config, "SENTIMENT_BATCH_SIZE", 100)

    def run_analysis(self, limit=None):
        """Classify unprocessed posts and write results to Post.sentiment."""
        with self.app.app_context():
            total_pending = self._pending_count()
            if total_pending == 0:
                print("情感分析：当前没有需要分类的数据。")
                return {"total": 0, "正向": 0, "中性": 0, "负向": 0, "失败": 0}

            target_total = min(total_pending, limit) if limit else total_pending
            print(f"情感分析：发现 {total_pending} 条未分类数据，本轮处理 {target_total} 条。")

            stats = Counter()
            processed = 0

            while processed < target_total:
                current_limit = min(self.batch_size, target_total - processed)
                posts = self._load_pending_posts(current_limit)
                if not posts:
                    break

                for post in posts:
                    label = self.classify(post.content)
                    post.sentiment = label
                    stats[label] += 1
                    processed += 1

                try:
                    db.session.commit()
                except Exception as exc:
                    db.session.rollback()
                    stats["失败"] += len(posts)
                    print(f"情感分析：批量写入失败，已回滚本批 {len(posts)} 条: {exc}")
                    break

            result = {
                "total": processed,
                "正向": stats["正向"],
                "中性": stats["中性"],
                "负向": stats["负向"],
                "失败": stats["失败"],
            }
            print(
                "情感分析完成："
                f"处理 {result['total']} 条，"
                f"正向 {result['正向']} 条，"
                f"中性 {result['中性']} 条，"
                f"负向 {result['负向']} 条，"
                f"失败 {result['失败']} 条。"
            )
            return result

    def classify(self, text):
        """Return one of: 正向 / 中性 / 负向."""
        text = self._clean_text(text)
        if not text:
            return "中性"

        try:
            score = SnowNLP(text).sentiments
        except Exception:
            return "中性"

        if score >= self.positive_threshold:
            return "正向"
        if score <= self.negative_threshold:
            return "负向"
        return "中性"

    def _pending_count(self):
        return Post.query.filter(Post.sentiment == "未分类").count()

    def _load_pending_posts(self, limit):
        return (
            Post.query.filter(Post.sentiment == "未分类")
            .order_by(Post.create_time.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def _clean_text(text):
        if not text:
            return ""
        text = re.sub(r"\s+", " ", str(text)).strip()
        # SnowNLP on very long informal posts is slow and often noisy; keep the most useful prefix.
        return text[:500]
