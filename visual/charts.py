# visual/charts.py
import re
from collections import Counter
from datetime import datetime, timedelta

import jieba
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, WordCloud

from analysis.llm_qwen_classifier import LABEL_SCHEMA
from core.models import Post, db


WORDCLOUD_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "吗", "呢", "啊", "吧", "这个", "那个", "有没有", "可以",
    "什么", "一下", "怎么", "现在", "真的", "还是", "就是", "不是", "没有", "想问", "求", "找", "如果",
    "还有", "的话", "以及", "然后", "但是", "因为", "所以", "已经", "还是说", "一个人", "一下子", "一下下",
    "我们", "你们", "他们", "她们", "自己", "学校", "同学", "请问", "有人", "一下嘛", "谢谢", "麻烦", "感觉",
    "应该", "这种", "那个啥", "就是想", "一下哈", "一下啦", "一下呗", "一下呢", "今天", "明天", "昨天",
}

UNCLASSIFIED_LABEL = "未分类"
CATEGORY_DISPLAY_ORDER = list(LABEL_SCHEMA.keys()) + [UNCLASSIFIED_LABEL]
CATEGORY_2_DISPLAY_ORDER = [
    label
    for labels in LABEL_SCHEMA.values()
    for label in labels
] + [UNCLASSIFIED_LABEL]


class DashboardData:
    @staticmethod
    def get_raw_df():
        """Load post data from MySQL into a DataFrame for chart aggregation."""
        query = db.session.query(Post).statement
        df = pd.read_sql(query, db.engine)
        if not df.empty:
            df["create_time"] = pd.to_datetime(df["create_time"], errors="coerce")
            for column in ("views", "comments", "likes"):
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
            df["content"] = df["content"].fillna("")
            df["sentiment"] = df["sentiment"].fillna("未分类")
            df["llm_category_1"] = df["llm_category_1"].fillna("未分类")
            df["llm_category_2"] = df["llm_category_2"].fillna("未分类")
            df["category"] = df["category"].fillna("未分类")
            df["display_category_1"] = df.apply(DashboardData._display_category_1, axis=1)
            df["display_category_2"] = df.apply(DashboardData._display_category_2, axis=1)
        return df

    @staticmethod
    def get_line_chart(period="week"):
        """Generate post count trend for the last 7 or 30 days."""
        df = DashboardData.get_raw_df()
        days = 30 if period == "month" else 7

        if df.empty:
            return DashboardData._empty_line(days).dump_options()

        today = datetime.now().date()
        start_date = today - timedelta(days=days - 1)
        df = df.dropna(subset=["create_time"])
        df["date"] = df["create_time"].dt.date
        df_filtered = df[df["date"] >= start_date]

        all_dates = pd.date_range(start=start_date, end=today).date
        daily_counts = df_filtered.groupby("date").size().reindex(all_dates, fill_value=0)

        line = (
            Line()
            .add_xaxis([str(day) for day in all_dates])
            .add_yaxis("发帖量", daily_counts.astype(int).tolist(), is_smooth=True, color="#5470c6")
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"近{days}天发帖趋势", pos_left="center", pos_top="2%"),
                legend_opts=opts.LegendOpts(pos_top="12%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                yaxis_opts=opts.AxisOpts(min_=0, min_interval=1),
            )
        )
        return line.dump_options()

    @staticmethod
    def get_pie_chart():
        """Generate category distribution pie chart."""
        df = DashboardData.get_raw_df()
        if df.empty:
            return (
                Pie()
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="LLM一级分类占比", pos_left="center", pos_top="2%"),
                    legend_opts=opts.LegendOpts(pos_left="left", pos_top="18%"),
                )
                .dump_options()
            )

        counts = DashboardData._ordered_counts(df["display_category_1"].value_counts(), CATEGORY_DISPLAY_ORDER)
        data_pair = [[str(label), int(count)] for label, count in counts.items()]

        pie = (
            Pie()
            .add("", data_pair, radius=["36%", "64%"], center=["58%", "57%"])
            .set_global_opts(
                title_opts=opts.TitleOpts(title="LLM一级分类占比", pos_left="center", pos_top="2%"),
                legend_opts=opts.LegendOpts(
                    type_="scroll",
                    orient="vertical",
                    pos_left="2%",
                    pos_top="18%",
                    pos_bottom="8%",
                ),
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}\n{d}%"))
        )
        return pie.dump_options()

    @staticmethod
    def get_category_2_bar_chart(category_1=None):
        """Generate a drill-down bar chart for second-level categories."""
        df = DashboardData.get_raw_df()
        if df.empty:
            return DashboardData._empty_bar("LLM二级分类分布").dump_options()

        filtered = df
        title = "LLM二级分类分布"
        if category_1:
            filtered = df[df["display_category_1"] == category_1]
            title = f"{category_1} 下的 LLM二级分类"

        if filtered.empty:
            return DashboardData._empty_bar(title).dump_options()

        counts = DashboardData._ordered_counts(
            filtered["display_category_2"].fillna("未分类").value_counts(),
            CATEGORY_2_DISPLAY_ORDER,
        )
        labels = [str(label) for label in counts.index.tolist()]
        values = [int(value) for value in counts.tolist()]

        bar = (
            Bar()
            .add_xaxis(labels)
            .add_yaxis("帖子数", values, color="#91cc75", category_gap="35%")
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center", pos_top="2%"),
                legend_opts=opts.LegendOpts(pos_top="12%"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=18)),
                yaxis_opts=opts.AxisOpts(min_=0, min_interval=1),
            )
        )
        return bar.dump_options()

    @staticmethod
    def get_wordcloud_chart():
        """Generate word cloud from posts in the last 7 days."""
        df = DashboardData.get_raw_df()
        if df.empty:
            return WordCloud().set_global_opts(title_opts=opts.TitleOpts(title="本周热词", pos_left="center")).dump_options()

        cutoff = datetime.now() - timedelta(days=7)
        recent_texts = df[df["create_time"] >= cutoff]["content"].tolist()

        words = []
        for text in recent_texts:
            for word in jieba.cut(str(text)):
                normalized = word.strip().lower()
                if not DashboardData._is_valid_wordcloud_word(normalized):
                    continue
                words.append(normalized)

        word_counts = Counter(words).most_common(100)
        wc = (
            WordCloud()
            .add("", word_counts, word_size_range=[15, 50], shape="circle")
            .set_global_opts(title_opts=opts.TitleOpts(title="本周热词", pos_left="center"))
        )
        return wc.dump_options()

    @staticmethod
    def get_top_hot_posts(limit=10):
        """Return hot posts ordered by a simple interaction score."""
        posts = (
            Post.query.order_by(
                (Post.comments * 5 + Post.likes * 3 + Post.views).desc(),
                Post.create_time.desc(),
            )
            .limit(limit)
            .all()
        )

        hot_posts = []
        for post in posts:
            content = post.content or ""
            snippet = content[:48] + "..." if len(content) > 48 else content
            hot_posts.append(
                {
                    "id": post.id,
                    "snippet": snippet,
                    "comments": post.comments or 0,
                    "likes": post.likes or 0,
                    "views": post.views or 0,
                    "sentiment": post.sentiment or "未分类",
                    "hot_score": (post.comments or 0) * 5 + (post.likes or 0) * 3 + (post.views or 0),
                }
            )
        return hot_posts

    @staticmethod
    def _is_valid_wordcloud_word(word):
        if not word or len(word) <= 1:
            return False
        if word in WORDCLOUD_STOP_WORDS:
            return False
        if word.isdigit():
            return False
        if re.fullmatch(r"[\W_]+", word):
            return False
        if re.fullmatch(r"[a-zA-Z]+", word):
            return False
        return True

    @staticmethod
    def _display_category_1(row):
        return DashboardData._display_label(row.get("llm_category_1"))

    @staticmethod
    def _display_category_2(row):
        return DashboardData._display_label(row.get("llm_category_2"))

    @staticmethod
    def _display_label(label):
        label = str(label or "").strip()
        if not label or label.lower() == "nan":
            return UNCLASSIFIED_LABEL
        return label

    @staticmethod
    def _ordered_counts(counts, display_order):
        ordered_items = []
        used_labels = set()
        for label in display_order:
            if label in counts:
                ordered_items.append((label, int(counts[label])))
                used_labels.add(label)

        remaining_items = [
            (str(label), int(count))
            for label, count in counts.items()
            if label not in used_labels
        ]
        ordered_items.extend(sorted(remaining_items, key=lambda item: item[1], reverse=True))
        return pd.Series(dict(ordered_items), dtype="int64")

    @staticmethod
    def _empty_line(days):
        today = datetime.now().date()
        start_date = today - timedelta(days=days - 1)
        all_dates = pd.date_range(start=start_date, end=today).date
        return (
            Line()
            .add_xaxis([str(day) for day in all_dates])
            .add_yaxis("发帖量", [0 for _ in all_dates], is_smooth=True, color="#5470c6")
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"近{days}天发帖趋势", pos_left="center", pos_top="2%"),
                legend_opts=opts.LegendOpts(pos_top="12%"),
            )
        )

    @staticmethod
    def _empty_bar(title):
        return (
            Bar()
            .add_xaxis(["暂无数据"])
            .add_yaxis("帖子数", [0], color="#91cc75")
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center", pos_top="2%"),
                legend_opts=opts.LegendOpts(pos_top="12%"),
                yaxis_opts=opts.AxisOpts(min_=0, min_interval=1),
            )
        )
