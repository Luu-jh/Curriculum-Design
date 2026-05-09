# visual/charts.py
import re
from collections import Counter
from datetime import datetime, timedelta

import jieba
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, WordCloud

from core.models import Post, db


WORDCLOUD_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "吗", "呢", "啊", "吧", "这个", "那个", "有没有", "可以",
    "什么", "一下", "怎么", "现在", "真的", "还是", "就是", "不是", "没有", "想问", "求", "找", "如果",
    "还有", "的话", "以及", "然后", "但是", "因为", "所以", "已经", "还是说", "一个人", "一下子", "一下下",
    "我们", "你们", "他们", "她们", "自己", "学校", "同学", "请问", "有人", "一下嘛", "谢谢", "麻烦", "感觉",
    "应该", "这种", "那个啥", "就是想", "一下哈", "一下啦", "一下呗", "一下呢", "今天", "明天", "昨天",
}

CATEGORY_DISPLAY_ORDER = [
    "日常投稿",
    "选课互助",
    "二手闲置",
    "恋爱交友",
    "失物招领",
    "出行跑腿",
    "求助咨询",
    "文娱吐槽",
    "体测健康",
    "其他小类",
]

CATEGORY_GROUP_RULES = [
    ("日常投稿", ("日常投稿", "校园网", "宿舍", "新开水机", "干洗", "早点自助")),
    ("选课互助", ("选课互助", "转专业", "小学期", "体育期末", "计算机考研", "python", "算法", "作业", "课程", "课间", "学习", "考研", "期末")),
    ("二手闲置", ("二手闲置", "闲置", "求购", "转让", "电器", "拍立得", "洗衣液")),
    ("恋爱交友", ("恋爱交友", "亲友访校", "搭子", "寻找队友", "队友", "交友")),
    ("失物招领", ("失物招领", "找物", "寻物", "拿错", "遗失")),
    ("出行跑腿", ("拼车", "跑腿", "快递", "外卖", "代取", "出行", "退货")),
    ("求助咨询", ("求助", "求解", "求帮忙", "帮忙", "咨询", "请问")),
    ("文娱吐槽", ("吃瓜爆料", "游戏", "kpl", "明日方舟", "抽象文案", "沧元图", "吐槽", "爆料", "赛事", "不要把梦想埋没")),
    ("体测健康", ("体测", "体育", "健身")),
]

LLM_CATEGORY_ALIASES = {
    "日常": "日常投稿",
    "交易": "二手闲置",
    "求助": "求助咨询",
    "找人找物": "失物招领",
    "吐槽": "文娱吐槽",
    "其他": "其他小类",
}

OTHER_CATEGORY_LABEL = "其他小类"


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
                    title_opts=opts.TitleOpts(title="发帖类型占比", pos_left="center", pos_top="2%"),
                    legend_opts=opts.LegendOpts(pos_left="left", pos_top="18%"),
                )
                .dump_options()
            )

        counts = DashboardData._ordered_counts(df["display_category_1"].value_counts())
        data_pair = [[str(label), int(count)] for label, count in counts.items()]

        pie = (
            Pie()
            .add("", data_pair, radius=["36%", "64%"], center=["58%", "57%"])
            .set_global_opts(
                title_opts=opts.TitleOpts(title="发帖类型占比", pos_left="center", pos_top="2%"),
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
            return DashboardData._empty_bar("二级分类分布").dump_options()

        filtered = df
        title = "二级分类分布"
        if category_1:
            filtered = df[df["display_category_1"] == category_1]
            title = f"{category_1} 下的二级分类"

        if filtered.empty:
            return DashboardData._empty_bar(title).dump_options()

        counts = DashboardData._compact_counts(
            filtered["display_category_2"].fillna("未分类").value_counts(),
            max_items=12,
            min_count=2,
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
        llm_category = str(row.get("llm_category_1") or "").strip()
        raw_category = str(row.get("category") or "").strip()
        if llm_category and llm_category != "未分类":
            return DashboardData._group_category(llm_category, raw_category)
        return DashboardData._group_category(raw_category)

    @staticmethod
    def _display_category_2(row):
        llm_category = str(row.get("llm_category_2") or "").strip()
        raw_category = str(row.get("category") or "").strip()
        if llm_category and llm_category != "未分类":
            return llm_category
        return raw_category or "未分类"

    @staticmethod
    def _group_category(label, raw_label=""):
        label = str(label or "").strip()
        raw_label = str(raw_label or "").strip()
        text = f"{label} {raw_label}".strip()
        if not text or text == "未分类":
            return OTHER_CATEGORY_LABEL

        if label in LLM_CATEGORY_ALIASES:
            return LLM_CATEGORY_ALIASES[label]

        normalized = text.lower()
        if label.startswith("出") and not label.startswith("出行"):
            return "二手闲置"

        for group_name, keywords in CATEGORY_GROUP_RULES:
            if any(str(keyword).lower() in normalized for keyword in keywords):
                return group_name
        return OTHER_CATEGORY_LABEL

    @staticmethod
    def _ordered_counts(counts):
        ordered_items = []
        used_labels = set()
        for label in CATEGORY_DISPLAY_ORDER:
            if label in counts:
                ordered_items.append((label, int(counts[label])))
                used_labels.add(label)

        remaining_total = sum(int(count) for label, count in counts.items() if label not in used_labels)
        if remaining_total:
            for index, (label, count) in enumerate(ordered_items):
                if label == OTHER_CATEGORY_LABEL:
                    ordered_items[index] = (label, count + remaining_total)
                    break
            else:
                ordered_items.append((OTHER_CATEGORY_LABEL, remaining_total))
        return pd.Series(dict(ordered_items), dtype="int64")

    @staticmethod
    def _compact_counts(counts, max_items=12, min_count=2):
        if counts.empty:
            return counts
        head = counts[(counts >= min_count)].head(max_items)
        other_total = int(counts.drop(head.index, errors="ignore").sum())
        if other_total:
            head.loc[OTHER_CATEGORY_LABEL] = other_total
        return head

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
