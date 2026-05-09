import json
import re
from collections import Counter
from datetime import datetime

import config
from analysis.agent_analyzer import AgentAnalyzer, PERIOD_LABELS
from analysis.llm_qwen_classifier import QwenPostClassifier
from analysis.llm_qwen_lora_classifier import QwenLoraPostClassifier


QUESTION_STOP_WORDS = {
    "最近", "今天", "这个", "那个", "一下", "请问", "分析", "什么", "哪些", "怎么", "有没有",
    "是否", "主要", "情况", "问题", "帮我", "校园墙", "学生", "帖子", "相关", "一下子",
}

BASE_BACKEND_DEPENDENCIES = ("torch", "transformers", "accelerate")
LORA_BACKEND_DEPENDENCIES = BASE_BACKEND_DEPENDENCIES + ("peft",)


class AgentQuestionAnswerer:
    """Answer free-form questions with local Qwen grounded in posts table context."""

    def __init__(self, app):
        self.app = app
        self.max_context_posts = getattr(config, "AGENT_QA_MAX_CONTEXT_POSTS", 30)
        self.max_new_tokens = getattr(config, "AGENT_QA_MAX_NEW_TOKENS", 900)
        self.max_question_length = getattr(config, "AGENT_QA_MAX_QUESTION_LENGTH", 300)
        self.generator = None

    def answer(self, question, period="week"):
        question = self._clean_question(question)
        period = AgentAnalyzer.normalize_period(period)

        with self.app.app_context():
            analyzer = AgentAnalyzer(self.app, use_llm=False)
            posts = analyzer._load_posts(period)
            material = analyzer._build_material(posts, period)
            report = AgentAnalyzer.get_latest_report(period)
            if report is None or AgentAnalyzer.is_report_stale(report, material["data_state"]):
                report = analyzer._build_rule_report(material)
                report["source"] = "rules"
                report["status"] = "preview"
            all_post_items = [analyzer._post_to_item(post) for post in posts]
            relevant_posts = self._find_relevant_posts(all_post_items, question)
            if not relevant_posts:
                relevant_posts = material["hot_posts"][: min(self.max_context_posts, 10)]

        context = self._build_context(material, report, relevant_posts)
        prompt = self._build_prompt(question, context)

        if getattr(config, "ENABLE_AGENT_QA_LLM", False):
            try:
                answer_text = self._generate_answer(prompt)
                source = "qwen"
                error_message = None
            except Exception as exc:
                answer_text = self._build_fallback_answer(question, context, exc)
                source = "rules_fallback"
                error_message = f"{type(exc).__name__}: {exc}"
                print(f"智能体问答：本地大模型回答失败，已降级: {error_message}")
        else:
            answer_text = self._build_rule_answer(question, context)
            source = "rules"
            error_message = None

        return {
            "question": question,
            "answer": answer_text,
            "period": period,
            "period_label": PERIOD_LABELS.get(period, period),
            "source": source,
            "error_message": error_message,
            "evidence_posts": relevant_posts[:8],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _generate_answer(self, prompt):
        self._ensure_generator()
        return self.generator.generate_text(
            prompt,
            system_content=(
                "你是校园墙舆情问答智能体。你必须基于给定数据库上下文回答，"
                "不能编造未提供的事实。证据不足时要明确说明。"
            ),
            max_new_tokens=self.max_new_tokens,
        ).strip()

    def _ensure_generator(self):
        if self.generator is not None:
            return
        backend = getattr(config, "LLM_CLASSIFIER_BACKEND", "base").lower()
        self._check_dependencies(backend)
        if backend == "lora":
            self.generator = QwenLoraPostClassifier(self.app)
        else:
            self.generator = QwenPostClassifier(self.app)

    @staticmethod
    def _check_dependencies(backend):
        dependencies = LORA_BACKEND_DEPENDENCIES if backend == "lora" else BASE_BACKEND_DEPENDENCIES
        missing = []
        for module_name in dependencies:
            try:
                __import__(module_name)
            except ModuleNotFoundError:
                missing.append(module_name)
        if missing:
            install_hint = (
                ".\\.venv\\Scripts\\python.exe -m pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu124\n"
                ".\\.venv\\Scripts\\python.exe -m pip install -r model_Qwen3\\requirements-qwen-light.txt"
            )
            if backend == "lora":
                install_hint += "\n.\\.venv\\Scripts\\python.exe -m pip install -r model_Qwen3\\lora\\requirements-qwen-lora.txt"
            raise RuntimeError(
                "本地大模型依赖缺失: "
                + ", ".join(missing)
                + "。请在运行 Flask 的同一个 Python 环境中安装依赖，例如：\n"
                + install_hint
            )

    def _build_context(self, material, report, relevant_posts):
        metrics = material["metrics"]
        return {
            "period_label": material["period_label"],
            "post_count": material["post_count"],
            "metrics": metrics,
            "overview": report.get("overview", {}),
            "hot_topics": report.get("hot_topics", [])[:5],
            "risks": report.get("risks", [])[:4],
            "suggestions": report.get("suggestions", [])[:5],
            "relevant_posts": relevant_posts[: self.max_context_posts],
            "top_sentiments": self._top_items(metrics.get("sentiment_distribution", {}), 5),
            "top_categories": self._top_items(metrics.get("category_distribution", {}), 5),
            "top_subcategories": self._top_items(metrics.get("subcategory_distribution", {}), 8),
        }

    @staticmethod
    def _build_prompt(question, context):
        return (
            "请回答用户关于校园墙舆情数据的问题。\n"
            "回答要求：\n"
            "1. 只基于 CONTEXT_DATA 中的信息回答，不要编造数量、事件、学校部门或处理结果。\n"
            "2. 如果相关帖子不足，先说明证据不足，再给出谨慎判断。\n"
            "3. 尽量引用 relevant_posts 中的帖子摘要作为依据。\n"
            "4. 用中文回答，结构清晰，长度控制在 3 到 6 个要点或短段落。\n\n"
            f"用户问题：{question}\n\n"
            f"CONTEXT_DATA = {json.dumps(context, ensure_ascii=False)}"
        )

    def _find_relevant_posts(self, posts, question):
        keywords = self._extract_keywords(question)
        scored = []
        for post in posts:
            searchable = " ".join(
                str(post.get(key, ""))
                for key in (
                    "snippet",
                    "raw_category",
                    "llm_category_1",
                    "llm_category_2",
                    "display_category_1",
                    "display_category_2",
                    "sentiment",
                )
            )
            score = 0
            for keyword in keywords:
                if keyword and keyword in searchable:
                    score += len(keyword) * 4
            if score:
                score += min(int(post.get("heat", 0) or 0), 1000) / 1000
                scored.append((score, post))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [post for _score, post in scored[: self.max_context_posts]]

    @staticmethod
    def _extract_keywords(question):
        question = re.sub(r"\s+", " ", question).strip()
        keywords = []

        try:
            import jieba

            candidates = jieba.cut(question)
        except Exception:
            candidates = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", question)

        for word in candidates:
            word = str(word).strip()
            if len(word) < 2 or word in QUESTION_STOP_WORDS:
                continue
            if re.fullmatch(r"[\W_]+", word):
                continue
            keywords.append(word)

        if not keywords:
            keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", question)
        return list(dict.fromkeys(keywords))[:12]

    def _clean_question(self, question):
        question = re.sub(r"\s+", " ", str(question or "")).strip()
        if not question:
            raise ValueError("问题不能为空。")
        if len(question) > self.max_question_length:
            question = question[: self.max_question_length]
        return question

    @staticmethod
    def _top_items(distribution, limit):
        counter = Counter(distribution or {})
        return [
            {"label": str(label), "count": int(count)}
            for label, count in counter.most_common(limit)
        ]

    @staticmethod
    def _build_fallback_answer(question, context, exc):
        overview = context.get("overview", {})
        topic_names = [
            item.get("topic")
            for item in context.get("hot_topics", [])
            if item.get("topic")
        ]
        evidence = [
            item.get("snippet")
            for item in context.get("relevant_posts", [])[:3]
            if item.get("snippet")
        ]
        parts = [
            f"本地大模型暂时没有成功返回结果，当前先给出规则分析。问题是：{question}",
            overview.get("summary") or f"{context.get('period_label', '')}共覆盖 {context.get('post_count', 0)} 条帖子。",
        ]
        if topic_names:
            parts.append("相关热点主要包括：" + "、".join(topic_names[:5]) + "。")
        if evidence:
            parts.append("可参考的帖子摘要：" + "；".join(evidence) + "。")
        parts.append(f"降级原因：{type(exc).__name__}: {exc}。")
        return "\n".join(parts)

    @staticmethod
    def _build_rule_answer(question, context):
        overview = context.get("overview", {})
        hot_topics = context.get("hot_topics", [])
        risks = context.get("risks", [])
        evidence_posts = context.get("relevant_posts", [])
        top_categories = context.get("top_categories", [])
        top_sentiments = context.get("top_sentiments", [])

        parts = [
            f"根据{context.get('period_label', '当前周期')}数据库中的 {context.get('post_count', 0)} 条帖子，当前使用规则检索模式回答：{question}",
        ]
        if overview.get("summary"):
            parts.append(overview["summary"])

        if top_categories:
            category_text = "、".join(
                f"{item['label']}({item['count']})"
                for item in top_categories[:5]
            )
            parts.append(f"主要分类集中在：{category_text}。")

        if top_sentiments:
            sentiment_text = "、".join(
                f"{item['label']}({item['count']})"
                for item in top_sentiments[:3]
            )
            parts.append(f"情绪分布为：{sentiment_text}。")

        if hot_topics:
            topic_text = "、".join(
                f"{item.get('topic', '未命名')}({item.get('count', 0)})"
                for item in hot_topics[:5]
            )
            parts.append(f"热点话题包括：{topic_text}。")

        if risks:
            risk = risks[0]
            parts.append(
                f"当前首要风险提示：{risk.get('title', '暂无明显风险')}，"
                f"等级为{risk.get('level', '低')}。{risk.get('suggestion', '')}"
            )

        evidence = [post.get("snippet") for post in evidence_posts[:3] if post.get("snippet")]
        if evidence:
            parts.append("参考帖子：" + "；".join(evidence) + "。")
        return "\n".join(parts)
