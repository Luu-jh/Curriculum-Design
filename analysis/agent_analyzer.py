import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import config
from analysis.llm_qwen_classifier import QWEN_MODEL_LOCK, configure_qwen_runtime
from core.models import AgentReport, Post, db


PERIOD_LABELS = {
    "today": "今日",
    "week": "近7天",
    "month": "近30天",
}

PERIOD_DAYS = {
    "today": 1,
    "week": 7,
    "month": 30,
}

UNCLASSIFIED_LABEL = "未分类"


class AgentAnalyzer:
    """Generate campus-wall public-opinion reports from stored post data."""

    def __init__(self, app, llm_generator=None, use_llm=None):
        self.app = app
        self.llm_generator = llm_generator
        self.use_llm = (
            getattr(config, "ENABLE_AGENT_LLM_ANALYSIS", True)
            if use_llm is None
            else use_llm
        )
        self.max_posts = getattr(config, "AGENT_ANALYSIS_MAX_POSTS", 80)
        self.max_new_tokens = getattr(config, "AGENT_ANALYSIS_MAX_NEW_TOKENS", 700)
        self.model_dir = Path(getattr(config, "QWEN_CLASSIFIER_MODEL_DIR", ""))
        self.model_id = getattr(config, "QWEN_CLASSIFIER_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        self.allow_remote_download = getattr(config, "QWEN_ALLOW_REMOTE_MODEL_DOWNLOAD", False)
        self.model = None
        self.tokenizer = None

    def run(self, periods=None):
        results = []
        for period in periods or getattr(config, "AGENT_ANALYSIS_PERIODS", ("today", "week", "month")):
            results.append(self.run_period(period))
        return results

    def run_period(self, period):
        period = self.normalize_period(period)
        with self.app.app_context():
            posts = self._load_posts(period)
            material = self._build_material(posts, period)
            report = self._build_rule_report(material)
            status = "success"
            error_message = None

            if self.use_llm and posts:
                try:
                    llm_report = self._generate_llm_report(material)
                    report = self._merge_llm_report(report, llm_report)
                    report["source"] = "llm"
                except Exception as exc:
                    status = "fallback"
                    error_message = f"{type(exc).__name__}: {exc}"
                    report["source"] = "rules"
                    report["llm_error"] = error_message
                    print(f"智能体分析：LLM生成失败，已降级为规则报告: {error_message}")
            else:
                report["source"] = "rules"

            report["status"] = status
            report["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            record = AgentReport(
                period=period,
                post_count=material["post_count"],
                report_json=json.dumps(report, ensure_ascii=False),
                status=status,
                error_message=error_message,
            )
            db.session.add(record)
            db.session.commit()
            print(f"智能体分析：{PERIOD_LABELS[period]}报告已生成，覆盖 {material['post_count']} 条帖子。")
            return self._record_to_payload(record)

    def build_preview_report(self, period):
        period = self.normalize_period(period)
        with self.app.app_context():
            posts = self._load_posts(period)
            material = self._build_material(posts, period)
            report = self._build_rule_report(material)
            report["source"] = "rules"
            report["status"] = "preview"
            report["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report["message"] = "暂无后台报告，当前展示基于数据库即时生成的规则预览。"
            return report

    def get_current_data_state(self, period):
        period = self.normalize_period(period)
        with self.app.app_context():
            posts = self._load_posts(period)
            return self._build_data_state([self._post_to_item(post) for post in posts])

    @classmethod
    def get_latest_report(cls, period):
        period = cls.normalize_period(period)
        record = (
            AgentReport.query.filter(AgentReport.period == period)
            .order_by(AgentReport.created_at.desc())
            .first()
        )
        if record is None:
            return None
        return cls._record_to_payload(record)

    @staticmethod
    def is_report_stale(report, current_state):
        if not report:
            return True
        report_state = report.get("data_state") or {}
        if not report_state:
            return True
        for key in ("post_count", "latest_post_time", "total_heat"):
            if report_state.get(key) != current_state.get(key):
                return True
        return False

    @staticmethod
    def normalize_period(period):
        period = (period or "week").strip().lower()
        return period if period in PERIOD_LABELS else "week"

    @staticmethod
    def _record_to_payload(record):
        try:
            report = json.loads(record.report_json)
        except json.JSONDecodeError:
            report = {"overview": {"summary": "报告内容解析失败。"}}

        report["report_id"] = record.id
        report["period"] = record.period
        report["period_label"] = PERIOD_LABELS.get(record.period, record.period)
        report["post_count"] = record.post_count or report.get("post_count", 0)
        report["status"] = record.status or report.get("status", "success")
        report["created_at"] = (
            record.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if record.created_at
            else report.get("generated_at")
        )
        if record.error_message:
            report["error_message"] = record.error_message
        return report

    def _load_posts(self, period):
        start_time = self._period_start(period)
        return (
            Post.query.filter(Post.create_time >= start_time)
            .order_by(Post.create_time.desc())
            .all()
        )

    @staticmethod
    def _period_start(period):
        now = datetime.now()
        if period == "today":
            return datetime(now.year, now.month, now.day)
        return now - timedelta(days=PERIOD_DAYS.get(period, 7) - 1)

    def _build_material(self, posts, period):
        post_items = [self._post_to_item(post) for post in posts]
        selected_posts = self._select_representative_posts(post_items)
        sentiment_counts = Counter(item["sentiment"] for item in post_items)
        category_1_counts = Counter(item["display_category_1"] for item in post_items)
        category_2_counts = Counter(item["display_category_2"] for item in post_items)
        data_state = self._build_data_state(post_items)

        return {
            "period": period,
            "period_label": PERIOD_LABELS[period],
            "post_count": len(post_items),
            "data_state": data_state,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "sentiment_distribution": dict(sentiment_counts),
                "category_distribution": dict(category_1_counts),
                "subcategory_distribution": dict(category_2_counts),
                "average_heat": self._average_heat(post_items),
            },
            "hot_posts": sorted(post_items, key=lambda item: item["heat"], reverse=True)[:10],
            "negative_posts": [
                item
                for item in sorted(post_items, key=lambda item: item["heat"], reverse=True)
                if item["sentiment"] == "负向"
            ][:10],
            "topic_candidates": self._build_topic_candidates(post_items),
            "representative_posts": selected_posts,
        }

    def _post_to_item(self, post):
        heat = (post.comments or 0) * 5 + (post.likes or 0) * 3 + (post.views or 0)
        raw_category = post.category or "未分类"
        llm_category_1 = self._clean_label(post.llm_category_1)
        llm_category_2 = self._clean_label(post.llm_category_2)
        return {
            "id": post.id,
            "snippet": self._snippet(post.content, 90),
            "raw_category": raw_category,
            "llm_category_1": llm_category_1,
            "llm_category_2": llm_category_2,
            "display_category_1": llm_category_1,
            "display_category_2": llm_category_2,
            "sentiment": post.sentiment or "未分类",
            "views": post.views or 0,
            "comments": post.comments or 0,
            "likes": post.likes or 0,
            "heat": heat,
            "create_time": post.create_time.strftime("%Y-%m-%d %H:%M:%S") if post.create_time else "",
        }

    def _select_representative_posts(self, items):
        selected = []
        seen_ids = set()

        def add_many(candidates):
            for item in candidates:
                if item["id"] in seen_ids or len(selected) >= self.max_posts:
                    continue
                selected.append(item)
                seen_ids.add(item["id"])

        by_heat = sorted(items, key=lambda item: item["heat"], reverse=True)
        add_many(by_heat[:20])
        add_many([item for item in by_heat if item["sentiment"] == "负向"][:20])

        by_category = defaultdict(list)
        for item in by_heat:
            by_category[item["display_category_1"]].append(item)
        for category_items in by_category.values():
            add_many(category_items[:3])

        add_many(items[: self.max_posts])
        return selected[: self.max_posts]

    def _build_topic_candidates(self, items):
        grouped = defaultdict(list)
        for item in items:
            category_1 = item["display_category_1"]
            category_2 = item["display_category_2"]
            if category_1 == UNCLASSIFIED_LABEL and category_2 == UNCLASSIFIED_LABEL:
                key = UNCLASSIFIED_LABEL
                topic_name = UNCLASSIFIED_LABEL
            elif category_2 == "未分类":
                key = category_1
                topic_name = category_1
            else:
                key = f"{category_1}/{category_2}"
                topic_name = category_2
            grouped[key].append((topic_name, item))

        candidates = []
        for key, values in grouped.items():
            sorted_values = sorted(values, key=lambda pair: pair[1]["heat"], reverse=True)
            candidates.append(
                {
                    "key": key,
                    "topic": sorted_values[0][0],
                    "count": len(sorted_values),
                    "category": key,
                    "evidence": [pair[1]["snippet"] for pair in sorted_values[:3]],
                    "avg_heat": round(
                        sum(pair[1]["heat"] for pair in sorted_values) / max(len(sorted_values), 1),
                        1,
                    ),
                }
            )
        return sorted(candidates, key=lambda item: (item["count"], item["avg_heat"]), reverse=True)[:8]

    def _build_rule_report(self, material):
        post_count = material["post_count"]
        metrics = material["metrics"]
        sentiment_counts = metrics["sentiment_distribution"]
        category_counts = metrics["category_distribution"]
        top_categories = [
            label for label, _count in Counter(category_counts).most_common(3)
            if label and label != "未分类"
        ]

        negative_count = sentiment_counts.get("负向", 0)
        positive_count = sentiment_counts.get("正向", 0)
        negative_ratio = negative_count / post_count if post_count else 0
        positive_ratio = positive_count / post_count if post_count else 0
        risk_level = self._risk_level(negative_ratio, material)
        tone = self._tone_label(positive_ratio, negative_ratio)

        if post_count == 0:
            summary = f"{material['period_label']}暂无可分析帖子，建议等待爬虫采集后再查看。"
        else:
            topic_text = "、".join(top_categories) if top_categories else "未分类内容"
            summary = (
                f"{material['period_label']}共覆盖 {post_count} 条帖子，整体情绪{tone}，"
                f"主要集中在 {topic_text} 等方向，当前综合风险等级为{risk_level}。"
            )

        hot_topics = [
            {
                "topic": topic["topic"],
                "category": topic["category"],
                "count": topic["count"],
                "reason": f"该主题在{material['period_label']}出现 {topic['count']} 次，平均热度 {topic['avg_heat']}。",
                "evidence": topic["evidence"],
            }
            for topic in material["topic_candidates"][:5]
        ]

        return {
            "period": material["period"],
            "period_label": material["period_label"],
            "post_count": post_count,
            "overview": {
                "summary": summary,
                "sentiment_tone": tone,
                "risk_level": risk_level,
            },
            "metrics": metrics,
            "hot_topics": hot_topics,
            "risks": self._build_risks(material, negative_ratio),
            "suggestions": self._build_suggestions(material, negative_ratio),
            "representative_posts": material["hot_posts"][:6],
            "data_state": material["data_state"],
        }

    @staticmethod
    def _build_data_state(post_items):
        post_times = [item["create_time"] for item in post_items if item["create_time"]]
        latest_post_time = max(post_times) if post_times else ""
        return {
            "post_count": len(post_items),
            "latest_post_time": latest_post_time,
            "total_heat": sum(int(item.get("heat", 0) or 0) for item in post_items),
        }

    def _build_risks(self, material, negative_ratio):
        risks = []
        subcategory_counts = material["metrics"]["subcategory_distribution"]
        category_counts = material["metrics"]["category_distribution"]

        if negative_ratio >= 0.25:
            risks.append(
                {
                    "level": "高" if negative_ratio >= 0.4 else "中",
                    "title": "负向情绪占比较高",
                    "description": f"负向帖子占比约 {negative_ratio:.0%}，需要关注集中吐槽点和高互动负面内容。",
                    "suggestion": "优先查看负向高热帖子，确认是否涉及教学、宿舍、后勤或考试安排等具体问题。",
                    "evidence": [item["snippet"] for item in material["negative_posts"][:3]],
                }
            )

        if subcategory_counts.get("代办", 0) >= 3:
            risks.append(
                {
                    "level": "中",
                    "title": "代办代课类需求集中",
                    "description": "求代办、代课、代跑腿等帖子数量较多，可能涉及课堂纪律或校园管理风险。",
                    "suggestion": "建议结合课程时段、体测安排等信息，定向加强提醒和规则说明。",
                    "evidence": self._topic_evidence(material, "代办"),
                }
            )

        if category_counts.get("交易", 0) >= 5:
            risks.append(
                {
                    "level": "中",
                    "title": "交易信息需要防范纠纷",
                    "description": "二手交易、求购、转让类帖子较活跃，可能出现价格争议、虚假信息或私下交易纠纷。",
                    "suggestion": "建议在校园墙运营规则中补充交易提醒，引导保留凭证并避免敏感交易。",
                    "evidence": self._topic_evidence(material, "交易"),
                }
            )

        if not risks:
            risks.append(
                {
                    "level": "低",
                    "title": "暂无明显集中风险",
                    "description": "当前数据中未发现高占比负向情绪或明显异常主题。",
                    "suggestion": "保持常规观察，重点留意后续高互动帖子和突增主题。",
                    "evidence": [item["snippet"] for item in material["hot_posts"][:2]],
                }
            )
        return risks[:4]

    def _build_suggestions(self, material, negative_ratio):
        post_count = material["post_count"]
        subcategory_counts = material["metrics"]["subcategory_distribution"]
        hot_posts = material["hot_posts"]
        hot_post_ids = [
            str(item["id"])
            for item in hot_posts[:3]
            if item.get("id") is not None
        ]
        hot_post_text = "、".join(hot_post_ids) if hot_post_ids else "当前高热帖子"

        if post_count == 0:
            return [
                "第1步 立即处置：暂无可分析帖子，先确认爬虫接口、Cookie 和数据库写入是否正常。",
                "第2步 后续跟踪：等待采集到帖子后再刷新智能体分析，避免基于空数据做判断。",
            ]

        suggestions = []
        step = 1
        if negative_ratio >= 0.25:
            suggestions.append(f"第{step}步 立即处置：将负向高热帖子列为优先巡检对象，提取具体诉求、责任场景和可能涉及部门。")
            step += 1

        suggestions.extend(
            [
                f"第{step}步 重点核查：先查看帖子 {hot_post_text}，确认是否存在具体诉求、争议点或需要人工回复的内容。",
                f"第{step + 1}步 今日跟踪：每 2-3 小时复看高互动帖子，结合评论数、点赞数和浏览量判断扩散程度。",
                f"第{step + 2}步 持续校验：抽样复核一级分类和二级分类，避免模型分类偏差影响风险研判。",
            ]
        )

        if subcategory_counts.get("问题咨询", 0) >= 5:
            suggestions.append("专项处理：把高频咨询问题整理成公告或 FAQ，减少重复提问和信息不对称。")
        if subcategory_counts.get("代办", 0) >= 3:
            suggestions.append("专项处理：对代课、代跑、代取等高频内容增加规则提醒，降低违规和纠纷风险。")
        return suggestions[:5]

    def _topic_evidence(self, material, keyword):
        evidence = []
        for topic in material["topic_candidates"]:
            if keyword in topic["key"] or keyword in topic["topic"]:
                evidence.extend(topic["evidence"])
        return evidence[:3]

    def _generate_llm_report(self, material):
        prompt = self._build_llm_prompt(material)
        system_content = "你是校园舆情分析智能体。必须只基于给定数据输出严格JSON，不要解释。"
        if self.llm_generator is not None:
            answer = self.llm_generator.generate_text(
                prompt,
                system_content=system_content,
                max_new_tokens=self.max_new_tokens,
            )
        else:
            answer = self._generate_with_local_model(prompt, system_content)
        try:
            data = self._extract_json(answer)
        except (json.JSONDecodeError, ValueError) as exc:
            return self._build_text_llm_report(answer, exc)
        if not isinstance(data, dict):
            raise ValueError("模型输出不是JSON对象")
        return data

    def _build_llm_prompt(self, material):
        compact_material = {
            "period_label": material["period_label"],
            "post_count": material["post_count"],
            "metrics": material["metrics"],
            "topic_candidates": material["topic_candidates"][:5],
            "hot_posts": material["hot_posts"][:5],
            "negative_posts": material["negative_posts"][:5],
            "representative_posts": material["representative_posts"][:12],
        }
        return (
            "请生成校园墙舆情智能体研判报告。\n"
            "硬性要求：\n"
            "1. 只能基于 INPUT_DATA 中的事实，不要编造学校、部门、事件和数量。\n"
            "2. summary 用 1 到 2 句话概括，必须包含帖子数量、主要类别和风险等级。\n"
            "3. sentiment_tone 只能是：平稳、略有波动、偏负向、偏正向。\n"
            "4. risk_level 只能是：低、中、高。\n"
            "5. suggestions 输出 3 到 5 条可执行处置建议，每条必须包含明确动作、处理对象或跟踪频率。\n"
            "7. suggestions 建议使用“第1步 立即处置：...”“第2步 今日跟踪：...”“专项处理：...”这类格式。\n"
            "6. 只输出严格JSON，不要输出Markdown或解释。\n\n"
            "输出JSON结构："
            "{\"overview\":{\"summary\":\"\",\"sentiment_tone\":\"平稳\",\"risk_level\":\"低\"},"
            "\"suggestions\":[\"\",\"\"]}\n\n"
            f"INPUT_DATA = {json.dumps(compact_material, ensure_ascii=False)}"
        )

    def _generate_with_local_model(self, prompt, system_content):
        self._load_local_model()

        import torch

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _load_local_model(self):
        if self.model is not None and self.tokenizer is not None:
            return
        with QWEN_MODEL_LOCK:
            if self.model is not None and self.tokenizer is not None:
                return
            model_name_or_path = self._model_name_or_path()
            local_files_only = not self.allow_remote_download
            configure_qwen_runtime()

            print(f"智能体分析：正在加载模型 {model_name_or_path} ...")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            model_kwargs = dict(
                local_files_only=local_files_only,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
            self.model.eval()
            print(f"智能体分析：模型加载完成，CUDA可用={torch.cuda.is_available()}。")

    def _model_name_or_path(self):
        if self.model_dir.exists():
            return str(self.model_dir)
        if self.allow_remote_download:
            return self.model_id
        raise FileNotFoundError(
            "智能体分析轻量模型尚未放到本地目录："
            f"{self.model_dir}。请下载 {self.model_id} 到该目录，"
            "或临时设置环境变量 QWEN_ALLOW_REMOTE_MODEL_DOWNLOAD=1 后再启动项目。"
        )

    @staticmethod
    def _merge_llm_report(rule_report, llm_report):
        merged = dict(rule_report)
        overview = llm_report.get("overview")
        if isinstance(overview, dict):
            for key, value in overview.items():
                if key not in {"summary", "sentiment_tone", "risk_level"} or not value:
                    continue
                if key == "summary" and not AgentAnalyzer._is_useful_llm_summary(value):
                    merged["llm_summary"] = str(value).strip()
                    continue
                merged["overview"][key] = value
        elif isinstance(overview, str) and overview.strip():
            if AgentAnalyzer._is_useful_llm_summary(overview):
                merged["overview"]["summary"] = overview.strip()
            else:
                merged["llm_summary"] = overview.strip()

        for key in ("hot_topics", "risks", "suggestions"):
            value = llm_report.get(key)
            if isinstance(value, list) and value:
                merged[key] = value[:6] if key != "suggestions" else value[:5]
        return merged

    @staticmethod
    def _is_useful_llm_summary(summary):
        summary = str(summary or "").strip()
        return len(summary) >= 24 and ("帖子" in summary or "风险" in summary or "情绪" in summary)

    @staticmethod
    def _extract_json(answer):
        try:
            return json.loads(answer)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", answer, flags=re.S)
        if not match:
            raise ValueError(f"模型没有返回JSON: {answer[:120]}")
        return json.loads(match.group(0))

    @staticmethod
    def _build_text_llm_report(answer, exc):
        summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', answer or "")
        if summary_match:
            summary = summary_match.group(1).strip()
        else:
            summary = re.sub(r"```(?:json)?|```", "", str(answer or "")).strip()
            summary = re.sub(r"\s+", " ", summary)
            summary = summary[:260]
        if not summary:
            summary = "大模型已返回内容，但格式不完整，当前保留规则报告作为主体研判。"
        return {
            "overview": {"summary": summary},
            "llm_parse_warning": f"{type(exc).__name__}: {exc}",
        }

    @staticmethod
    def _clean_label(label):
        label = (label or "").strip()
        return label if label else UNCLASSIFIED_LABEL

    @staticmethod
    def _snippet(text, limit):
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text[:limit] + "..." if len(text) > limit else text

    @staticmethod
    def _average_heat(items):
        if not items:
            return 0
        return round(sum(item["heat"] for item in items) / len(items), 1)

    @staticmethod
    def _risk_level(negative_ratio, material):
        subcategory_counts = material["metrics"]["subcategory_distribution"]
        if negative_ratio >= 0.4:
            return "高"
        if negative_ratio >= 0.25 or subcategory_counts.get("代办", 0) >= 8:
            return "中"
        return "低"

    @staticmethod
    def _tone_label(positive_ratio, negative_ratio):
        if negative_ratio >= 0.35:
            return "偏负向"
        if positive_ratio >= 0.45 and positive_ratio > negative_ratio:
            return "偏正向"
        if negative_ratio >= 0.2:
            return "略有波动"
        return "平稳"
