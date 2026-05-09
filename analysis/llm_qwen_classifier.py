# analysis/llm_qwen_classifier.py
import json
import os
import re
import threading
from collections import Counter
from pathlib import Path

import config
from core.models import Post, db


LABEL_SCHEMA = {
    "求助": ["代办", "资源获取", "问题咨询"],
    "交易": ["出售", "求购", "转让"],
    "找人找物": ["找人", "找物", "拿错归还"],
    "吐槽": ["吐槽环境", "吐槽生活"],
    "日常": ["生活分享", "交友扩列", "日常闲聊"],
    "其他": ["其他"],
}

QWEN_MODEL_LOCK = threading.Lock()


def configure_qwen_runtime():
    """Keep local model loading quiet and safe in hidden/background Flask processes."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


class QwenPostClassifier:
    """Use local Qwen to classify posts into llm_category_1/2."""

    def __init__(self, app):
        self.app = app
        self.model_dir = Path(config.QWEN_CLASSIFIER_MODEL_DIR)
        self.model_id = getattr(config, "QWEN_CLASSIFIER_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        self.allow_remote_download = getattr(config, "QWEN_ALLOW_REMOTE_MODEL_DOWNLOAD", False)
        self.batch_size = config.LLM_CLASSIFICATION_BATCH_SIZE
        self.max_text_length = config.LLM_CLASSIFICATION_MAX_TEXT_LENGTH
        self.max_new_tokens = config.LLM_CLASSIFICATION_MAX_NEW_TOKENS
        self.model = None
        self.tokenizer = None
        self.backend_name = "base"

    def run_classification(self, limit=None):
        """Classify pending posts and write labels back to MySQL."""
        with self.app.app_context():
            total_pending = self._pending_count()
            if total_pending == 0:
                print("Qwen分类：当前没有需要分类的数据。")
                return {"total": 0, "成功": 0, "失败": 0}

            target_total = min(total_pending, limit or self.batch_size)
            print(f"Qwen分类：发现 {total_pending} 条未分类数据，本轮处理 {target_total} 条。")

            self._load_model()
            posts = self._load_pending_posts(target_total)
            stats = Counter()

            for post in posts:
                try:
                    category_1, category_2 = self.classify(post.content, post.category)
                    post.llm_category_1 = category_1
                    post.llm_category_2 = category_2
                    stats["成功"] += 1
                    print(f"  Qwen分类成功：{post.id} -> {category_1} / {category_2}")
                except Exception as exc:
                    post.llm_category_1 = "其他"
                    post.llm_category_2 = "其他"
                    stats["失败"] += 1
                    print(f"  Qwen分类失败：{post.id}: {exc}")

            try:
                db.session.commit()
            except Exception as exc:
                db.session.rollback()
                print(f"Qwen分类：写入数据库失败，已回滚本批 {len(posts)} 条: {exc}")
                return {"total": 0, "成功": 0, "失败": len(posts)}

            result = {"total": len(posts), "成功": stats["成功"], "失败": stats["失败"]}
            print(f"Qwen分类完成：处理 {result['total']} 条，成功 {result['成功']} 条，失败 {result['失败']} 条。")
            return result

    def classify(self, content, raw_category=None):
        self._load_model()
        prompt = self._build_prompt(content, raw_category)
        answer = self._generate(prompt)
        return self._parse_answer(answer)

    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return
        with QWEN_MODEL_LOCK:
            if self.model is not None and self.tokenizer is not None:
                return
            model_name_or_path = self._model_name_or_path()
            local_files_only = not self.allow_remote_download
            configure_qwen_runtime()

            print(f"Qwen分类（{self.backend_name}）：正在加载模型 {model_name_or_path} ...")
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
            print(f"Qwen分类（{self.backend_name}）：模型加载完成，CUDA可用={torch.cuda.is_available()}。")

    def _model_name_or_path(self):
        if self.model_dir.exists():
            return str(self.model_dir)
        if self.allow_remote_download:
            return self.model_id
        raise FileNotFoundError(
            "Qwen轻量模型尚未放到本地目录："
            f"{self.model_dir}。请下载 {self.model_id} 到该目录，"
            "或临时设置环境变量 QWEN_ALLOW_REMOTE_MODEL_DOWNLOAD=1 后再启动项目。"
        )

    def _generate(self, prompt):
        return self.generate_text(
            prompt,
            system_content="你是校园墙帖子分类器。必须只输出JSON，不要解释。",
            max_new_tokens=self.max_new_tokens,
        )

    def generate_text(self, prompt, system_content=None, max_new_tokens=None):
        self._load_model()

        import torch

        messages = [
            {"role": "system", "content": system_content or "你是校园墙帖子分类器。必须只输出JSON，不要解释。"},
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
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _build_prompt(self, content, raw_category=None):
        label_text = json.dumps(LABEL_SCHEMA, ensure_ascii=False, indent=2)
        post_text = self._clean_text(content)
        return (
            "请对校园墙帖子进行两级分类。\n"
            "要求：\n"
            "1. 一级分类必须从 LABEL_SCHEMA 的键中选择。\n"
            "2. 二级分类必须从该一级分类对应的列表中选择。\n"
            "3. 如果不确定，选择 {\"llm_category_1\": \"其他\", \"llm_category_2\": \"其他\"}。\n"
            "4. 只输出严格JSON，不要输出解释。\n\n"
            "重要判定规则：\n"
            "- 找代课、代取、代拿、代办、帮忙跑腿，归为 求助/代办。\n"
            "- 求资料、求链接、求文件、求课程资源，归为 求助/资源获取。\n"
            "- 问制度、问安排、问地点、问怎么办，归为 求助/问题咨询。\n"
            "- 出售、出闲置、卖东西，归为 交易/出售。\n"
            "- 求购、想买、收东西，归为 交易/求购。\n"
            "- 找人找物只用于寻人、寻物、拿错东西归还，不用于找人代办。\n\n"
            f"LABEL_SCHEMA = {label_text}\n\n"
            "示例：\n"
            "帖子：找4月30号下午前两节的代课，女生，有偿。\n"
            "输出：{\"llm_category_1\": \"求助\", \"llm_category_2\": \"代办\"}\n"
            "帖子：出一个九成新的电动车头盔，价格可刀。\n"
            "输出：{\"llm_category_1\": \"交易\", \"llm_category_2\": \"出售\"}\n\n"
            f"原始分类：{raw_category or '无'}\n"
            f"帖子内容：{post_text}\n\n"
            "输出格式：{\"llm_category_1\": \"求助\", \"llm_category_2\": \"问题咨询\"}"
        )

    def _parse_answer(self, answer):
        data = self._extract_json(answer)
        category_1 = str(data.get("llm_category_1", "")).strip()
        category_2 = str(data.get("llm_category_2", "")).strip()

        if category_1 not in LABEL_SCHEMA:
            return "其他", "其他"
        if category_2 not in LABEL_SCHEMA[category_1]:
            return category_1, LABEL_SCHEMA[category_1][0]
        return category_1, category_2

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

    def _pending_count(self):
        return Post.query.filter(
            (Post.llm_category_1.is_(None))
            | (Post.llm_category_1 == "")
            | (Post.llm_category_1 == "未分类")
        ).count()

    def _load_pending_posts(self, limit):
        return (
            Post.query.filter(
                (Post.llm_category_1.is_(None))
                | (Post.llm_category_1 == "")
                | (Post.llm_category_1 == "未分类")
            )
            .order_by(Post.create_time.desc())
            .limit(limit)
            .all()
        )

    def _clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r"\s+", " ", str(text)).strip()
        return text[: self.max_text_length]


