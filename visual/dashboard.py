from flask import Blueprint, Response, current_app, jsonify, render_template, request

import config
from analysis.agent_analyzer import AgentAnalyzer
from analysis.agent_qa import AgentQuestionAnswerer
from visual.charts import DashboardData


dashboard_bp = Blueprint("dashboard", __name__)
_agent_question_answerer = None


@dashboard_bp.route("/")
def index():
    """Render dashboard page with initial hot-post data."""
    hot_posts = DashboardData.get_top_hot_posts(10)
    return render_template("index.html", hot_posts=hot_posts)


@dashboard_bp.route("/api/chart/line")
def api_line_chart():
    period = request.args.get("period", "week")
    chart_json = DashboardData.get_line_chart(period)
    return Response(chart_json, content_type="application/json")


@dashboard_bp.route("/api/chart/pie")
def api_pie_chart():
    chart_json = DashboardData.get_pie_chart()
    return Response(chart_json, content_type="application/json")


@dashboard_bp.route("/api/chart/category2")
def api_category2_chart():
    category_1 = request.args.get("category1")
    chart_json = DashboardData.get_category_2_bar_chart(category_1)
    return Response(chart_json, content_type="application/json")


@dashboard_bp.route("/api/chart/wordcloud")
def api_wordcloud_chart():
    chart_json = DashboardData.get_wordcloud_chart()
    return Response(chart_json, content_type="application/json")


@dashboard_bp.route("/api/agent/report")
def api_agent_report():
    period = request.args.get("period", "week")
    app = current_app._get_current_object()
    analyzer = AgentAnalyzer(app, use_llm=False)
    report = AgentAnalyzer.get_latest_report(period)
    current_state = analyzer.get_current_data_state(period)
    if report is None or AgentAnalyzer.is_report_stale(report, current_state):
        report = analyzer.build_preview_report(period)
        if current_state.get("post_count", 0) > 0:
            report["message"] = "后台报告与当前帖子数据不同步，当前展示基于最新数据库即时生成的规则预览。"
            report["sync_status"] = "live_preview"
    return jsonify(report)


@dashboard_bp.route("/api/agent/refresh", methods=["POST"])
def api_agent_refresh():
    period = request.args.get("period", "week")
    use_llm = request.args.get("llm", "0") == "1"
    app = current_app._get_current_object()
    try:
        report = AgentAnalyzer(app, use_llm=use_llm).run_period(period)
        return jsonify(report)
    except Exception as exc:
        try:
            report = AgentAnalyzer(app, use_llm=False).run_period(period)
            report["status"] = "fallback"
            report["message"] = f"大模型分析失败，已改用规则分析刷新：{exc}"
            return jsonify(report)
        except Exception as fallback_exc:
            return jsonify({"status": "failed", "message": f"智能体分析刷新失败：{fallback_exc}"}), 500


@dashboard_bp.route("/api/agent/chat", methods=["POST"])
def api_agent_chat():
    if not getattr(config, "ENABLE_AGENT_QA", True):
        return jsonify({"status": "failed", "message": "智能体问答功能未启用。"}), 403

    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")
    period = payload.get("period", "week")

    global _agent_question_answerer
    if _agent_question_answerer is None:
        app = current_app._get_current_object()
        _agent_question_answerer = AgentQuestionAnswerer(app)

    try:
        result = _agent_question_answerer.answer(question, period)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"status": "failed", "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "failed", "message": f"智能体问答失败：{exc}"}), 500
