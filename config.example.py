# config.example.py
# Copy this file to config.py and fill in your local database and crawler settings.
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# MySQL
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_USER = "root"
DB_PASS = "YOUR_MYSQL_PASSWORD"
DB_NAME = "campus_wall"

SQLALCHEMY_DATABASE_URI = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)
SQLALCHEMY_TRACK_MODIFICATIONS = False


# Campus wall crawler
API_URL = "https://example.com/api/article/lists"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Cookie": "YOUR_SESSION_COOKIE",
    "Content-Type": "application/json",
    "Referer": "YOUR_REFERER",
    "xweb_xhr": "1",
    "accept": "*/*",
}

REFRESH_INTERVAL = 60
CRAWL_PAGE_COUNT = 10
STARTUP_BACKFILL_PAGE_COUNT = 10
DAYS_TO_KEEP = 60
CRAWLER_VERIFY_SSL = True


# Sentiment analysis
ENABLE_SENTIMENT_ANALYSIS = True
SENTIMENT_POSITIVE_THRESHOLD = 0.75
SENTIMENT_NEGATIVE_THRESHOLD = 0.35
SENTIMENT_BATCH_SIZE = 100


# Qwen classifier and agent model
QWEN_CLASSIFIER_MODEL_ID = os.environ.get(
    "QWEN_CLASSIFIER_MODEL_ID",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
QWEN_ALLOW_REMOTE_MODEL_DOWNLOAD = os.environ.get(
    "QWEN_ALLOW_REMOTE_MODEL_DOWNLOAD",
    "0",
) == "1"

ENABLE_LLM_CLASSIFICATION = True
LLM_CLASSIFIER_BACKEND = "lora"

QWEN_CLASSIFIER_LOCAL_MODEL_DIR = os.environ.get(
    "QWEN_CLASSIFIER_LOCAL_MODEL_DIR",
    os.path.join(BASE_DIR, "model_Qwen3", "Qwen2.5-0.5B-Instruct"),
)
QWEN_CLASSIFIER_MODEL_DIR = os.environ.get(
    "QWEN_CLASSIFIER_MODEL_DIR",
    QWEN_CLASSIFIER_LOCAL_MODEL_DIR,
)
QWEN_MODEL_READY = os.path.isdir(QWEN_CLASSIFIER_MODEL_DIR) or QWEN_ALLOW_REMOTE_MODEL_DOWNLOAD

LLM_CLASSIFICATION_BATCH_SIZE = 50
LLM_CLASSIFICATION_MAX_TEXT_LENGTH = 300
LLM_CLASSIFICATION_MAX_NEW_TOKENS = 80


# Agent analysis
ENABLE_AGENT_ANALYSIS = True
ENABLE_AGENT_LLM_ANALYSIS = False
AGENT_ANALYSIS_PERIODS = ("today", "week", "month")
AGENT_ANALYSIS_INTERVAL = 600
AGENT_ANALYSIS_MAX_POSTS = 30
AGENT_ANALYSIS_MAX_NEW_TOKENS = 360

ENABLE_AGENT_QA = True
ENABLE_AGENT_QA_LLM = QWEN_MODEL_READY
AGENT_QA_MAX_CONTEXT_POSTS = 12
AGENT_QA_MAX_NEW_TOKENS = 420
AGENT_QA_MAX_QUESTION_LENGTH = 300


# Flask
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
DEBUG_MODE = True
