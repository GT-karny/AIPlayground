import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "not-needed")
MODEL_NAME = os.getenv("MODEL_NAME", "LiquidAI/LFM2.5-1.2B-JP")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "320"))
FACTUAL_MAX_TOKENS = int(os.getenv("FACTUAL_MAX_TOKENS", "700"))
ROUTER_MODEL_NAME = os.getenv("ROUTER_MODEL_NAME", MODEL_NAME)
ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "220"))
ROUTER_TEMPERATURE = float(os.getenv("ROUTER_TEMPERATURE", "0.0"))
ENABLE_SELF_CHECK = os.getenv("ENABLE_SELF_CHECK", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SELF_CHECK_MAX_RETRY = int(os.getenv("SELF_CHECK_MAX_RETRY", "1"))
SELF_CHECK_MAX_TOKENS = int(os.getenv("SELF_CHECK_MAX_TOKENS", "320"))
AUTO_RESEARCH_ENABLED = os.getenv("AUTO_RESEARCH_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
AUTO_RESEARCH_MAX_QUERIES = int(os.getenv("AUTO_RESEARCH_MAX_QUERIES", "3"))
AUTO_RESEARCH_MAX_RESULTS = int(os.getenv("AUTO_RESEARCH_MAX_RESULTS", "4"))
TOPIC_SELECTOR_ENABLED = os.getenv("TOPIC_SELECTOR_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
TOPIC_SELECTOR_MAX_TURNS = int(os.getenv("TOPIC_SELECTOR_MAX_TURNS", "6"))
TOPIC_SELECTOR_MAX_TOKENS = int(os.getenv("TOPIC_SELECTOR_MAX_TOKENS", "180"))
TOPIC_SELECTOR_CONFIDENCE_THRESHOLD = float(os.getenv("TOPIC_SELECTOR_CONFIDENCE_THRESHOLD", "0.45"))
