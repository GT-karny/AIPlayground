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
