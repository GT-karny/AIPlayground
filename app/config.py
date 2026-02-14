import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "not-needed")
MODEL_NAME = os.getenv("MODEL_NAME", "LiquidAI/LFM2.5-1.2B-JP")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
