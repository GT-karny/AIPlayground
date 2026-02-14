import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("API_KEY", "not-needed")
MODEL_NAME = os.getenv("MODEL_NAME", "lfm2.5-1.2b-jp")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
