import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

load_dotenv(dotenv_path=dotenv_path)

HF_TOKEN = os.getenv("HF_TOKEN")
LOGGING_LEVEL = 'DEBUG'

VLLM_URL = "http://localhost:8000/v1/chat/completions"