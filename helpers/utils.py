import requests
import re
import json
from typing import List, Dict
from constants import VLLM_URL

def call_model(messages: List[Dict[str, str]], max_tokens, stream=False) -> str:
    """
    Call the LLM API with the provided messages.
    """
    
    payload = {
        "model": "casperhansen/llama-3.3-70b-instruct-awq",
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(VLLM_URL, headers=headers, data=json.dumps(payload))
    if not stream:
        return response.json()['choices'][0]['message']['content'].strip()
    return response

def clean_text(text):
    text = re.sub(r"[\"'\[\]]", "", text)  
    text = re.sub(r"[.,!?;:(){}]", "", text)
    return text.strip()

