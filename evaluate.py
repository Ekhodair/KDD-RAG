import os
import re
import json
import time
import pandas as pd
import requests
from typing import List, Dict, Any, Optional
import sys
sys.path.append("helpers")
from prompts import EVAL_AGENT_SYS_PROMPT, EVAL_AGENT_PROMPT
from utils import call_model

RAG_TYPES = ['fusion', 'graph', 'adaptive']

def parse_response(response: str):
    match = re.search(r'\[\[(.*?)\]\]', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    return None


def call_endpoint(query, session_id, rag_type):
    """
    Function to call the /chat endpoint for the specified RAG type.
    """
    url = "http://localhost:8081/chat"
    
    payload = {
        "question": query,
        "session_id": session_id,
        "rag_type": rag_type
    }
    
    headers = {"Content-Type": "application/json"}
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Error calling endpoint: {response.status_code}, {response.text}")
    
    final_text = ""
    session_id = None
    
    # Process the stream to get the final response
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                json_str = line[6:]  # Remove 'data: ' prefix
                try:
                    data = json.loads(json_str)
                    token = data.get('token', '')
                    final_text += token
                    session_id = data.get('session_id')
                except json.JSONDecodeError:
                    continue
    
    end_time = time.time()
    latency = end_time - start_time
    
    return {
        'session_id': session_id,
        'final_text': final_text,
        'latency': latency
    }


def evaluate(messages):
    """
    Evaluate the conversation using the LLM.
    """
    # Prepare evaluation messages
    eval_messages = [
        {"role": "system", "content": EVAL_AGENT_SYS_PROMPT}
    ]
    
    # Format conversation for evaluation
    conversation_str = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            conversation_str += f"User: {content}\n\n"
        elif role == 'assistant':
            conversation_str += f"Assistant: {content}\n\n"
    
    # Add evaluation prompt with the conversation
    eval_prompt = EVAL_AGENT_PROMPT.format(chat_history=conversation_str)
    eval_messages.append({"role": "user", "content": eval_prompt})
    
    # Call model for evaluation
    resp = call_model(messages=eval_messages, max_tokens=600, stream=False)
    parsed = parse_response(resp)
    return parsed


def load_data(file_path):
    """
    Load prompts from a text file.
    """
    with open(file_path, 'r') as f:
        data = [line.strip() for line in f.readlines() if line.strip()]
    return data


def main(data_path):
    """
    Main function to evaluate different RAG types.
    """
    paths = [f for f in os.listdir(data_path) if f.lower().endswith('.txt')]
    
    # Initialize results dictionary
    result_dict = {}
    for rag in RAG_TYPES:
        result_dict[rag] = {
            "Relevance": [],
            "SPAR": [],
            "CH": [],
            "RQ": [],
            "Latency": []
        }
    
    # Process each file for each RAG type
    for rag in RAG_TYPES:
        print(f"Evaluating {rag} RAG...")
        total_latency = 0
        total_requests = 0
        
        for fname in paths:
            print(f"  Processing file: {fname}")
            session_id = None
            prompts = load_data(f'{data_path}/{fname}')
            history = []
            file_latency = 0
            
            for prompt in prompts:
                print(f"    Query: {prompt}")
                res = call_endpoint(query=prompt, session_id=session_id, rag_type=rag)
                print(f"    AI: {res['final_text']}")
                session_id = res['session_id']
                history.append({'role': 'user', 'content': prompt})
                history.append({'role': 'assistant', 'content': res['final_text']})
                file_latency += res['latency']
                total_latency += res['latency']
                total_requests += 1
            
            # Evaluate the conversation
            metrics = evaluate(messages=history)
            if metrics:
                for metric in ["Relevance", "SPAR", "CH", "RQ"]:
                    if metric in metrics:
                        result_dict[rag][metric].append(metrics[metric])
                
                result_dict[rag]["Latency"].append(file_latency / len(prompts))
    final_results = {}
    for rag in RAG_TYPES:
        metrics = {}
        for metric in ["Relevance", "SPAR", "CH", "RQ", "Latency"]:
            values = result_dict[rag][metric]
            if values:
                avg_value = round(sum(values) / len(values), 2)
            else:
                avg_value = 0.0
            metrics[metric] = avg_value
        final_results[rag] = metrics
    
    # Save results to CSV
    save_results(final_results)
    
    print("\nEvaluation Results:")
    df = pd.DataFrame(final_results).T
    print(df)
    return df


def save_results(results):
   
    df = pd.DataFrame(results).T
    df.index.name = "RAG Type"
    df.to_csv("rag_evaluation_results.csv")
    print(f"Results saved to rag_evaluation_results.csv")


if __name__ == "__main__":
    base_dir = "./eval_prompts"
    main(base_dir)