import requests
import json
import time
import os
import sys
from datetime import datetime

# Configuration
API_URL = "http://localhost:8001/chat"
rag_type = "fusion"  # Can be: "fusion", "graph", "adaptive"


def print_header():
    """Print the chat application header."""
    print("=== RAG-Powered Chat Interface ===")
    print(f"Using RAG type: {rag_type}")
    print(f"Server: {API_URL}")
    print("=" * 50)
    print()

def format_timestamp():
    """Return a formatted timestamp for the current time."""
    return datetime.now().strftime("%H:%M:%S")


def print_bot_typing():
    """Show that the bot is typing."""
    sys.stdout.write(f"[{format_timestamp()}] Bot: ")
    sys.stdout.flush()

def chat_with_bot(question, session_id=None):
    """
    Send a message to the chat endpoint and stream the response.
    Returns the full response and session ID.
    """
    payload = {
        "question": question,
        "session_id": session_id,
        "rag_type": rag_type
    }

    try:
        print_bot_typing()
        
        with requests.post(API_URL, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return "", session_id

            full_response = ""
            received_session_id = session_id

            for line in response.iter_lines():
                if line:
                    # print('### line', line)
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data:"):
                        try:
                            data = json.loads(decoded[5:])
                            content = data.get("token", "")
                            received_session_id = data.get("session_id", received_session_id)
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            
                            full_response += content
                        except json.JSONDecodeError:
                            continue

            print("\n")  # Add extra line after response
            
            if not session_id and received_session_id:
                print(f"Session started with ID: {received_session_id}\n")
                
            return full_response, received_session_id
    
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {str(e)}")
        return "", session_id

def interactive_chat():
    """Run an interactive chat session with the bot."""
    session_id = None
    chat_history = []
    
    print_header()
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You > ")
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
                
            # Only exit command is supported
                
            elif not user_input.strip():
                continue
            
            response, session_id = chat_with_bot(user_input, session_id)
            chat_history.append((user_input, response))
            
        except KeyboardInterrupt:
            print("\nChat session interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    interactive_chat()
