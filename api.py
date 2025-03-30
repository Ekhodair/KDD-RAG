import uuid
import json
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from db.unstructured_db import ElasticsearchHistoryManager


sys.path.append("helpers")

from prompts import GENERATION_SYSTEM_PROMPT
from logger import create_logger
from schemas import QueryInput
from core.fusion_rag import FusionRAG
from core.adaptive_rag import AdaptiveRAG
from core.graph_rag import GraphRAG


# Map of available RAG types
RAG_REGISTRY = {
    "fusion": FusionRAG(),
    "graph": GraphRAG(),
    "adaptive": AdaptiveRAG()
}

chat_manager = ElasticsearchHistoryManager()
logger = create_logger(__name__)

app = FastAPI()

@app.post("/chat")
async def chat(request: QueryInput):
    
    session_id = request.session_id

    rag_type = request.rag_type.lower()
    if rag_type not in RAG_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid RAG type: {rag_type}. Available types: {list(RAG_REGISTRY.keys())}"
        )
    rag_instance = RAG_REGISTRY[rag_type]
    logger.info(
        f"Session ID: {session_id}, User Query: {request.question}"
    )
    if not session_id:
        session_id = str(uuid.uuid4())
        messages = []
        print("generated session id", session_id)
    else:
        messages = await chat_manager.get_chat_history(session_id)

    if not messages:
        messages = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]
    response_stream = await rag_instance(query=request.question, messages=messages)

    messages.append({"role": "user", "content": request.question})

    async def generate_response():
        final_full_response = ""
        for line in response_stream.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    json_str = decoded_line[5:].strip()
                    if json_str == "[DONE]":
                        break
                    token_data = json.loads(json_str)
                    content = token_data["choices"][0]["delta"].get("content", "")
                    final_full_response += content
                    yield f"data: {json.dumps({'token': content, 'session_id': session_id})}\n\n"        
        # Signal the end of the stream
        yield f"data: {json.dumps({'token': '', 'session_id': session_id})}\n\n"
        messages.append({
            "role": "assistant",
            "content": final_full_response
        })
        
        # Store the updated chat history
        await chat_manager.store_chat_history(
            session_id=session_id,
            messages=messages
        )
        
        logger.info(f"Session ID: {session_id}, AI Response: {final_full_response}")

    return StreamingResponse(generate_response(), media_type="text/event-stream")








