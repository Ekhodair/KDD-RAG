
from typing import List, Dict, Any
from core.base_rag import BaseRAG
from db.graph_db import GraphDatabaseManager
sys.path.append("helpers")
from logger import create_logger
from utils import call_model
from prompts import GENERATION_PROMPT

logger = create_logger(__name__)

class GraphRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.db_obj = GraphDatabaseManager()

    async def retrieve(self, query: str, k: int):
        results = await self.db_obj.search(query)
        logger.info("Document retrieval completed")

        return results 

    def generate(self, query: str, messages: List[Dict], context: str):
        messages_copy = messages.copy()
        messages_copy.append(
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(question=query, context=context),
            }
        )
        response = call_model(messages_copy, max_tokens=500, stream=True)
        logger.info("Response generation completed")

        return response