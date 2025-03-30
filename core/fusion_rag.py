import sys
from typing import Dict, List

from core.base_rag import BaseRAG
from db.unstructured_db import ElasticsearchRetrievalManager

sys.path.append("helpers")
from logger import create_logger
from utils import call_model
from prompts import GENERATION_PROMPT


logger = create_logger(__name__)


class FusionRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.db_obj = ElasticsearchRetrievalManager()

    async def retrieve(self, query: str, k: int = 5):
        logger.info(f"Retrieving documents for query with k={k}")
        results = await self.db_obj.search(query, top_k=k)
        logger.info("Document retrieval completed")
        return results
    
    def generate(self, query: str, messages: List[Dict], context: str):
        logger.info("Generating response")
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