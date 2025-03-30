import sys
from typing import Dict, List

from core.base_rag import BaseRAG
from db.unstructured_db import ElasticsearchRetrievalManager
from db.graph_db import GraphDatabaseManager
sys.path.append("helpers")
from logger import create_logger
from utils import call_model
from prompts import GENERATION_PROMPT, CLASSIFIER_SYSTEM_PROMPT, CLASSIFIER_PROMPT


logger = create_logger(__name__)


class AdaptiveRAG(BaseRAG):
    def __init__(self):
        super().__init__()
        self.db_obj = ElasticsearchRetrievalManager()
        self.graph_db_obj = GraphDatabaseManager()

    async def retrieve(self, query: str, k: int = 5):
        category = self._classify(query)
        
        result = ""
        if category == 'relevant':
            logger.info("Using standard Elasticsearch vector/syntatic retrieval")
            result = await self.db_obj.search(query, top_k=k)
            logger.debug(f"Retrieved {len(result.split()) if result else 0} words from Elasticsearch")

        elif category == 'complex':
            logger.info("Using hybrid retrieval (Elasticsearch + Graph DB)")
            hybrid_res = await self.db_obj.search(query, top_k=k)
            g_res = await self.graph_db_obj.search(query)
            result = hybrid_res + "\n" + g_res
        
        return result
    
    def generate(self, query: str, messages: List[Dict], context: str):
       
        messages_copy = messages.copy()
        messages_copy.append(
            {
                'role': 'user',
                'content': GENERATION_PROMPT.format(question=query, context=context),
            }
        )
        response = call_model(messages_copy, max_tokens=500, stream=True)
        return response

    @staticmethod
    def _classify(query):
        logger.info(f"Classifying query: '{query}'")
        messages = [{'role': 'system', 'content': CLASSIFIER_SYSTEM_PROMPT},
                    {'role': 'system', 'content': CLASSIFIER_PROMPT.format(query=query)},
                    ]
        logger.debug("Calling LLM for query classification")
        category = call_model(messages, max_tokens=5)
        logger.debug(f"Raw classification result: '{category}'")
        
        valid_categories = ['irrelevant', 'relevant', 'complex']
        # Ensure the returned category is valid
        for valid in valid_categories:
            if valid.lower() in category.lower():
                logger.info(f"Query classified as: '{valid}'")
                return valid
                
        # default
        logger.warning(f"Invalid classification result: '{category}', defaulting to 'relevant'")
        return "relevant"