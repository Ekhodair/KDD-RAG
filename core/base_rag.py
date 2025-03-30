from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRAG(ABC):
    
    @abstractmethod
    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        pass
    
    async def __call__(self, query: str, messages, k: int = 5):
        contexts = await self.retrieve(query, k)        
        response = self.generate(query, messages, contexts)
        
        return response

