from typing import Optional
from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    question: str
    rag_type: str = Field(default='Fusion')
    session_id:  Optional[str] = Field(default=None)

