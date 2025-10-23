from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    content: str
    score: float
    source: str
    rerank_score: Optional[float] = None