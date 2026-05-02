from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    
class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)

class DocumentSource(BaseModel):
    source_id: str
    content_snippet: str
    score: float
    metadata: Dict[str, Any]

class EvaluationMetrics(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_recall: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_human_review: bool = Field(default=False)
    evaluation: Optional[EvaluationMetrics] = None

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
