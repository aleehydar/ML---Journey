from fastapi import APIRouter, Depends
from typing import Dict, Any

from auth_middleware import AuthClaims, require_permission
from api.schemas import QuestionRequest, ChatResponse, ErrorResponse, ErrorDetail
from generation_service import generation_service

router = APIRouter(prefix="/api/v1", tags=["chat"])

@router.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat_json_endpoint(
    req: QuestionRequest,
    claims: AuthClaims = Depends(require_permission("chat:write")),
) -> Any:
    """Standard JSON endpoint for Chat/QA logic with typed error structures and SLAs."""
    
    result = await generation_service.answer_legal_question_json(
        question=req.question,
        history=req.history,
        user_id=claims.sub,
        org_id=claims.org_id,
        permissions=claims.permissions,
    )

    if "error" in result:
        # Construct error response
        # Even on errors, backend supplies require_human_review, etc.
        # But for an actual 500 error, we can return the ErrorModel
        # For our spec, we may return 200 with abstention flag or 500 if internal.
        if result["error"]["code"] == "INTERNAL":
            return ErrorResponse(
                error=ErrorDetail(
                    code=result["error"]["code"],
                    message=result["error"]["message"]
                )
            )

    return ChatResponse(
        answer=result.get("answer", "ERROR: INSUFFICIENT_EVIDENCE"),
        sources=result.get("sources", []),
        confidence_score=result.get("confidence_score", 0.0),
        requires_human_review=result.get("requires_human_review", True),
        evaluation=result.get("evaluation")
    )
