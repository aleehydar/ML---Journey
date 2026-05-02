import pytest
import asyncio
from generation_service import generation_service

@pytest.mark.asyncio
async def test_numeric_trap_abstention():
    """
    Test that the RAG model abstains when queried with a highly specific numeric trap 
    that does not exist in the legal corpus (e.g., a made up tax bracket penalty).
    """
    question = "What is the penalty if my tax bracket is exactly 59.25% under the fictional Article 99?"
    
    result = await generation_service.answer_legal_question_json(
        question=question,
        user_id="test-user",
        org_id="public"
    )
    
    # It should abstain due to missing grounding for the numeric trap
    assert result["answer"] == "ERROR: INSUFFICIENT_EVIDENCE", "Model failed to abstain on numeric trap"
    assert result["requires_human_review"] is True

@pytest.mark.asyncio
async def test_conflicting_documents_handling():
    """
    Test how the system behaves when asking an ambiguous or out-of-context question 
    that might cause retrieval but low confidence.
    """
    question = "According to both the medical act and the companies act, what is the exact overlapping tenure of a director?"
    
    result = await generation_service.answer_legal_question_json(
        question=question,
        user_id="test-user",
        org_id="public"
    )
    
    # We don't strictly assert abstention unless the documents truly don't exist,
    # but we DO assert that confidence is either explicitly measured and humans are flagged.
    if result["answer"] != "ERROR: INSUFFICIENT_EVIDENCE":
        # If it somehow answers, it must be low confidence because the intersection of those acts on that specific topic is extremely nuanced/absent.
        assert result["requires_human_review"] is True
