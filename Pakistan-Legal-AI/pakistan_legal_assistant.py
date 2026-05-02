"""
Backward-compatible facade for legacy imports.
Core logic now lives in retrieval_service.py and generation_service.py.
"""

from generation_service import generation_service
from retrieval_service import retrieval_service

legal_texts = retrieval_service.legal_texts
vectorstore = retrieval_service.vectorstore
cross_encoder = retrieval_service.cross_encoder


def _search_pakistan_law(query: str, org_id: str = "public") -> str:
    result = retrieval_service.retrieve(query=query, org_id=org_id, k=6)
    if not result.chunks:
        return "ERROR: INSUFFICIENT_EVIDENCE"
    return "\n\n".join(
        [f"[Source: {chunk.source_id}]\n{chunk.text}" for chunk in result.chunks]
    )


def _calculate_tax(annual_income: float) -> str:
    income = float(annual_income)
    if income <= 600000:
        tax = 0.0
    elif income <= 1200000:
        tax = (income - 600000) * 0.025
    elif income <= 2400000:
        tax = 15000 + ((income - 1200000) * 0.125)
    elif income <= 3200000:
        tax = 165000 + ((income - 2400000) * 0.225)
    elif income <= 4100000:
        tax = 345000 + ((income - 3200000) * 0.275)
    else:
        tax = 592500 + ((income - 4100000) * 0.35)
    return f"For annual income {income:,.2f} PKR, annual tax is {tax:,.2f} PKR."


async def answer_legal_question(
    question: str,
    history=None,
    user_id: str = "legacy-user",
    org_id: str = "public",
):
    async for event in generation_service.answer_legal_question(
        question=question, history=history or [], user_id=user_id, org_id=org_id
    ):
        yield event


class ToolCompat:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def invoke(self, payload):
        if isinstance(payload, dict):
            return self.fn(**payload)
        return self.fn(payload)


search_pakistan_law = ToolCompat(_search_pakistan_law)
calculate_tax = ToolCompat(_calculate_tax)
