import asyncio
import json
import logging
import os
import re
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()
logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """
You are an expert AI evaluator assessing RAG (Retrieval-Augmented Generation) system quality for Pakistani Legal AI.
You evaluate three metrics on a scale of 0.0 to 1.0 based on strict rubrics:

1. FAITHFULNESS (0.0 to 1.0):
   - 1.0: Every statement and article citation in the answer is strictly derived from and supported by the retrieved context. No invented facts or unsupported claims.
   - 0.5: Mostly supported, but contains minor ungrounded assumptions or slight extrapolations.
   - 0.0: Contains explicit factual contradictions, hallucinated article numbers, or unsupported legal claims.

2. ANSWER_RELEVANCE (0.0 to 1.0):
   - 1.0: The answer directly, completely, and accurately addresses the user's specific legal question.
   - 0.5: Partially answers the question or includes excessive irrelevant boilerplate while missing key points.
   - 0.0: Completely misses the question, gives off-topic advice, or returns a generic refusal when context was available.

3. CONTEXT_RECALL (0.0 to 1.0):
   - 1.0: The retrieved context chunks contain all necessary legal articles, text, or figures to fully answer the question.
   - 0.5: The context contains partial information (e.g. general legal domain) but misses the specific article or exact rule.
   - 0.0: The retrieved context is completely irrelevant or empty.

Return ONLY a valid JSON object in this exact schema:
{
  "faithfulness": <float 0.0-1.0>,
  "answer_relevance": <float 0.0-1.0>,
  "context_recall": <float 0.0-1.0>,
  "reasoning": "<1-2 sentence explanation of the scores>"
}
"""


class RAGASEvaluator:
    """
    Direct LLM-Judge Evaluator powered by ChatGroq.
    Bypasses external RAGAS library dependencies entirely to eliminate import errors
    and provide transparent, rubric-backed RAG quality metrics.
    """

    def __init__(self):
        try:
            self.judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
            self.available = True
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGroq Judge: {e}")
            self.judge_llm = None
            self.available = False

    @staticmethod
    def _heuristic_context_recall(question: str, contexts: List[str]) -> float:
        """Fallback approximation when model-backed evaluation is disabled."""
        question_tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())
        keywords = [t for t in question_tokens if len(t) > 3]
        if not keywords or not contexts:
            return 0.0
        context_blob = " ".join(contexts).lower()
        covered = sum(1 for token in set(keywords) if token in context_blob)
        return covered / max(len(set(keywords)), 1)

    async def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Evaluates (question, answer, contexts) triple using single-call LLM judge rubric.
        """
        if not self.judge_llm or not question or not answer:
            fallback = self._heuristic_context_recall(question, contexts)
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": round(fallback, 3),
                "overall_score": round(fallback / 3.0, 3),
                "reasoning": "Judge unavailable or empty input",
            }

        context_blob = "\n\n".join(contexts) if contexts else "No context retrieved."
        prompt = f"""Evaluate this RAG interaction:

[USER QUESTION]
{question}

[RETRIEVED CONTEXT CHUNKS]
{context_blob[:2500]}

[SYSTEM GENERATED ANSWER]
{answer}
"""
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            resp = await asyncio.wait_for(self.judge_llm.ainvoke(messages), timeout=25.0)
            text = resp.content or ""
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                f_val = max(0.0, min(1.0, float(data.get("faithfulness", 0.0))))
                r_val = max(0.0, min(1.0, float(data.get("answer_relevance", 0.0))))
                c_val = max(0.0, min(1.0, float(data.get("context_recall", 0.0))))
                overall = round((f_val + r_val + c_val) / 3.0, 3)

                return {
                    "faithfulness": round(f_val, 3),
                    "answer_relevance": round(r_val, 3),
                    "context_recall": round(c_val, 3),
                    "overall_score": overall,
                    "reasoning": str(data.get("reasoning", "")),
                }
        except Exception as e:
            logger.warning(f"LLM judge evaluation exception: {e}")

        fallback = self._heuristic_context_recall(question, contexts)
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": round(fallback, 3),
            "overall_score": round(fallback / 3.0, 3),
            "reasoning": "Fallback to heuristic due to judge exception",
        }

    def evaluate_single_sync(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        return asyncio.run(self.evaluate_single(question, answer, contexts))


# Global evaluator instance
ragas_evaluator = RAGASEvaluator()
