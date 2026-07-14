import asyncio
import logging
import re
from typing import Dict, List

from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import AnswerRelevancy, Faithfulness
    try:
        # Newer/alternate ragas variants expose ContextRecall directly.
        from ragas.metrics import ContextRecall  # type: ignore
    except Exception:  # pragma: no cover - compatibility guard
        ContextRecall = None
    RAGAS_AVAILABLE = True
except Exception as e:  # pragma: no cover - compatibility guard
    print(f"WARNING: ragas import failed, evaluation features disabled: {e}")
    evaluate = None
    AnswerRelevancy = None
    Faithfulness = None
    ContextRecall = None
    RAGAS_AVAILABLE = False
except Exception:  # pragma: no cover - compatibility guard
    ContextRecall = None

logger = logging.getLogger(__name__)

class RAGASEvaluator:
    def __init__(self):
        """Initialize the RAGAS evaluator with required metrics."""
        self.metrics = []
        if RAGAS_AVAILABLE:
            self.metrics = [Faithfulness(), AnswerRelevancy()]
            if ContextRecall is not None:
                self.metrics.append(ContextRecall())

    @staticmethod
    def _normalize_score(value) -> float:
        try:
            if value is None:
                return 0.0
            score = float(value)
            if score < 0.0:
                return 0.0
            if score > 1.0:
                return 1.0
            return score
        except Exception:
            return 0.0

    @staticmethod
    def _heuristic_context_recall(question: str, contexts: List[str]) -> float:
        """
        Fallback approximation when model-backed ContextRecall cannot run.
        Measures how many key question terms appear in retrieved contexts.
        """
        question_tokens = re.findall(r"[a-zA-Z0-9]+", question.lower())
        keywords = [t for t in question_tokens if len(t) > 3]
        if not keywords or not contexts:
            return 0.0
        context_blob = " ".join(contexts).lower()
        covered = sum(1 for token in set(keywords) if token in context_blob)
        return covered / max(len(set(keywords)), 1)
    
    async def evaluate_single(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Evaluate a single (question, answer, contexts) triple using RAGAS.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
            
        Returns:
            Dictionary with evaluation scores
        """
        if not RAGAS_AVAILABLE:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": self._heuristic_context_recall(question, contexts),
                "ragas_unavailable": True,
            }
        try:
            # Create a dataset with a single sample.
            # Include both old and newer ragas field names for compatibility.
            dataset_dict = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "user_input": [question],
                "response": [answer],
                "retrieved_contexts": [contexts],
                # We do not have human gold references in production traffic.
                # Use model answer as reference placeholder so ContextRecall can run.
                "reference": [answer],
            }

            dataset = Dataset.from_dict(dataset_dict)

            # Run evaluation
            result = evaluate(dataset=dataset, metrics=self.metrics)

            context_recall = 0.0
            if "context_recall" in result and result["context_recall"]:
                context_recall = self._normalize_score(result["context_recall"][0])
            else:
                context_recall = self._heuristic_context_recall(question, contexts)

            # Extract scores
            scores = {
                "faithfulness": self._normalize_score(result["faithfulness"][0] if result["faithfulness"] else 0.0),
                "answer_relevance": self._normalize_score(result["answer_relevancy"][0] if result["answer_relevancy"] else 0.0),
                "context_recall": self._normalize_score(context_recall),
            }

            # Calculate overall score
            scores["overall_score"] = sum(scores.values()) / len(scores)

            return scores

        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {str(e)}")
            # Keep metrics measurable even if model-backed evaluator is unavailable.
            fallback_context_recall = self._heuristic_context_recall(question, contexts)
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_recall": self._normalize_score(fallback_context_recall),
                "overall_score": self._normalize_score(fallback_context_recall / 3.0),
            }
    
    def evaluate_single_sync(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Synchronous wrapper for evaluate_single.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
            
        Returns:
            Dictionary with evaluation scores
        """
        return asyncio.run(self.evaluate_single(question, answer, contexts))
    
    async def evaluate_batch(self, evaluations: List[Dict[str, any]]) -> List[Dict[str, float]]:
        """
        Evaluate multiple (question, answer, contexts) triples in batch.
        
        Args:
            evaluations: List of dictionaries with 'question', 'answer', and 'contexts' keys
            
        Returns:
            List of evaluation score dictionaries
        """
        if not evaluations:
            return []

        if not RAGAS_AVAILABLE:
            return [
                {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_recall": self._heuristic_context_recall(
                        eval_item.get("question", ""), eval_item.get("contexts", [])
                    ),
                    "ragas_unavailable": True,
                }
                for eval_item in evaluations
            ]

        try:
            # Create dataset from batch (old + new ragas keys for compatibility)
            dataset_dict = {
                "question": [eval_item["question"] for eval_item in evaluations],
                "answer": [eval_item["answer"] for eval_item in evaluations],
                "contexts": [eval_item["contexts"] for eval_item in evaluations],
                "user_input": [eval_item["question"] for eval_item in evaluations],
                "response": [eval_item["answer"] for eval_item in evaluations],
                "retrieved_contexts": [eval_item["contexts"] for eval_item in evaluations],
                "reference": [eval_item["answer"] for eval_item in evaluations],
            }

            dataset = Dataset.from_dict(dataset_dict)

            # Run batch evaluation
            result = evaluate(dataset=dataset, metrics=self.metrics)

            # Extract scores for each sample
            batch_results = []
            for i in range(len(evaluations)):
                raw_context_recall = None
                if "context_recall" in result and result["context_recall"] and i < len(result["context_recall"]):
                    raw_context_recall = result["context_recall"][i]

                scores = {
                    "faithfulness": self._normalize_score(
                        result["faithfulness"][i] if result["faithfulness"] and i < len(result["faithfulness"]) else 0.0
                    ),
                    "answer_relevance": self._normalize_score(
                        result["answer_relevancy"][i] if result["answer_relevancy"] and i < len(result["answer_relevancy"]) else 0.0
                    ),
                    "context_recall": self._normalize_score(
                        raw_context_recall
                        if raw_context_recall is not None
                        else self._heuristic_context_recall(
                            evaluations[i]["question"], evaluations[i]["contexts"]
                        )
                    ),
                }
                scores["overall_score"] = sum(scores.values()) / len(scores)
                batch_results.append(scores)

            return batch_results

        except Exception as e:
            logger.error(f"Error in batch RAGAS evaluation: {str(e)}")
            # Return resilient fallback scores on failure.
            fallback = []
            for item in evaluations:
                c_recall = self._heuristic_context_recall(item["question"], item["contexts"])
                fallback.append(
                    {
                        "faithfulness": 0.0,
                        "answer_relevance": 0.0,
                        "context_recall": self._normalize_score(c_recall),
                        "overall_score": self._normalize_score(c_recall / 3.0),
                    }
                )
            return fallback

# Global evaluator instance
ragas_evaluator = RAGASEvaluator()
