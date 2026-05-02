from typing import List, Dict

class RetrievalMetrics:
    @staticmethod
    def recall_at_k(retrieved_sources: List[str], gold_sources: set, k: int = 5) -> float:
        """Calculates Recall@K: proportion of gold sources found in top K retrieved chunks."""
        if not gold_sources:
            return 1.0  # Nothing to retrieve
        top_k = set(retrieved_sources[:k])
        hits = len(top_k.intersection(gold_sources))
        return hits / len(gold_sources)

    @staticmethod
    def mrr(retrieved_sources: List[str], gold_sources: set) -> float:
        """Calculates Mean Reciprocal Rank for the first relevant document."""
        if not gold_sources:
            return 0.0
        for i, source in enumerate(retrieved_sources):
            if source in gold_sources:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def hit_rate(retrieved_sources: List[str], gold_sources: set) -> float:
        """Calculates Hit Rate: 1.0 if any gold source is retrieved, 0.0 otherwise."""
        if not gold_sources:
            return 0.0
        return 1.0 if any(s in gold_sources for s in retrieved_sources) else 0.0

retrieval_evaluator = RetrievalMetrics()
