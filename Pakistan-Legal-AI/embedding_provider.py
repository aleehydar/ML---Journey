"""Embedding provider with forward-compatible imports."""

from typing import Any


def build_embeddings(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Return a HuggingFace embeddings instance.
    Prefers langchain_huggingface (new), falls back to community package.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    except Exception:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    return HuggingFaceEmbeddings(model_name=model_name)
