import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from embedding_provider import build_embeddings

import time
from abc import ABC, abstractmethod

@dataclass
class RetrievalChunk:
    source_id: str
    text: str
    score: float
    metadata: Dict


@dataclass
class RetrievalResult:
    chunks: List[RetrievalChunk]
    top_score: float
    hyde_doc: Optional[str] = None


class SparseRetriever(ABC):
    @abstractmethod
    def get_top_k(self, query: str, k: int) -> List[Document]:
        pass

class RankBM25Retriever(SparseRetriever):
    def __init__(self, docs: List[Document]):
        from rank_bm25 import BM25Okapi
        self.docs = docs
        tokenized = [self._tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def get_top_k(self, query: str, k: int) -> List[Document]:
        q_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(zip(scores, self.docs), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in ranked[:k]]

class QdrantSparseRetriever(SparseRetriever):
    def __init__(self, docs: List[Document]):
        # Placeholder for future Qdrant/Elasticsearch sparse backend
        pass

    def get_top_k(self, query: str, k: int) -> List[Document]:
        return []


class RetrievalService:
    def __init__(self):
        self.base_dir = os.path.dirname(__file__)
        self.default_org_id = os.getenv("DEFAULT_ORG_ID", "public")
        self.legal_texts_file = os.path.join(self.base_dir, "legal_texts.json")
        self.vectorstore_dir = os.path.join(self.base_dir, "vectorstore")
        self.embeddings = build_embeddings("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            self.intent_llm = ChatGroq(model="llama-3.1-8b-instant")
        except Exception:
            self.intent_llm = None
        self.legal_texts = self._load_legal_texts()
        
        t0 = time.time()
        self.vectorstore = self._load_or_build_vectorstore()
        
        # Ensure we can safely check index size regardless of FAISS version
        ntotal = getattr(self.vectorstore.index, "ntotal", "unknown") if hasattr(self.vectorstore, "index") else "unknown"
        print(f"FAISS index loaded in {time.time()-t0:.2f}s with {ntotal} chunks")
        
        self.chunk_docs = self._chunk_documents()
        t1 = time.time()
        self.sparse_retriever = RankBM25Retriever(self.chunk_docs)
        print(f"BM25 index built in {time.time()-t1:.2f}s for {len(self.chunk_docs)} chunks")

    def _load_legal_texts(self) -> List[Dict]:
        if not os.path.exists(self.legal_texts_file):
            return []
        with open(self.legal_texts_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        normalized = []
        for row in raw:
            row = dict(row)
            row["organization_id"] = row.get("organization_id", self.default_org_id)
            normalized.append(row)
        return normalized

    def _chunk_documents(self) -> List[Document]:
        docs = [
            Document(
                page_content=item["text"],
                metadata={
                    "source": item["source"],
                    "organization_id": item.get("organization_id", "public"),
                },
            )
            for item in self.legal_texts
        ]
        splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)
        return splitter.split_documents(docs)

    def _load_or_build_vectorstore(self):
        if os.path.isdir(self.vectorstore_dir):
            try:
                return FAISS.load_local(
                    self.vectorstore_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                pass
        chunks = self._chunk_documents()
        store = FAISS.from_documents(chunks, self.embeddings)
        store.save_local(self.vectorstore_dir)
        return store

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def generate_hypothetical_document(self, query: str) -> str:
        prompt = f"Write a short passage (2-4 sentences) that would answer this legal question as if it appeared in the Pakistani Constitution or relevant statute: {query}"
        try:
            if self.intent_llm:
                return self.intent_llm.invoke(prompt).content.strip()
        except Exception as e:
            print(f"HyDE generation failed: {e}")
        return query

    def _extract_intent_filters(self, query: str) -> Dict[str, Optional[str]]:
        if self.intent_llm is None:
            q = query.lower()
            article_match = re.search(r"\barticle\s+(\d+[A-Za-z]*)\b", q)
            return {
                "law_family": "companies" if "companies" in q else None,
                "year": re.search(r"\b(19|20)\d{2}\b", q).group(0)
                if re.search(r"\b(19|20)\d{2}\b", q)
                else None,
                "document_type": "notification" if "notification" in q else None,
                "article": article_match.group(1) if article_match else None,
            }
        prompt = f"""
Extract retrieval filters from this user query.
Return ONLY JSON with keys: law_family, year, document_type, article.
For article, extract only the number/identifier (e.g. "14" or "2A").
If unknown set null.

Query: {query}
JSON:
"""
        try:
            raw = self.intent_llm.invoke(prompt).content
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return {"law_family": None, "year": None, "document_type": None, "article": None}
            parsed = json.loads(match.group(0))
            
            def _clean(val):
                if val is None: return None
                if str(val).lower().strip() in ["null", "none", "unknown", ""]: return None
                return str(val)

            return {
                "law_family": _clean(parsed.get("law_family")),
                "year": _clean(parsed.get("year")),
                "document_type": _clean(parsed.get("document_type")),
                "article": _clean(parsed.get("article")),
            }
        except Exception:
            return {"law_family": None, "year": None, "document_type": None, "article": None}

    def _doc_matches_filters(self, doc: Document, filters: Dict[str, Optional[str]]) -> bool:
        source = str(doc.metadata.get("source", "")).lower()
        if filters.get("year") and filters["year"] not in source:
            return False
        if filters.get("law_family") and filters["law_family"].lower() not in source:
            return False
        if filters.get("document_type") and filters["document_type"].lower() not in source:
            return False
        return True

    def _hybrid_candidates(self, query: str, hyde_doc: Optional[str], org_id: str, filters: Dict[str, Optional[str]]) -> List[Document]:
        print(f"\n--- DIAGNOSTICS: RETRIEVAL TRACE ---")
        print(f"Raw Query: {query}")
        print(f"Extracted Filters: {filters}")
        
        # Dense candidates (Union of raw query and HyDE if present)
        dense_docs = []
        raw_dense = self.vectorstore.similarity_search_with_score(query, k=30)
        for doc, _ in raw_dense:
            if doc.metadata.get("organization_id", "public") == org_id:
                dense_docs.append(doc)
                
        if hyde_doc:
            hyde_dense = self.vectorstore.similarity_search_with_score(hyde_doc, k=30)
            for doc, _ in hyde_dense:
                if doc.metadata.get("organization_id", "public") == org_id:
                    dense_docs.append(doc)
            
        print(f"FAISS Candidates Retrieved: {len(dense_docs)}")

        # BM25 candidates
        bm25_raw_docs = self.sparse_retriever.get_top_k(query, k=30)
        bm25_docs = []
        for doc in bm25_raw_docs:
            if doc.metadata.get("organization_id", "public") == org_id:
                bm25_docs.append(doc)
            
        print(f"BM25 Candidates Retrieved: {len(bm25_docs)}")
        
        # Exact Article Injection
        exact_article_docs = []
        article_filter = filters.get("article")
        if article_filter:
            # Look for exact article number in chunks, e.g., "Article 14" or "14. "
            article_pattern = re.compile(rf"\barticle\s+{article_filter}\b|\b{article_filter}\.\s+", re.IGNORECASE)
            for doc in self.chunk_docs:
                if doc.metadata.get("organization_id", "public") == org_id:
                    if article_pattern.search(doc.page_content):
                        exact_article_docs.append(doc)
            print(f"Exact Article Candidates Injected: {len(exact_article_docs)}")

        # union by (source,text)
        seen = set()
        merged = []
        for doc in dense_docs + bm25_docs + exact_article_docs:
            key = (doc.metadata.get("source", ""), doc.page_content[:120])
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged

    def retrieve(self, query: str, org_id: str, k: int = 6) -> RetrievalResult:
        use_hyde = os.getenv("USE_HYDE", "true").lower() == "true"
        hyde_doc = None
        if use_hyde:
            hyde_doc = self.generate_hypothetical_document(query)
            # Sanity check for hallucinated hedging language
            hedge_words = ["i couldn't find", "unfortunately", "i don't have", "i do not have", "i cannot", "unknown", "no information"]
            if hyde_doc and any(hw in hyde_doc.lower() for hw in hedge_words):
                print(f"Skipping HyDE due to hedging language: {hyde_doc[:50]}...")
                hyde_doc = None
            
        filters = self._extract_intent_filters(query)
        candidates = self._hybrid_candidates(query, hyde_doc, org_id=org_id, filters=filters)
        if not candidates:
            return RetrievalResult(chunks=[], top_score=0.0, hyde_doc=hyde_doc)

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        
        print(f"Total Unique Candidates for Reranking: {len(candidates)}")
        print(f"Top 5 CrossEncoder Scores:")
        for i, (score, doc) in enumerate(ranked[:5]):
            print(f"  {i+1}. Score: {score:.2f} | Source: {doc.metadata.get('source', 'unknown')}")
        
        # Adaptive retrieval depth
        dynamic_k = k
        if ranked:
            top_raw_score = ranked[0][0]
            for idx, (score, doc) in enumerate(ranked):
                if idx >= k:
                    # Continue gathering up to 12 if score remains very close to the top candidate
                    if score >= top_raw_score - 2.0 and idx < 12:
                        dynamic_k = idx + 1
                    else:
                        break
                        
        chunks = [
            RetrievalChunk(
                source_id=str(doc.metadata.get("source", "unknown")),
                text=doc.page_content,
                score=float(score),
                metadata=doc.metadata,
            )
            for score, doc in ranked[:dynamic_k]
        ]
        top = chunks[0].score if chunks else 0.0
        # normalize cross-encoder style score to 0..1 usable confidence
        normalized_top = 1.0 / (1.0 + pow(2.71828, -top))
        return RetrievalResult(chunks=chunks, top_score=normalized_top, hyde_doc=hyde_doc)

    def get_documents_for_org(self, org_id: str) -> Dict[str, str]:
        rows = {}
        for item in self.legal_texts:
            if item.get("organization_id", "public") != org_id:
                continue
            rows[item["source"]] = item["text"]
        return rows


retrieval_service = RetrievalService()
