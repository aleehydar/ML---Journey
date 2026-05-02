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

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None


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
        self.vectorstore = self._load_or_build_vectorstore()
        self.bm25, self.chunk_docs = self._build_bm25_index()

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

    def _build_bm25_index(self):
        chunk_docs = self._chunk_documents()
        tokenized = [self._tokenize(d.page_content) for d in chunk_docs]
        if BM25Okapi is None:
            return None, chunk_docs
        return BM25Okapi(tokenized), chunk_docs

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _extract_intent_filters(self, query: str) -> Dict[str, Optional[str]]:
        if self.intent_llm is None:
            q = query.lower()
            return {
                "law_family": "companies" if "companies" in q else None,
                "year": re.search(r"\b(19|20)\d{2}\b", q).group(0)
                if re.search(r"\b(19|20)\d{2}\b", q)
                else None,
                "document_type": "notification" if "notification" in q else None,
            }
        prompt = f"""
Extract retrieval filters from this user query.
Return ONLY JSON with keys: law_family, year, document_type.
If unknown set null.

Query: {query}
JSON:
"""
        try:
            raw = self.intent_llm.invoke(prompt).content
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return {"law_family": None, "year": None, "document_type": None}
            parsed = json.loads(match.group(0))
            return {
                "law_family": parsed.get("law_family"),
                "year": str(parsed.get("year")) if parsed.get("year") else None,
                "document_type": parsed.get("document_type"),
            }
        except Exception:
            return {"law_family": None, "year": None, "document_type": None}

    def _doc_matches_filters(self, doc: Document, filters: Dict[str, Optional[str]]) -> bool:
        source = str(doc.metadata.get("source", "")).lower()
        if filters.get("year") and filters["year"] not in source:
            return False
        if filters.get("law_family") and filters["law_family"].lower() not in source:
            return False
        if filters.get("document_type") and filters["document_type"].lower() not in source:
            return False
        return True

    def _hybrid_candidates(self, query: str, org_id: str, filters: Dict[str, Optional[str]]) -> List[Document]:
        # Dense candidates
        dense = self.vectorstore.similarity_search_with_score(query, k=30)
        dense_docs = []
        for doc, _score in dense:
            if doc.metadata.get("organization_id", "public") != org_id:
                continue
            if not self._doc_matches_filters(doc, filters):
                continue
            dense_docs.append(doc)

        # BM25 candidates
        q_tokens = self._tokenize(query)
        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(q_tokens)
            bm25_ranked = sorted(
                zip(bm25_scores, self.chunk_docs), key=lambda x: x[0], reverse=True
            )[:30]
        else:
            # Fallback lexical scorer: token overlap count.
            bm25_ranked = []
            q_set = set(q_tokens)
            for doc in self.chunk_docs:
                score = len(q_set.intersection(set(self._tokenize(doc.page_content))))
                bm25_ranked.append((score, doc))
            bm25_ranked = sorted(bm25_ranked, key=lambda x: x[0], reverse=True)[:30]
        bm25_docs = []
        for _score, doc in bm25_ranked:
            if doc.metadata.get("organization_id", "public") != org_id:
                continue
            if not self._doc_matches_filters(doc, filters):
                continue
            bm25_docs.append(doc)

        # union by (source,text)
        seen = set()
        merged = []
        for doc in dense_docs + bm25_docs:
            key = (doc.metadata.get("source", ""), doc.page_content[:120])
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged

    def retrieve(self, query: str, org_id: str, k: int = 6) -> RetrievalResult:
        filters = self._extract_intent_filters(query)
        candidates = self._hybrid_candidates(query, org_id=org_id, filters=filters)
        if not candidates:
            return RetrievalResult(chunks=[], top_score=0.0)

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        
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
        return RetrievalResult(chunks=chunks, top_score=normalized_top)

    def get_documents_for_org(self, org_id: str) -> Dict[str, str]:
        rows = {}
        for item in self.legal_texts:
            if item.get("organization_id", "public") != org_id:
                continue
            rows[item["source"]] = item["text"]
        return rows


retrieval_service = RetrievalService()
