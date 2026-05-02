import time
import hashlib
from typing import Optional, Dict, Any

class SemanticCache:
    """
    A simple dictionary-based cache that stores precise answers for previously computed queries.
    In a real production environment, this could be backed by Redis or FAISS exact-match logic.
    Here we implement a TTL-based exact query hash match for demonstration and rapid cache hits.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _hash_key(self, query: str, org_id: str) -> str:
        key = f"{org_id}::{query.lower().strip()}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get(self, query: str, org_id: str) -> Optional[Dict[str, Any]]:
        key = self._hash_key(query, org_id)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["data"]
            else:
                # Expired
                del self.cache[key]
        return None

    def set(self, query: str, org_id: str, data: Dict[str, Any]):
        key = self._hash_key(query, org_id)
        self.cache[key] = {
            "timestamp": time.time(),
            "data": data
        }

# Global cache instance
semantic_cache = SemanticCache()
