"""
Simplified RAG (Retrieval-Augmented Generation) Tool
=====================================================
Lightweight, self-contained RAG implementation using:
- ChromaDB for local vector storage
- LLM API for embeddings (OpenAI-compatible), or local SentenceTransformer as fallback
- Designed as a reusable tool for ReactAgent

Pattern: Any format → text extraction → chunking → embedding → vector store → semantic search
"""

import os
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

# --- Vector Store Backend (ChromaDB) ---
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

# --- Local Embedding Backend (optional fallback) ---
try:
    from sentence_transformers import SentenceTransformer
    _LOCAL_EMBED_AVAILABLE = True
except ImportError:
    _LOCAL_EMBED_AVAILABLE = False


class RAGTool:
    """
    Lightweight RAG tool for the Meta-Agent system.

    Embedding priority: LLM API > local SentenceTransformer > keyword fallback

    Usage:
        rag = RAGTool(knowledge_base_path="./rag_kb")
        rag.add_text("Python is a programming language.", doc_id="doc1")
        results = rag.search("What is Python?", top_k=3)
        answer = rag.ask("What is Python?")  # returns context snippets
    """

    def __init__(
        self,
        knowledge_base_path: str = "./rag_kb",
        collection_name: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        embed_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.knowledge_base_path = knowledge_base_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._client = None
        self._collection = None
        self._embedder = None
        self._embed_mode = "keyword"  # keyword / local / api
        self._ready = False

        # Config for API-based embeddings
        self._api_key = os.environ.get('LLM_API_KEY', '')
        self._api_base = os.environ.get('LLM_BASE_URL', '')
        self._api_model = os.environ.get('LLM_MODEL_ID', '')

        os.makedirs(knowledge_base_path, exist_ok=True)

        # Initialize ChromaDB
        if _CHROMA_AVAILABLE:
            try:
                self._client = chromadb.PersistentClient(
                    path=os.path.join(knowledge_base_path, "chroma_db"),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                self._client = None
                self._collection = None

        # Try API-based embeddings first (no download needed)
        if self._api_key and self._api_base:
            self._embed_mode = "api"
        # Try local embedder as fallback
        elif _LOCAL_EMBED_AVAILABLE:
            try:
                import os as _os
                _os.environ.setdefault('HF_HUB_DISABLE_SSL_VERIFY', '1')
                self._embedder = SentenceTransformer(embed_model_name)
                self._embed_mode = "local"
            except Exception:
                self._embedder = None

        self._ready = self._collection is not None

    @property
    def ready(self) -> bool:
        return self._ready

    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the LLM API's embedding endpoint."""
        url = self._api_base.rstrip('/')
        if not url.endswith('/v1'):
            url += '/v1'
        url += '/embeddings'

        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': self._api_model,
            'input': texts,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            data = resp.json()
            if 'data' in data:
                return [item['embedding'] for item in data['data']]
        except Exception:
            pass
        return None

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local SentenceTransformer model."""
        if not self._embedder:
            return None
        try:
            embeddings = self._embedder.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception:
            return None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the best available method."""
        # Try local model first
        if self._embed_mode == "local":
            result = self._embed_local(texts)
            if result:
                return result

        # Try API
        if self._embed_mode == "api":
            result = self._embed_api(texts)
            if result:
                return result

        # Fallback to keyword-based (simple bag-of-words average)
        return self._keyword_embed(texts)

    def _keyword_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Simple keyword-based embedding fallback.
        Uses character n-gram overlap for rough semantic matching.
        """
        import re
        dim = 128  # small fixed dimension
        embeddings = []
        for text in texts:
            # Build a simple frequency vector from character bigrams
            vec = [0.0] * dim
            text_lower = re.sub(r'[^a-zA-Z一-鿿\s]', '', (text or "").lower())
            words = text_lower.split()
            for word in words:
                for i in range(0, len(word) - 1):
                    bigram = word[i:i + 2]
                    idx = hash(bigram) % dim
                    vec[idx] += 1.0
            # Normalize
            total = sum(v * v for v in vec) ** 0.5
            if total > 0:
                vec = [v / total for v in vec]
            embeddings.append(vec)
        return embeddings

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

    # ---- Public API ----

    def add_text(self, text: str, doc_id: str = "", metadata: dict = None) -> dict:
        """
        Add a text document to the knowledge base.

        Args:
            text: The document text content
            doc_id: Optional document ID (auto-generated if empty)
            metadata: Optional metadata dict (title, source, etc.)

        Returns:
            dict with status, chunk_count, doc_id
        """
        if not self._ready:
            return {"status": "error", "message": "RAG tool not ready. Check chromadb + sentence-transformers installation."}

        doc_id = doc_id or str(uuid.uuid4())[:8]
        chunks = self._chunk_text(text)

        if not chunks:
            return {"status": "error", "message": "Empty text"}

        embeddings = self._embed(chunks)
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        # ChromaDB requires non-empty metadata or None
        meta = metadata if metadata else {"doc_id": doc_id}
        metadatas = [meta for _ in chunks]

        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        return {
            "status": "ok",
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "message": f"Added {len(chunks)} chunks from document '{doc_id}'",
        }

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search over the knowledge base.

        Returns list of {doc_id, chunk_id, content, score}
        """
        if not self._ready:
            return []

        query_embedding = self._embed([query])
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, 20),
        )

        formatted = []
        if results and results.get("ids") and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                formatted.append({
                    "chunk_id": chunk_id,
                    "doc_id": results["metadatas"][0][i].get("doc_id", "") if results["metadatas"] else "",
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "score": round(1.0 - results["distances"][0][i], 4) if results.get("distances") else 0.0,
                })

        return formatted

    def ask(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve context snippets for a query.
        Returns formatted context string for LLM consumption.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return "[No relevant documents found in the knowledge base.]"

        parts = []
        for i, r in enumerate(results):
            parts.append(f"--- Context {i + 1} (score: {r['score']}) ---\n{r['content']}")

        return "\n\n".join(parts)

    def stats(self) -> dict:
        """Return knowledge base statistics."""
        if not self._collection:
            return {"status": "not ready", "collection": self.collection_name, "documents": 0, "chunks": 0}

        try:
            count = self._collection.count()
        except Exception:
            count = 0

        return {
            "status": "ready" if self._ready else "partial",
            "collection": self.collection_name,
            "chunks": count,
            "path": self.knowledge_base_path,
        }

    def delete(self, doc_id: str) -> dict:
        """Delete a document and all its chunks."""
        if not self._collection:
            return {"status": "error", "message": "Not ready"}

        try:
            results = self._collection.get(where={"doc_id": doc_id})
            if results and results.get("ids"):
                self._collection.delete(ids=results["ids"])
                return {"status": "ok", "deleted": len(results["ids"]), "message": f"Deleted {len(results['ids'])} chunks for '{doc_id}'"}
            return {"status": "ok", "deleted": 0, "message": f"No chunks found for '{doc_id}'"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
