from typing import Dict, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TextEmbedder:
    """
    Embeds instruction text using a local transformer model.
    Caches embeddings to avoid redundant computation.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, model_name: Optional[str] = None, use_cache: bool = True):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.use_cache = use_cache
        self._cache: Dict[str, np.ndarray] = {}
        self._model: Optional["SentenceTransformer"] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for text embedding. "
                "Install with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(self.model_name)
        self._initialized = True

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string. Returns cached result if available.
        """
        if self.use_cache and text in self._cache:
            return self._cache[text]

        self._ensure_initialized()

        embedding = self._model.encode(text, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)

        if self.use_cache:
            self._cache[text] = embedding

        return embedding

    def embed_batch(self, texts: list) -> np.ndarray:
        """
        Embed multiple texts at once. More efficient than individual calls.
        """
        self._ensure_initialized()

        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if self.use_cache and text in self._cache:
                results.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts in batch
        if uncached_texts:
            new_embeddings = self._model.encode(uncached_texts, convert_to_numpy=True)
            new_embeddings = new_embeddings.astype(np.float32)

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                if self.use_cache:
                    self._cache[text] = emb
                results.append((idx, emb))

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.stack([emb for _, emb in results])

    def get_zero_embedding(self) -> np.ndarray:
        """Return a zero embedding for when no intent is active."""
        return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    @property
    def cache_size(self) -> int:
        return len(self._cache)
