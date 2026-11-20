"""Turn answers into vectors, so answers with the same idea end up close together.

Two embedders share one interface:
- MiniLMEmbedder: a sentence-transformer model that understands meaning (default, runs locally).
- GeminiEmbedder: Google's hosted embedding model (used on the live site, no PyTorch needed).
- TfidfEmbedder: word-overlap vectors. No download needed; used in tests and as a baseline.

Both return L2-normalized vectors, so a dot product is cosine similarity.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from unmuddle.config import EMBEDDING_MODEL


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) array of unit-length vectors."""
        ...


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class TfidfEmbedder:
    """Fits on the texts it is given. Fine for clustering one question's answers at a time."""

    name = "tfidf"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform(texts).toarray()
        except ValueError:  # every word was a stop word
            return np.zeros((len(texts), 1))
        return normalize(matrix)


class GeminiEmbedder:
    """Google's hosted embedding model. No local model, so it suits small servers."""

    name = "gemini"
    batch_size = 100  # API limit per request

    def __init__(self, model: str = "gemini-embedding-001", dimensions: int = 768, api_key: str | None = None):
        import os

        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        from google.genai import types

        if not texts:
            return np.zeros((0, self.dimensions))
        config = types.EmbedContentConfig(task_type="CLUSTERING", output_dimensionality=self.dimensions)
        rows = []
        for i in range(0, len(texts), self.batch_size):
            result = self.client.models.embed_content(
                model=self.model, contents=texts[i : i + self.batch_size], config=config
            )
            rows.extend(e.values for e in result.embeddings)
        # Shorter-than-full Gemini vectors are not unit length, so normalize them.
        return normalize(np.array(rows, dtype=float))


def get_embedder(name: str = "minilm") -> Embedder:
    if name == "minilm":
        return MiniLMEmbedder()
    if name == "gemini":
        return GeminiEmbedder()
    if name == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedder: {name}")
