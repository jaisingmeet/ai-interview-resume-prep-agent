from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # Optional acceleration; cloud-safe fallback needs no compiled packages.
    faiss = None
    np = None


class InterviewRetriever:
    """Local interview retriever with an optional FAISS accelerator.

    The default path uses a small standard-library TF-IDF implementation. This
    is intentionally CPU-friendly and avoids forcing Streamlit Cloud to compile
    SciPy/scikit-learn. If FAISS and NumPy are installed, normalized vectors are
    also loaded into a FAISS inner-product index.
    """

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what", "when", "why", "with", "would"
    }

    def __init__(self, records: list[dict[str, Any]] | None = None):
        self.records = records or []
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.matrix: list[list[float]] = []
        self.index = None
        if self.records:
            self.build(self.records)

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in cls.STOP_WORDS and len(token) > 1]

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        values = [record.get(key, "") for key in ("question", "answer_framework", "topic", "domain", "difficulty", "tags")]
        return " ".join(" ".join(value) if isinstance(value, list) else str(value) for value in values)

    def _vectorize(self, text: str) -> list[float]:
        counts = Counter(self._tokens(text))
        total = max(sum(counts.values()), 1)
        vector = [0.0] * len(self.vocabulary)
        for token, count in counts.items():
            if token in self.vocabulary:
                vector[self.vocabulary[token]] = (count / total) * self.idf[self.vocabulary[token]]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def build(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        tokenized = [self._tokens(self._record_text(record)) for record in records]
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(document_frequency))}
        documents = len(records)
        self.idf = [math.log((1 + documents) / (1 + document_frequency[token])) + 1 for token in sorted(document_frequency)]
        self.matrix = [self._vectorize(self._record_text(record)) for record in records]
        if faiss is not None and np is not None and self.matrix:
            vectors = np.asarray(self.matrix, dtype="float32")
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.records or not self.matrix:
            return []
        limit = min(max(k, 1), len(self.records))
        query_vector = self._vectorize(query)
        if self.index is not None and np is not None:
            scores, indices = self.index.search(np.asarray([query_vector], dtype="float32"), limit)
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:
            scored = [(sum(a * b for a, b in zip(vector, query_vector)), idx) for idx, vector in enumerate(self.matrix)]
            pairs = ((idx, score) for score, idx in sorted(scored, reverse=True)[:limit])
        results = []
        for idx, score in pairs:
            record = dict(self.records[idx])
            record["similarity"] = round(float(score), 4)
            results.append(record)
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "records.json").write_text(json.dumps(self.records, indent=2, ensure_ascii=False), encoding="utf-8")
        (directory / "vectorizer.json").write_text(json.dumps({"vocabulary": self.vocabulary, "idf": self.idf}), encoding="utf-8")
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))
