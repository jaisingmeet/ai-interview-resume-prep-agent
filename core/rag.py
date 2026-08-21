from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - deployment installs faiss-cpu.
    faiss = None


class InterviewRetriever:
    """Semantic-ish local retriever that uses FAISS when available.

    TF-IDF vectors are normalized, so inner product is cosine similarity. The
    NumPy fallback keeps tests and minimal environments usable without hiding
    the production FAISS dependency.
    """

    def __init__(self, records: list[dict[str, Any]] | None = None):
        self.records = records or []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        self.matrix: np.ndarray | None = None
        self.index = None
        if self.records:
            self.build(self.records)

    def build(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        texts = [self._record_text(record) for record in records]
        self.matrix = self.vectorizer.fit_transform(texts).astype(np.float32).toarray()
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.where(norms == 0, 1, norms)
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)

    @staticmethod
    def _record_text(record: dict[str, Any]) -> str:
        return " ".join(
            str(record.get(key, ""))
            for key in ("question", "answer_framework", "topic", "domain", "difficulty", "tags")
        )

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if not self.records or self.matrix is None:
            return []
        q = self.vectorizer.transform([query]).astype(np.float32).toarray()
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.where(norm == 0, 1, norm)
        limit = min(max(k, 1), len(self.records))
        if self.index is not None:
            scores, indices = self.index.search(q, limit)
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:
            scores = (self.matrix @ q[0]).tolist()
            indices = np.argsort(scores)[::-1][:limit].tolist()
            pairs = ((idx, scores[idx]) for idx in indices)
        results = []
        for idx, score in pairs:
            record = dict(self.records[idx])
            record["similarity"] = round(float(score), 4)
            results.append(record)
        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "records.json").open("w", encoding="utf-8") as handle:
            json.dump(self.records, handle, indent=2, ensure_ascii=False)
        with (directory / "vectorizer.json").open("w", encoding="utf-8") as handle:
            json.dump({"vocabulary": {str(key): int(value) for key, value in self.vectorizer.vocabulary_.items()}, "idf": self.vectorizer.idf_.tolist()}, handle)
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))

    @classmethod
    def load(cls, directory: str | Path) -> "InterviewRetriever":
        directory = Path(directory)
        with (directory / "records.json").open(encoding="utf-8") as handle:
            records = json.load(handle)
        return cls(records)
