from __future__ import annotations

import json
from pathlib import Path

from core.rag import InterviewRetriever

ROOT = Path(__file__).parent
records = json.loads((ROOT / "data" / "knowledge_base.json").read_text(encoding="utf-8"))
retriever = InterviewRetriever(records)
retriever.save(ROOT / "data" / "vector_store")
print(f"Indexed {len(records)} records")
