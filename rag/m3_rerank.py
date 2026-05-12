"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._backend = None  # "flag" hoặc "crossencoder"

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self.model_name, use_fp16=True)
            self._backend = "flag"
        except ImportError:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._backend = "crossencoder"
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-N → top-k bằng cross-encoder."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]

        if self._backend == "flag":
            raw_scores = model.compute_score(pairs)
            if not isinstance(raw_scores, list):
                raw_scores = [raw_scores]
        else:
            raw_scores = model.predict(pairs)

        scored = sorted(
            zip((float(s) for s in raw_scores), documents),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
                rank=rank,
            )
            for rank, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional — chỉ chạy khi `flashrank` đã cài."""
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        try:
            from flashrank import RerankRequest
        except ImportError:
            return []

        model = self._load_model()
        passages = [
            {"id": i, "text": d["text"], "meta": d.get("metadata", {})}
            for i, d in enumerate(documents)
        ]
        ranked = model.rerank(RerankRequest(query=query, passages=passages))

        results = []
        for rank, item in enumerate(ranked[:top_k]):
            idx = item.get("id", rank)
            doc = documents[idx] if idx < len(documents) else documents[rank]
            results.append(RerankResult(
                text=item.get("text", doc["text"]),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(item.get("score", 0.0)),
                metadata=doc.get("metadata", {}),
                rank=rank,
            ))
        return results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Đo latency rerank qua n_runs (ms). Lần đầu warm-up có thể chậm."""
    if n_runs <= 0:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "n_runs": 0}

    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "n_runs": n_runs,
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
