"""C.1 — Input PII guardrail: VN regex chain + Presidio NER."""

import re
import time

VN_PII = {
    "CCCD": r"\b\d{12}\b",
    "PHONE_VN": r"(?:\+84|0)\d{9,10}\b",
    "TAX_CODE": r"\b\d{10}(?:-\d{3})?\b",
    "EMAIL": r"\b[\w.\-]+@[\w.\-]+\.\w+\b",
}


class InputGuard:
    """PII redaction: VN regex first (fast, deterministic), then Presidio NER (catches names)."""

    def __init__(self, use_presidio: bool = True):
        self.use_presidio = use_presidio
        self._analyzer = None
        self._anonymizer = None
        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except Exception as e:
                print(f"[InputGuard] Presidio unavailable: {e}. Fall back to regex-only.")
                self.use_presidio = False

    def scrub_vn(self, text: str) -> str:
        for name, pattern in VN_PII.items():
            text = re.sub(pattern, f"[{name}]", text)
        return text

    def scrub_ner(self, text: str) -> str:
        if not self.use_presidio or not text.strip():
            return text
        try:
            results = self._analyzer.analyze(text=text, language="en")
            return self._anonymizer.anonymize(text=text, analyzer_results=results).text
        except Exception:
            return text

    def sanitize(self, text: str) -> tuple[str, float]:
        """Full pipeline. Returns (scrubbed_text, latency_ms)."""
        start = time.perf_counter()
        out = self.scrub_ner(self.scrub_vn(text))
        latency_ms = (time.perf_counter() - start) * 1000
        return out, latency_ms


def _build_test_set():
    return [
        "Hi, I'm John Smith from Microsoft. Email: john@ms.com",
        "Call me at +1-555-1234 or visit 123 Main Street, NYC",
        "Số CCCD của tôi là 012345678901",
        "Liên hệ qua 0987654321 hoặc tax 0123456789-001",
        "Customer Nguyễn Văn A, CCCD 098765432101, phone 0912345678",
        "",
        "Just a normal question",
        "A" * 5000,
        "Lý Văn Bình ở 123 Lê Lợi",
        "tax_code:0123456789-001 cccd:012345678901",
    ]


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    guard = InputGuard()
    rows = []
    test_set = _build_test_set()
    for inp in test_set:
        out, ms = guard.sanitize(inp)
        pii_found = []
        for name, pat in VN_PII.items():
            if re.search(pat, inp):
                pii_found.append(name)
        rows.append({
            "input": inp[:100],
            "output": out[:100],
            "pii_found": ",".join(pii_found) or "-",
            "latency_ms": round(ms, 2),
        })

    df = pd.DataFrame(rows)
    out_path = Path(__file__).parent / "pii_test_results.csv"
    df.to_csv(out_path, index=False)

    p95 = df["latency_ms"].quantile(0.95)
    inputs_with_pii = [r for r in rows if r["pii_found"] != "-"]
    if inputs_with_pii:
        detection = sum(1 for r in inputs_with_pii if r["output"] != r["input"]) / len(inputs_with_pii)
    else:
        detection = 0
    print(f"Latency P95: {p95:.1f} ms (target < 50)")
    print(f"Detection rate (regex PII): {detection:.1%} (target >= 80%)")
    print(f"Saved {out_path}")


# === TopicGuard (C.2) — embedding similarity ===

import numpy as np


class TopicGuard:
    """Topic scope validator via embedding cosine sim."""

    def __init__(self, allowed_topics: list[str], threshold: float = 0.6):
        from langchain_openai import OpenAIEmbeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.topics = allowed_topics
        self.topic_vecs = [np.array(self.embeddings.embed_query(t)) for t in allowed_topics]
        self.threshold = threshold

    def check(self, text: str) -> tuple[bool, str, float]:
        """Returns (ok, reason, latency_ms)."""
        start = time.perf_counter()
        if not text.strip():
            return True, "empty input", 0.0
        q_vec = np.array(self.embeddings.embed_query(text))
        sims = [float(np.dot(q_vec, tv) / (np.linalg.norm(q_vec) * np.linalg.norm(tv)))
                for tv in self.topic_vecs]
        max_sim = max(sims)
        best = self.topics[sims.index(max_sim)]
        ok = max_sim > self.threshold
        latency_ms = (time.perf_counter() - start) * 1000
        reason = f"On topic: {best} (sim={max_sim:.2f})" if ok else f"Off topic. Closest: {best} ({max_sim:.2f})"
        return ok, reason, latency_ms
