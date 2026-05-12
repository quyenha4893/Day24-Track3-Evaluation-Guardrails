"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


def _get_openai_client():
    """Return OpenAI client if API key is available, else None."""
    if (
        not OPENAI_API_KEY
        or OPENAI_API_KEY.startswith("sk-")
        and len(OPENAI_API_KEY) < 20
    ):
        return None
    try:
        from openai import OpenAI

        return OpenAI()
    except Exception:
        return None


def _llm_call(
    system_prompt: str, user_content: str, max_tokens: int = 200
) -> str | None:
    """Call OpenAI LLM with fallback to None."""
    client = _get_openai_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    summary = _llm_call(
        system_prompt="Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
        user_content=text,
        max_tokens=150,
    )
    if summary:
        return summary

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text

    n = min(2, len(sentences))
    return ". ".join(sentences[:n]) + "."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    response = _llm_call(
        system_prompt=f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
        user_content=text,
        max_tokens=200,
    )
    if response:
        questions = [q.strip() for q in response.split("\n") if q.strip()]
        questions = [re.sub(r"^[\d\.\-\)\s]+", "", q).strip() for q in questions]
        return questions

    topics = []
    keywords = [
        (r"nghỉ phép", "Nhân viên được nghỉ phép bao nhiêu ngày?"),
        (r"mật khẩu", "Mật khẩu cần thay đổi sau bao lâu?"),
        (r"lương|thu nhập", "Mức lương được quy định như thế nào?"),
        (r"thử việc", "Thời gian thử việc là bao lâu?"),
        (r"bảo hiểm", "Chính sách bảo hiểm được quy định ra sao?"),
        (r"ngày làm việc", "Thời gian làm việc quy định như thế nào?"),
        (r"thâm niên", "Thâm niên công tác được tính như thế nào?"),
    ]
    for pattern, question in keywords:
        if re.search(pattern, text, re.IGNORECASE):
            topics.append(question)

    if topics:
        return topics[:n_questions]

    if text.strip():
        return ["Đoạn văn này nói về điều gì?"]
    return []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    context = _llm_call(
        system_prompt="Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu.",
        user_content=f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    if context:
        return f"{context}\n\n{text}"

    if document_title:
        return f"[Trích từ tài liệu: {document_title}] {text}"

    return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    response = _llm_call(
        system_prompt='Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
        user_content=text,
        max_tokens=150,
    )
    if response:
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

    metadata = {
        "topic": "",
        "entities": [],
        "category": "policy",
        "language": "vi",
    }

    categories = {
        "hr": [
            r"nhân viên",
            r"nghỉ phép",
            r"lương",
            r"thâm niên",
            r"thử việc",
            r"bảo hiểm",
            r"phúc lợi",
        ],
        "it": [
            r"mật khẩu",
            r"phần mềm",
            r"máy tính",
            r"mạng",
            r"vpn",
            r"email",
            r"bảo mật",
        ],
        "finance": [r"tài chính", r"ngân sách", r"chi phí", r"thanh toán", r"hóa đơn"],
        "policy": [r"quy định", r"chính sách", r"nội quy", r"điều lệ"],
    }
    for cat, patterns in categories.items():
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            metadata["category"] = cat
            break

    text_lower = text.lower()
    if any(w in text_lower for w in ["nghỉ", "phép", "lương", "viên"]):
        metadata["language"] = "vi"
    else:
        metadata["language"] = "vi"

    entities = re.findall(
        r"(?:VinUni|VinGroup|Hà Nội|TP\.?HCM|Việt Nam|\d{1,2}/\d{1,2}/\d{4})", text
    )
    metadata["entities"] = entities

    first_line = text.split("\n")[0].strip()
    metadata["topic"] = first_line[:80] if first_line else text[:80]

    return metadata


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    for chunk in chunks:
        summary = ""
        questions = []
        enriched_text = chunk["text"]
        auto_meta = {}

        if "summary" in methods or "full" in methods:
            summary = summarize_chunk(chunk["text"])

        if "hyqa" in methods or "full" in methods:
            questions = generate_hypothesis_questions(chunk["text"])

        if "contextual" in methods or "full" in methods:
            enriched_text = contextual_prepend(
                chunk["text"],
                chunk.get("metadata", {}).get("source", ""),
            )

        if "metadata" in methods or "full" in methods:
            auto_meta = extract_metadata(chunk["text"])

        enriched.append(
            EnrichedChunk(
                original_text=chunk["text"],
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**chunk.get("metadata", {}), **auto_meta},
                method="+".join(methods),
            )
        )

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
