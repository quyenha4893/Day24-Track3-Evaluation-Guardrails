"""A.3 — Identify bottom 10 questions, cluster failures, propose fixes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import PHASE_A_DIR


def main():
    df = pd.read_csv(PHASE_A_DIR / "ragas_results.csv")
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    df["avg"] = df[metrics].mean(axis=1)
    bottom = df.nsmallest(10, "avg").copy()

    table_rows = []
    for i, row in enumerate(bottom.itertuples(), 1):
        q = str(row.user_input)[:60].replace("|", "/").replace("\n", " ")
        f_val = row.faithfulness if not pd.isna(row.faithfulness) else 0.0
        table_rows.append(
            f"| {i} | \"{q}...\" | {f_val:.2f} | "
            f"{row.answer_relevancy:.2f} | {row.context_precision:.2f} | "
            f"{row.context_recall:.2f} | {row.avg:.2f} | C? |"
        )

    md = f"""# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question (truncated) | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---|---|---|---|---|
{chr(10).join(table_rows)}

## Clusters Identified

### Cluster C1: Low answer relevancy (vocabulary mismatch)

**Pattern:** AR < 0.5 even when CP/CR cao. RAG returns relevant chunks but answer uses vocabulary lệch khỏi question wording → semantic similarity (AR) drops.

**Examples từ bottom 10:** (fill 2 examples)

**Root cause:** GPT-4o-mini paraphrases instead of echoing key terms. Vietnamese legal text dùng terminology cứng → mismatch khi LLM rewords.

**Proposed fix (technical):**
- Add prompt instruction: "Use key terms verbatim from context where possible."
- Lower temperature từ default → 0.0 (đã làm) + add few-shot example showing verbatim quoting.
- Post-process: extract key noun phrases từ question → check answer contains them; if not, regenerate.

### Cluster C2: Faithfulness drops (hallucination on edge cases)

**Pattern:** Faithfulness < 0.6 — answer contains facts not in retrieved contexts.

**Examples từ bottom 10:** (fill 2 examples)

**Root cause:** Prompt không strict enough về "chỉ trả lời từ context". GPT-4o-mini fills gaps with prior knowledge khi context mơ hồ hoặc cụt.

**Proposed fix (technical):**
- Update prompt trong `rag/pipeline.py::run_query`: thêm "Nếu context không đủ, trả lời: 'Không có thông tin trong tài liệu.' Cấm bịa thêm thông tin ngoài context."
- Add post-hoc check: tokenize answer, compute % tokens appearing in concatenated contexts; flag if < 60%.
- Tăng `RERANK_TOP_K` từ 3 → 5 để giảm thiếu context.

## Aggregate observations

- Total bottom-10 questions: 10/52 = {10/52*100:.0f}% có avg < {bottom['avg'].max():.2f}.
- Most affected metric: (fill — check df[metrics].mean() vs bottom[metrics].mean() delta)
- Cluster size distribution: (fill based on assignments)
"""
    out = PHASE_A_DIR / "failure_analysis.md"
    out.write_text(md, encoding="utf-8")
    print(f"Written {out}")
    print("\n=== BOTTOM 10 (review needed) ===")
    # Use sys.stdout with utf-8 for Windows compatibility
    import io
    buf = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    buf.write(bottom[["user_input", "avg"] + metrics].to_string() + "\n")
    buf.flush()


if __name__ == "__main__":
    main()
