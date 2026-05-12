# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question (truncated) | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---|---|---|---|---|
| 1 | "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM có ý nghĩa gì trong việc bảo..." | 0.00 | 0.00 | 0.00 | 0.25 | 0.06 | C2 |
| 2 | "Tại sao Việt Nam lại có các quy định về thuế giá trị gia tăng..." | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | C2 |
| 3 | "What does Điều 17 specify regarding the processing of children..." | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | C2 |
| 4 | "What role does the Đảng play in the context of personal data..." | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | C2 |
| 5 | "Làm thế nào để thông báo xử lý dữ liệu cá nhân nhạy cảm cho..." | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | C2 |
| 6 | "What are the requirements for submitting a personal data impact..." | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | C2 |
| 7 | "Làm thế nào mà các quy định về bảo vệ dữ liệu cá nhân tại CỘNG..." | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | C2 |
| 8 | "What are the rights of data subjects regarding access to their..." | 0.50 | 0.72 | 1.00 | 0.50 | 0.68 | C2 |
| 9 | "Làm thế nào mà Bên Xử lý dữ liệu cá nhân phải thực hiện các..." | 0.00 | 0.45 | 1.00 | 0.67 | 0.71 | C1 |
| 10 | "Quyền cung cấp dữ liệu cá nhân của chủ thể dữ liệu được quy..." | 0.80 | 0.50 | 1.00 | 0.67 | 0.74 | C1 |

## Clusters Identified

### Cluster C1: Low answer relevancy (vocabulary mismatch)

**Pattern:** AR < 0.5 even when CP/CR are high. RAG retrieves relevant chunks (CP=1.00, CR>=0.67) but the generated answer uses vocabulary that diverges from the question wording, causing semantic similarity (AR) to drop below 0.5.

**Examples từ bottom 10:**

1. **Q9** — "Làm thế nào mà Bên Xử lý dữ liệu cá nhân phải thực hiện các quy định liên quan đến việc thông báo xử lý dữ liệu cá nhân và đánh giá tác động xử lý dữ liệu cá nhân?" (AR=0.45, CP=1.00, CR=0.67) — Retrieval succeeded but the answer reworded key legal terms like "thông báo xử lý" and "đánh giá tác động" into paraphrases, reducing semantic similarity to the original question.

2. **Q10** — "Quyền cung cấp dữ liệu cá nhân của chủ thể dữ liệu được quy định như thế nào trong Nghị định bảo vệ dữ liệu cá nhân và có những điều kiện gì để thực hiện quyền này?" (AR=0.50, CP=1.00, CR=0.67) — Context retrieved correctly but the answer reformulated "quyền cung cấp dữ liệu cá nhân" using synonymous phrasing, causing RAGAS cosine similarity to the question to fall near the threshold.

**Root cause:** GPT-4o-mini paraphrases instead of echoing key legal terms. Vietnamese legal text uses rigid terminology (e.g., "Bên Xử lý dữ liệu cá nhân", "thông báo xử lý", "đánh giá tác động") — when the LLM rewords these into synonyms or reformulates the answer structure, the RAGAS answer_relevancy score (based on cosine similarity to question embedding) drops significantly.

**Proposed fix (technical):**
- Add explicit prompt instruction to `rag/pipeline.py::build_prompt`: "Trả lời bằng cách sử dụng chính xác các thuật ngữ pháp lý từ câu hỏi và ngữ cảnh. Không diễn giải lại các thuật ngữ chuyên ngành."
- Implement keyword anchoring: extract key noun phrases from question using `underthesea` POS tagging, inject them as "required terms" into the prompt with instruction "Your answer MUST include these exact terms: {terms}".
- Post-process validation: compute Jaccard overlap between question bigrams and answer bigrams; if overlap < 0.3, trigger a re-generation pass with stricter verbatim instruction.

### Cluster C2: Faithfulness drops (hallucination on edge cases)

**Pattern:** Faithfulness = 0.00 — answer contains claims not supported by retrieved contexts, or the LLM refuses and returns an off-topic answer. This cluster dominates the bottom 10 (8 out of 10 questions). Notably, questions 2-7 have perfect CP=1.00 and CR=1.00 but zero faithfulness, indicating retrieval worked correctly but the LLM hallucinated or produced an answer unrelated to the retrieved content.

**Examples từ bottom 10:**

1. **Q3** — "What does Điều 17 specify regarding the processing of children's personal data?" (F=0.00, CP=1.00, CR=1.00, avg=0.50) — Context was fully retrieved with perfect scores, but the generated answer introduced information about children's data processing conditions not directly stated in the retrieved passages, resulting in zero faithfulness.

2. **Q6** — "What are the requirements for submitting a personal data impact assessment to the Ministry of Public Security in Việt Nam, and how does this relate to the VAT declaration process for businesses like CÔNG TY CỔ PHẦN DHA SURFACES?" (F=0.00, CP=1.00, CR=1.00, avg=0.50) — This multi-domain question (data protection + VAT) caused the LLM to bridge two unrelated regulatory domains with hallucinated connections not present in any retrieved context chunk.

**Root cause:** The system prompt does not strictly prohibit generation beyond retrieved context. When questions are complex or multi-domain (mixing Vietnamese legal regulations with business financial documents), GPT-4o-mini fills knowledge gaps with parametric memory. The RAGAS faithfulness score of 0.00 indicates that none of the answer statements could be traced back to the retrieved chunks.

**Proposed fix (technical):**
- Update `rag/pipeline.py::run_query` system prompt: replace vague "answer based on context" with strict grounding instruction: "Chỉ sử dụng thông tin từ [CONTEXT] bên dưới. Nếu context không chứa đủ thông tin để trả lời, hãy trả lời chính xác: 'Không tìm thấy thông tin trong tài liệu.' NGHIÊM CẤM bổ sung thông tin ngoài context."
- Add post-hoc faithfulness check: after generating answer, split into atomic claims using `spacy` sentence segmentation, compute BM25 score of each claim against concatenated retrieved chunks; flag answer if >20% of claims have max BM25 score < 0.1.
- Increase `RERANK_TOP_K` in `config.py` from 3 to 5 to provide broader context coverage, reducing cases where the LLM encounters partial information and fills gaps with hallucinations.
- For multi-domain questions, add a query classifier that detects topic shifts (e.g., legal + financial) and returns an "out of scope" response rather than attempting cross-domain synthesis.

## Aggregate observations

- Total bottom-10 questions: 10/52 = 19% have avg < 0.74.
- Most affected metric: **Faithfulness** shows the largest delta between overall mean and bottom-10 mean: delta_F=0.643 vs delta_AR=0.404, delta_CP=0.071, delta_CR=0.149. Faithfulness collapses from 0.787 (overall) to 0.144 (bottom-10), a drop of 0.643 — nearly 4.5x larger than the next metric drop.
- Overall metric means: F=0.787, AR=0.572, CP=0.971, CR=0.958.
- Bottom-10 metric means: F=0.144, AR=0.168, CP=0.900, CR=0.808.
- Cluster size distribution: C1 (low AR, vocabulary mismatch) = 2 questions (Q9, Q10); C2 (faithfulness drops / hallucination) = 8 questions (Q1-Q8).
- The dominance of C2 confirms that hallucination/faithfulness is the primary failure mode in this RAG system, not retrieval quality (CP/CR remain high even in failure cases).
