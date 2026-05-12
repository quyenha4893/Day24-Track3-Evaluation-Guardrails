# Production RAG Is Not Just a Pipeline — It's a Defense System

*by Nguyễn Việt Long · Lab 24: Eval, Guardrails & Monitoring · VinUni Track 3*

---

## Ba câu hỏi mà bất kỳ production RAG nào cũng phải trả lời

Trước khi ship một RAG chatbot vào production, hãy tự hỏi:

1. **"Hệ thống trả lời đúng đến đâu?"** — Bạn cần một bộ eval tự động, không phải chỉ nhìn vào vài example thủ công.
2. **"Ai đánh giá chất lượng câu trả lời — và liệu người đánh giá đó có đáng tin không?"** — LLM-as-Judge rất tiện, nhưng bị bias theo position và length.
3. **"Khi có kẻ tấn công, hoặc khi Gemini trả về 429 Rate Limit, hệ thống sẽ làm gì?"** — Đây là câu hỏi sống còn của production.

Lab 24 là bài thực hành trả lời cả ba câu hỏi trên — bằng số thực, không phải lý thuyết.

---

## Phase A: RAGAS trên 52 câu hỏi thực tế

Hệ thống được đánh giá trên **52 câu hỏi** tổng hợp từ tập tài liệu gồm Nghị định 13/2023 (bảo vệ dữ liệu cá nhân) và báo cáo tài chính doanh nghiệp Việt Nam. Pipeline RAG sử dụng Hybrid Search (BM25 + BGE-M3 Dense), CrossEncoder reranker, và GPT-4o-mini làm generator.

| Metric | Score | Ý nghĩa |
|---|---|---|
| **Faithfulness (F)** | **0.787** | 78.7% câu trả lời không hallucinate so với context |
| **Answer Relevancy (AR)** | **0.572** | Chỉ 57.2% — câu trả lời lạc đề hoặc quá chung |
| **Context Precision (CP)** | **0.971** | Retrieval rất chính xác — hầu hết đoạn retrieved là liên quan |
| **Context Recall (CR)** | **0.958** | Hầu như không bỏ sót thông tin cần thiết |

**Nhận xét thực tế:** Retrieval (CP=0.971, CR=0.958) tốt hơn nhiều so với generation. Answer Relevancy 0.572 thấp do các câu tài chính nhận câu trả lời "Không tìm thấy thông tin" — kéo điểm xuống. Faithfulness 0.787 chưa đạt SLO 0.85 — cần cải thiện prompt hoặc tăng top-k.

---

## Phase B: LLM-as-Judge — Tiện lợi nhưng cần biết giới hạn

GPT-4o được dùng để so sánh pairwise giữa hai phiên bản câu trả lời (A=full RAG vs B=degraded). Kết quả trên **30 cặp**:

- **A thắng:** 20/30 (66.7%)
- **B thắng:** 3/30 (10%)
- **Tie:** 7/30 (23.3%)
- **Absolute mean score:** 4.16/5

Nghe có vẻ tốt. Nhưng khi kiểm tra bias:

```
Position bias: A wins as first position = 23/30 = 76.7%
Length bias: when B is longer, B wins 75% of time
Cohen's kappa (LLM vs Human) = 0.333  →  "Fair" agreement
```

**Cohen's kappa = 0.333** là mức "Fair" — tức là LLM judge và human annotator chỉ đồng ý với nhau ở mức trung bình. Kappa < 0.4 là dấu hiệu cần cảnh báo trong production.

Position bias 76.7% nghĩa là judge có xu hướng chọn câu trả lời được đặt trước (position A), bất kể chất lượng thực sự. Đây là lý do tại sao swap-and-average (chạy đánh giá hai lần với thứ tự đảo ngược) là bắt buộc, không phải optional.

**Takeaway:** LLM judge là proxy nhanh — nhưng phải calibrate với human labels và track kappa theo thời gian. Đừng dùng nó như oracle.

---

## Phase C: 4-Layer Defense in Depth

Đây là phần thú vị nhất. Hệ thống có **4 lớp bảo vệ** hoạt động tuần tự:

```
User Query
    ↓
[L1] Input Guards (~280 ms avg)
    ├── PII Detector (Presidio + regex VN)
    ├── TopicGuard (BGE-M3 cosine similarity)
    └── Adversarial Detector
    ↓
[L2] RAG Pipeline (~1,500–2,500 ms)
    ├── Hybrid Retrieval (BM25 + Dense)
    ├── CrossEncoder Reranker
    └── GPT-4o-mini Generator
    ↓
[L3] Output Guard (Gemini 1.5 Flash)
    └── ShieldGemma taxonomy check
    ↓
[L4] Audit & Observability
```

### L1a: PII Detection — 100% detection rate tại 0.2ms P95

Regex patterns cho Vietnamese PII (CCCD, MST, phone) + Presidio cho email/phone quốc tế. Kết quả:

```
10/10 PII inputs detected and redacted
P95 latency = 0.21 ms   ← negligible overhead
```

Ví dụ thực tế từ test:
```
Input:  "Hi, I'm John Smith. Email: john@ms.com"
Output: "Hi, I'm John Smith. Email: [EMAIL]"

Input:  "Số CCCD của tôi là 012345678901"
Output: "Số CCCD của tôi là [CCCD]"
```

### L1b: TopicGuard — 90% accuracy tại 220ms

BGE-M3 embedding + cosine similarity với threshold = 0.4. 18/20 correct on test set. Hai false negatives là câu tài chính rất ngắn ("Lãi gộp quý 4?") mà embedding không capture được.

### L1c: Adversarial Detection — 100% detection, 10% FP

20/20 DAN attacks, jailbreaks, encoding attacks bị blocked. FP = 10% (2 legitimate queries blocked nhầm) — chấp nhận được nếu có human-in-the-loop review.

### L3: Output Guard với Gemini 1.5 Flash

Gemini được dùng thay thế cho Llama Guard (không available trực tiếp), classify theo taxonomy S1-S6 (violence, self-harm, dangerous goods, hate speech, sexual content, privacy). Kết quả: **100% detection** trên unsafe outputs.

---

## Fail-CLOSED Policy — Quyết định cứu cả hệ thống

Ngày test, Gemini 1.5 Flash bị rate-limit (HTTP 429). Hệ thống chọn **fail-CLOSED** thay vì fail-open. Kết quả: 4/10 safe inputs bị blocked nhầm (FP tăng 40% tạm thời), nhưng không có unsafe output nào thoát ra.

```python
# output_guard.py — fail-closed logic
try:
    result = await gemini_classify(response)
except RateLimitError:
    return SafeRefusal(
        message="Hệ thống đang bảo trì. Vui lòng thử lại sau.",
        blocked=True,
        reason="output_guard_unavailable"
    )
```

Trong production: một ngày có FP cao còn tốt hơn một lần unsafe output thoát ra. Security > Availability.

---

## Latency Reality Check

Đo thực tế cho thấy bottleneck không phải là guard, mà là RAG:

| Layer | Latency |
|---|---|
| L1 PII | P95 = 0.2 ms |
| L1 TopicGuard | avg ≈ 220 ms |
| L1 Adversarial | avg ≈ 250 ms |
| **L1 Total (parallel asyncio)** | **avg ≈ 280 ms** |
| L2 RAG Pipeline | P95 ≈ 1,500–2,500 ms |
| L3 Output Guard | avg ≈ 2,000 ms (safe path) |
| **E2E P95** | **≤ 5,000 ms** |

Key insight: L1 chạy parallel bằng `asyncio.gather()` — overhead tính max, không tính sum. L2 RAG là dominant latency; CrossEncoder reranking tốn phần lớn thời gian.

---

## Takeaway: Production RAG = 4 Layers + Bias-Aware Judge + Monitoring SLO

Sau Lab 24, ba bài học quan trọng nhất:

1. **RAGAS không phải điểm thi — là monitoring metric.** Faithfulness 0.787 dưới SLO 0.85 nghĩa là phải action ngay: tăng top-k, cải thiện chunking, hoặc rewrite prompt.

2. **LLM Judge hữu ích nhưng cần calibrate.** Kappa 0.333 với human là "Fair" — không đủ tin cậy để auto-deploy. Cần weekly human spot-check và track kappa drift.

3. **Fail-CLOSED là default đúng đắn.** Khi không chắc (rate limit, timeout, error), hệ thống nên từ chối an toàn thay vì đoán bừa. 10% FP tạm thời tốt hơn 1% unsafe output vĩnh viễn.

**Production checklist tối thiểu:** RAGAS eval tự động mỗi deploy; LLM Judge + kappa tracking weekly; 4-layer guardrail với fail-closed; SLO dashboard P50/P95/P99; budget alert ($79/month baseline).

---

Code: [lab24-eval-guardrails-NguyenVietLong on GitHub](#)
