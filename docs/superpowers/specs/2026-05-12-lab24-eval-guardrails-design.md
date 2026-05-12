# Lab 24 — Full Evaluation & Guardrail System — Design Spec

- **Project:** `lab24-eval-guardrails-NguyenVietLong`
- **Author:** Nguyễn Việt Long (2A202600242)
- **Date:** 2026-05-12
- **Source lab:** `lab24-student-edition.pdf` (AICB-P2T3, VinUniversity, 05/2026)
- **Upstream:** `Day18-Track3-Production-RAG`
- **Target score:** ~107/115 (Full 4 phases + 3 bonus)

---

## 1. Goals & Non-goals

### Goals

1. Build production-ready **evaluation pipeline** cho RAG từ Day 18 (RAGAS 4 metrics + LLM-as-Judge + Cohen's kappa).
2. Deploy **defense-in-depth 4-layer guardrail stack** (input PII + topic + injection / LLM / output safety / audit log).
3. Measure **end-to-end latency** với P50/P95/P99 cho ≥ 100 requests, đạt L1 P95 < 50ms và L3 P95 < 100ms.
4. Sản xuất **blueprint document** production-ready (SLO + diagram + alert playbook + cost).
5. Cover **3 bonus tasks easy-medium**: Prompt Guard (Meta), NeMo Guardrails, Blog post.

### Non-goals

- Không fine-tune Llama Guard cho tiếng Việt (bonus Very Hard, skip).
- Không build Streamlit dashboard (bonus Medium, skip).
- Không deploy production lên cloud — local benchmark đủ.
- Không re-implement RAG pipeline — reuse Day 18 nguyên trạng.

---

## 2. Constraints

| Constraint | Value |
|---|---|
| API keys available | `OPENAI_API_KEY`, `GEMINI_API_KEY` |
| GPU | Không có |
| Groq | Không có account |
| HuggingFace | Public model only (Prompt-Guard-86M OK, Llama-Guard-3-8B skip) |
| Python | 3.10+ |
| Budget | ≤ $15 USD total |
| Vector DB | Qdrant local (Docker) — kế thừa Day 18 |
| Submission deadline | Day 25 morning (sau Day 24) |
| Late policy | −10%/day, max 3 days |

### Substitutions vs PDF spec

| PDF requirement | Substitute | Lý do |
|---|---|---|
| Llama Guard 3 (self-hosted GPU) hoặc Groq API | `gemini-1.5-flash` với Llama Guard taxonomy prompt | Không có GPU, không có Groq key. Gemini free tier 15 RPM đủ benchmark 100 requests. |
| Llama Guard taxonomy (S1-S13) | Inline trong prompt template | Adapter generate `safe`/`unsafe` same contract. |

Document rõ substitution trong `README.md` § Setup và `phase-c/output_guard.py` docstring.

---

## 3. Architecture

### 3.1 High-level 4-layer stack

```
                        ┌──────────────────────────────────┐
                        │         User Input               │
                        └────────────┬─────────────────────┘
                                     │
                  ┌──────────────────▼────────────────────┐
                  │  L1 — Input Guard (async parallel)    │
                  │  ┌─────────────┐  ┌────────────────┐  │
                  │  │ PII Scrub   │  │ Topic Validator│  │
                  │  │ (Presidio   │  │ (embedding sim │  │
                  │  │ + VN regex) │  │  cosine)       │  │
                  │  └─────────────┘  └────────────────┘  │
                  │  ┌──────────────────────────────────┐ │
                  │  │ Prompt Guard 86M (BONUS, in-line)│ │
                  │  │ Injection classifier             │ │
                  │  └──────────────────────────────────┘ │
                  │  (NeMo bonus — offline compare only,  │
                  │   không tham gia L1 production gather)│
                  └──────────────────┬────────────────────┘
                          off-topic / injection → refuse
                                     │ sanitized text
                  ┌──────────────────▼────────────────────┐
                  │  L2 — RAG Pipeline (from Day18)       │
                  │  HybridSearch (BM25+Dense Qdrant)     │
                  │    → CrossEncoder Rerank → GPT-4o-mini│
                  └──────────────────┬────────────────────┘
                                     │ answer + contexts
                  ┌──────────────────▼────────────────────┐
                  │  L3 — Output Guard (async)            │
                  │  Gemini Safety Adapter                │
                  │  (Llama Guard taxonomy prompt)        │
                  └──────────────────┬────────────────────┘
                          unsafe → refuse_response()
                                     │ safe
                  ┌──────────────────▼────────────────────┐
                  │  L4 — Audit Log (async fire-forget)   │
                  └──────────────────┬────────────────────┘
                                     │
                              Response to User
```

### 3.2 Node table

| Node | File | Input | Output | Sync/Async |
|---|---|---|---|---|
| N1 PII Scrub | `phase-c/input_guard.py::InputGuard.sanitize` | `text: str` | `(scrubbed: str, latency_ms: float)` | Sync → thread |
| N2 Topic Validator | `phase-c/input_guard.py::TopicGuard.check` | `text: str` | `(ok: bool, reason: str, latency_ms: float)` | Sync → thread |
| N3 Prompt Guard (bonus) | `phase-c/prompt_guard.py::PromptGuard.detect` | `text: str` | `(injection_score: float, blocked: bool)` | Sync → thread |
| N4 HybridSearch | `rag/m2_search.py::HybridSearch.search` | `query: str` | `list[SearchResult]` | Sync |
| N5 Reranker | `rag/m3_rerank.py::CrossEncoderReranker.rerank` | `query, docs, top_k=3` | `list[RerankResult]` | Sync |
| N6 LLM Generate | `rag/pipeline.py::run_query` | `query, search, reranker` | `(answer: str, contexts: list[str])` | Sync |
| N7 Output Safety | `phase-c/output_guard.py::OutputGuard.check` | `(user_input, agent_response)` | `(is_safe: bool, verdict: str, latency_ms: float)` | Sync → thread |
| N8 Audit Log | `phase-c/full_pipeline.py::audit_log` | `(input, output, timings, safety)` | JSONL append | Async fire-forget |
| N9 Refuse | `phase-c/full_pipeline.py::refuse_response` | `reason: str` | `str` (graceful msg) | Sync |
| N10 RAGAS Evaluator | `phase-a/run_ragas.py::evaluate` | `Dataset` | `ragas_results.csv + summary.json` | Sync |
| N11 Pairwise Judge | `phase-b/judge.py::pairwise_judge_with_swap` | `(q, ans_a, ans_b, judge_llm)` | `'A'\|'B'\|'tie'` | Sync |
| N12 Absolute Scorer | `phase-b/judge.py::absolute_score` | `(q, answer, judge_llm)` | `dict (4-dim + overall)` | Sync |
| N13 Kappa Analyzer | `phase-b/kappa_analysis.ipynb` | `human_labels.csv + pairwise_results.csv` | kappa + interpretation | Sync |
| N14 Bias Reporter | `phase-b/bias_report.py` | `pairwise_results.csv` | `judge_bias_report.md + chart.png` | Sync |
| N15 Adversarial Tester | `phase-c/adversarial_test.py` | `attacks: list, guards` | `(detection_rate, results.csv)` | Async |
| N16 Latency Benchmark | `phase-c/full_pipeline.py::benchmark` | `n=100 queries` | `latency_benchmark.csv` | Async |
| N17 CI Eval Gate | `.github/workflows/eval-gate.yml` + `scripts/run_eval.py` | RAGAS metrics | exit 0/1 + artifact | — |
| N18 NeMo Adapter (bonus) | `phase-c/nemo_adapter.py::NemoTopicGuard` | `text: str` | `(ok: bool, reason: str)` | Sync → thread |
| N19 Blueprint Generator | `phase-d/build_blueprint.py` | Phase A/B/C outputs | `blueprint.md` | Sync |

### 3.3 Inter-node DAG (runtime)

```
                        ┌──────┐
                        │ user │
                        └───┬──┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
            ┌────┐        ┌────┐        ┌────┐
            │ N1 │        │ N2 │        │ N3 │   L1 parallel
            │PII │        │Top │        │Prom│   (asyncio.gather)
            └─┬──┘        └─┬──┘        └─┬──┘   N3 = bonus, optional
              └────┬────────┴────┬────────┘
                   │ all-ok      │ any-blocked → N9 refuse
                   ▼
                 ┌────┐  HybridSearch
                 │ N4 │
                 └─┬──┘
                   ▼
                 ┌────┐  Reranker
                 │ N5 │
                 └─┬──┘
                   ▼
                 ┌────┐  LLM Generate
                 │ N6 │
                 └─┬──┘
                   ▼
                 ┌────┐  Output Safety
                 │ N7 │
                 └─┬──┘
                   │
            ┌──────┴──────┐
            │ safe        │ unsafe → N9 refuse
            ▼
          ┌────┐  fire-forget   ┌────┐
          │User│ ◄────────────  │ N8 │  Audit Log
          └────┘                └────┘
```

---

## 4. Tech stack

| Layer | Tech |
|---|---|
| Runtime | Python 3.10+, asyncio |
| RAG (L2) | Day18 copy → Qdrant + BAAI/bge-m3 + Cohere rerank + GPT-4o-mini |
| L1 PII | `presidio-analyzer` + `presidio-anonymizer` + VN regex (`re`) |
| L1 Topic | `langchain-openai` `OpenAIEmbeddings` + numpy cosine |
| L1 Injection (bonus) | `transformers` + `meta-llama/Prompt-Guard-86M` |
| L1 NeMo (bonus) | `nemoguardrails` |
| L3 Safety | `google-generativeai` `gemini-1.5-flash` + Llama Guard taxonomy prompt |
| Eval | `ragas==0.2.x`, `datasets`, OpenAI judge, `scikit-learn` `cohen_kappa_score` |
| Plotting | `matplotlib` |
| Async wrappers | native `asyncio` + `asyncio.to_thread` cho sync RAG |
| Cost guard | tiktoken token counter + JSONL cost log |
| Testing | `pytest` (optional smoke tests) |

---

## 5. File layout

```
lab24-eval-guardrails-NguyenVietLong/
├── README.md                          # 200-300 từ overview
├── requirements.txt                   # pinned versions
├── prompts.md                         # AI prompts log (academic integrity)
├── .env.example                       # OPENAI_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, HF_TOKEN
├── .gitignore
│
├── config.py                          # shared config (paths, thresholds, budget)
│
├── rag/                               # copy từ Day18 (L2 RAG)
│   ├── __init__.py
│   ├── m1_chunking.py
│   ├── m2_search.py
│   ├── m3_rerank.py
│   ├── m4_eval.py
│   ├── m5_enrichment.py
│   ├── pipeline.py
│   └── data/                          # nd13_2023.md, BCTC.md
│
├── phase-a/                           # 30 điểm
│   ├── gen_testset.py
│   ├── run_ragas.py
│   ├── analyze_failures.py
│   ├── testset_v1.csv
│   ├── testset_review_notes.md
│   ├── ragas_results.csv
│   ├── ragas_summary.json
│   └── failure_analysis.md
│
├── phase-b/                           # 25 điểm
│   ├── judge.py
│   ├── kappa_analysis.ipynb
│   ├── bias_report.py
│   ├── pairwise_results.csv
│   ├── absolute_scores.csv
│   ├── human_labels.csv
│   ├── to_label.csv
│   ├── judge_bias_report.md
│   └── bias_chart.png
│
├── phase-c/                           # 35 điểm
│   ├── input_guard.py
│   ├── output_guard.py
│   ├── prompt_guard.py                # bonus
│   ├── nemo_adapter.py                # bonus
│   ├── adversarial_test.py
│   ├── full_pipeline.py
│   ├── pii_test_results.csv
│   ├── adversarial_test_results.csv
│   ├── topic_test_results.csv
│   ├── output_guard_test_results.csv
│   └── latency_benchmark.csv
│
├── phase-d/                           # 10 điểm
│   ├── build_blueprint.py
│   └── blueprint.md
│
├── bonus/
│   ├── blog_post.md
│   ├── bonus_prompt_guard.md
│   └── bonus_nemo_comparison.md
│
├── scripts/
│   ├── verify_setup.py
│   ├── run_phase_a.py
│   ├── run_phase_b.py
│   ├── run_phase_c.py
│   ├── run_eval.py                    # used by CI
│   └── check_budget.py
│
├── .github/workflows/
│   └── eval-gate.yml
│
├── demo/
│   └── demo_script.md                 # 5-min recording script
│
├── docs/
│   ├── architecture.md                # ASCII graph + node desc chi tiết
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-05-12-lab24-eval-guardrails-design.md
│       └── plans/                     # writing-plans output sau
│
└── tests/                             # optional smoke tests
    ├── test_input_guard.py
    ├── test_output_guard.py
    └── test_full_pipeline.py
```

---

## 6. Data flow — 1 user query qua full stack

```
[t=0ms]  user_input = "Số CCCD của tôi là 012345678901, NĐ 13 quy định gì?"
          │
          ▼
[t=0]    asyncio.gather(
            N1.sanitize  → "Số CCCD của tôi là [CCCD], NĐ 13 quy định gì?"
            N2.check     → (True, "On topic: legal/data_protection", sim=0.78)
            N3.detect    → (injection_score=0.02, blocked=False)
         )
[t=L1]   timings['L1'] ≈ 30-50ms
          │
          │ topic_ok ∧ ¬injection_blocked
          ▼
[t=L1]   sanitized_input → N4 HybridSearch (20 chunks)
          → N5 Rerank (top 3)
          → N6 GPT-4o-mini → answer
[t=L2]   timings['L2'] ≈ 1500-3000ms
          │
          ▼
[t=L2]   N7 Gemini Safety check(user_input, answer) → is_safe=True
[t=L3]   timings['L3'] ≈ 100-300ms
          │
          ▼
[t=L3]   asyncio.create_task(N8 audit_log(...))   # fire-forget
          │
          ▼
         return answer, timings
```

**Latency target:** L1 P95 < 50ms · L3 P95 < 100ms · End-to-end P95 < 3500ms.

---

## 7. Output artifact contracts

| File | Columns / Schema | Min rows |
|---|---|---|
| `phase-a/testset_v1.csv` | `question, ground_truth, contexts, evolution_type` | 50 |
| `phase-a/ragas_results.csv` | `question, faithfulness, answer_relevancy, context_precision, context_recall` | 50 |
| `phase-a/ragas_summary.json` | `{faithfulness, answer_relevancy, context_precision, context_recall, total_cost_usd}` | — |
| `phase-b/pairwise_results.csv` | `question, answer_a, answer_b, run1_winner, run2_winner_flipped, winner_after_swap, reason` | 30 |
| `phase-b/absolute_scores.csv` | `question, answer, accuracy, relevance, conciseness, helpfulness, overall` | 30 |
| `phase-b/human_labels.csv` | `question_id, human_winner ∈ {A,B,tie}, confidence ∈ {high,medium,low}, notes` | 10 |
| `phase-c/pii_test_results.csv` | `input, output, pii_found, latency_ms` | 10 |
| `phase-c/adversarial_test_results.csv` | `attack_type, text (≤50 chars), blocked, reason, latency_ms` | 20 |
| `phase-c/topic_test_results.csv` | `input, on_topic_expected, on_topic_predicted, reason, latency_ms` | 20 |
| `phase-c/output_guard_test_results.csv` | `case_id, user_input, agent_response, ground_truth_safe, predicted_safe, latency_ms, verdict_raw` | 20 |
| `phase-c/latency_benchmark.csv` | `request_id, L1_ms, L2_ms, L3_ms, total_ms, blocked_at_layer` | 100 |

---

## 8. Error handling

### 8.1 Per-node failure handling

| Node | Failure mode | Policy | Handling |
|---|---|---|---|
| N1 PII Scrub | Presidio load fail / regex backtrack | Fail-open | try/except, timeout 5s, return text gốc |
| N2 Topic | OpenAI rate limit / timeout | Fail-open | `tenacity.retry(3)` + fail-open `(True, "unavailable")` |
| N3 Prompt Guard | HF model OOM / not downloaded | Fail-open | Lazy load + skip if fail |
| N4 HybridSearch | Qdrant down | Hard-fail | Raise `RAGUnavailable` → N9 refuse |
| N5 Reranker | Model load fail | Soft-fail | Skip rerank, use top 3 from N4 |
| N6 LLM Generate | OpenAI 429/500 | Hard-fail | `tenacity.retry(3)` → N9 refuse |
| N7 Output Safety | Gemini fail | **Fail-CLOSED** | try/except → treat unsafe → N9 refuse |
| N8 Audit Log | Disk full | Fail-open silent | Swallow exception, log to stderr |
| N11 Pairwise Judge | JSON parse error | Tie fallback | `{"winner":"tie","reason":"parse_error"}` + count |
| N15 Adversarial | Any exception in attack run | Block-as-pass | `blocked=True, reason=str(e)` |

### 8.2 Async error semantics

L1 dùng `asyncio.gather(..., return_exceptions=True)` để 1 task fail không cancel toàn bộ. L3 dùng `try/except` quanh `asyncio.to_thread(output_guard.check, ...)`.

### 8.3 Refuse response template

```python
def refuse_response(reason: str) -> str:
    """Graceful fallback — không reveal internal failure."""
    table = {
        "off_topic": "Xin lỗi, câu hỏi này nằm ngoài phạm vi hỗ trợ.",
        "injection_detected": "Xin lỗi, câu hỏi không hợp lệ.",
        "unsafe": "Xin lỗi, không thể trả lời câu hỏi này.",
        "rag_error": "Hệ thống đang bận, vui lòng thử lại sau.",
        "guard_error_fail_closed": "Xin lỗi, không thể xử lý yêu cầu này.",
    }
    for key, msg in table.items():
        if key in reason:
            return msg
    return "Xin lỗi, không thể xử lý yêu cầu này."
```

---

## 9. Testing strategy

### 9.1 Unit tests (`tests/`, optional)

| Test file | Coverage |
|---|---|
| `test_input_guard.py` | CCCD regex, phone variants, Presidio NER, chain order, latency < 50ms |
| `test_output_guard.py` | Prompt format, parse safe/unsafe, fail-closed on exception, mock Gemini |
| `test_full_pipeline.py` | Async happy path, off-topic block at L1, unsafe block at L3, audit fires, timings populated |

### 9.2 Acceptance tests (embedded trong phase scripts)

| Script | Assertion |
|---|---|
| `phase-a/gen_testset.py` | `len(df) >= 50; set(evolution_type) >= {simple, reasoning, multi_context}` |
| `phase-a/run_ragas.py` | 4 metric columns present, NaN < 10% |
| `phase-c/input_guard.py` __main__ | PII detection ≥ 0.80, P95 latency < 50ms |
| `phase-c/output_guard.py` __main__ | Detection ≥ 0.80, FP rate ≤ 0.20 |
| `phase-c/full_pipeline.py` benchmark | ≥ 100 requests, P95 L1 < 50ms, P95 L3 < 100ms |

### 9.3 Manual verification (Self-Assessment Checklist Phần 8)

- `scripts/verify_setup.py` → all green
- Run từng phase script, check artifact match contract
- Mermaid diagram render trong GitHub preview
- 5 demo queries: normal / off-topic / PII / jailbreak / unsafe-output

---

## 10. Phase D blueprint document

`phase-d/blueprint.md` cấu trúc 4 section:

1. **SLO Definition** — bảng ≥ 5 SLO (Faithfulness, Answer Relevancy, Context Precision, Context Recall, P95 Latency, Guardrail Detection Rate, FP Rate) + alert threshold + severity (P1/P2/P3).
2. **Architecture Diagram** — Mermaid graph 4 layer (format Mermaid để GitHub render).
3. **Alert Playbook** — 3 incident: Faithfulness drop, Adversarial spike, Latency degradation. Mỗi incident có Severity / Detection / Likely causes / Investigation steps / Resolution / SLO impact (TTD + TTR).
4. **Cost Analysis** — bảng 100k queries/month + optimization opportunities.

---

## 11. Bonus integration

### 11.1 Prompt Guard (Meta) — Easy +2

- File: `phase-c/prompt_guard.py`
- Model: `meta-llama/Prompt-Guard-86M` (public, CPU OK ~50ms inference)
- Lazy singleton load
- Plug L1 parallel cùng N1/N2 (3-way gather)
- Threshold: `injection_score > 0.5` → block
- Test 20 adversarial → expect detection rate tăng ~70% → ~90%
- Doc: `bonus/bonus_prompt_guard.md` (before/after comparison)

### 11.2 NeMo Guardrails — Medium +3

- File: `phase-c/nemo_adapter.py`
- Wrap NeMo Dialog Rails với same interface `check(text) -> (ok, reason)`
- **Offline comparison only** — không nhét vào L1 production gather (L1 vẫn 3-way: PII + Topic + Prompt Guard). NeMo chạy thêm pass trên cùng test set để so accuracy/refuse-rate với custom TopicGuard.
- Add column `nemo_verdict` vào `topic_test_results.csv`
- Doc: `bonus/bonus_nemo_comparison.md`

### 11.3 Blog post — Easy +2

- File: `bonus/blog_post.md`
- Sections: 3-questions framing / RAGAS vs LLM-Judge / Cohen's kappa results / Defense-in-depth / Latency reality / Cost findings
- Publish Medium hoặc dev.to, link trong README; nếu chưa publish → đánh dấu draft

---

## 12. Cost & budget

| Component | Estimate |
|---|---|
| RAGAS gen (gpt-4o-mini, 50 questions) | ~$1.50 |
| Pairwise judge (gpt-4o-mini, 30×2 swap) | ~$1.00 |
| Absolute scoring (gpt-4o-mini, 30) | ~$0.50 |
| Gemini safety (free tier 15 RPM, 1500 RPD) | $0 |
| Embedding (OpenAIEmbeddings cho topic) | ~$0.10 |
| Day18 RAG calls trong benchmark (100 queries) | ~$0.50 |
| **Total** | **~$3.60** |

**Hard budget gate:** `config.py::MAX_COST_USD = 15.0`. `scripts/check_budget.py` chạy trước mỗi phase, abort if exceeded.

---

## 13. Implementation timeline

```
Day 1 — Setup + Phase A (2h)
├── 0. Bootstrap project (30')
├── 1-4. Phase A.1 → A.4 (60')
│   └─ CHECKPOINT 1: ragas_results.csv ✓
│
Day 1 — Phase B (1h)
├── 5-8. Phase B.1 → B.4
│   └─ CHECKPOINT 2: kappa computed + bias report ✓
│
Day 1 — Phase C foundation (1h)
├── 9-10. C.1 InputGuard + C.2 TopicGuard
│
Day 2 — Phase C completion (1h)
├── 11-13. C.3 adversarial + C.4 output_guard + C.5 full_pipeline benchmark
│   └─ CHECKPOINT 3: L1<50ms, L3<100ms, benchmark 100+ ✓
│
Day 2 — Phase D (30')
├── 14-15. blueprint.md (SLO + diagram + playbook + cost)
│
Day 2 — Bonus (1.5h)
├── 16. Prompt Guard
├── 17. NeMo Guardrails adapter
└── 18. Blog post draft
│
Day 2 — Final (30')
├── 19. README.md + prompts.md
├── 20. Self-Assessment Checklist verify
└── 21. Demo script (record sau hoặc placeholder)
```

**Total ~7h** (5h cốt lõi + 2h bonus), trong khung 4h focused + 2-3h homework PDF dự kiến.

---

## 14. Risks + mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Qdrant không up khi run | Medium | High | `verify_setup.py` check connection + auto docker-compose instruction |
| Gemini rate limit free tier | High | Medium | `asyncio.Semaphore(10)` + sleep 0.5s giữa requests |
| RAGAS 0.2+ breaking changes vs Day18 0.1.x | Medium | Medium | Pin `ragas==0.2.0` isolated venv |
| Cohen kappa < 0.2 vì sample nhỏ | Medium | Low | Document root cause analysis, accept lower score |
| Prompt Guard load chậm lần đầu | Low | Low | Lazy singleton + warm-up trong verify_setup |
| NeMo dep conflict với LangChain | Medium | Medium | Test sớm Step 17; fallback chỉ Prompt Guard + Blog (+4) |
| Demo video không kịp record | High | Medium | Script viết sẵn + screenshot output + placeholder YouTube link |

---

## 15. Success criteria

- ✅ All Self-Assessment Checklist items ≥ 80% checked
- ✅ All output artifacts match contract (§ 7)
- ✅ `verify_setup.py` exits 0
- ✅ All `scripts/run_phase_*.py` chạy được liên tiếp
- ✅ RAGAS scores documented dù không đạt target
- ✅ `blueprint.md` Mermaid render đúng trong GitHub preview
- ✅ Total cost ≤ $15 ghi rõ trong README
- ✅ Target score ~107/115 (60 pass / 90 excellent threshold)

---

## 16. Open items (post-spec)

1. **Qdrant:** start lại `docker-compose up -d` từ Day 18 repo, hoặc copy `docker-compose.yml` sang Lab24.
2. **HF token:** Prompt Guard 86M public, chỉ cần `HF_TOKEN` nếu rate-limited.
3. **GitHub remote:** user tự tạo repo `lab24-eval-guardrails-NguyenVietLong` trên GitHub, add remote sau.
4. **Demo recording:** Loom/OBS sau khi code xong, không block submit (placeholder YouTube unlisted link).
5. **Memory:** lưu user prefs (Gemini + OpenAI key, no GPU, no Groq) vào memory bank.

---

## 17. Out of scope (explicit)

- Llama Guard 3 native deployment (no GPU/Groq).
- Fine-tune Llama Guard for Vietnamese (bonus Very Hard).
- Streamlit eval dashboard (bonus Medium, skip để focus 3 bonus đã chọn).
- SelfCheckGPT / Semantic entropy (bonus Hard, skip).
- Custom VN classifier (bonus Very Hard, skip).
- Production cloud deployment (local Docker đủ).
- Re-implement RAG pipeline (Day 18 reused as-is).
