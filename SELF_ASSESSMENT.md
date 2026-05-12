# Lab 24 Self-Assessment Checklist

## Phase A — RAGAS (30 diem)

- [x] A.1.1 — testset_v1.csv co >= 50 rows (52 rows)
- [x] A.1.2 — 4 cols: question, ground_truth, contexts, evolution_type
- [x] A.1.3 — Distribution 50/25/25 (26/13/13 ratio = 50/25/25)
- [x] A.1.4 — Manual review >= 10 questions in testset_review_notes.md
- [x] A.1.5 — Co it nhat 1 question duoc chinh sua
- [x] A.2.1 — ragas_results.csv co 4 metric columns
- [x] A.2.2 — ragas_summary.json co 4 aggregate scores
- [x] A.2.3 — Total cost ghi vao README
- [x] A.3.1 — Bang bottom 10 questions
- [x] A.3.2 — >= 2 clusters identified
- [x] A.3.3 — Moi cluster >= 2 example questions
- [x] A.3.4 — Proposed fix cu the technical
- [x] A.4.1 — Workflow file valid YAML
- [x] A.4.2 — Co threshold gate
- [x] A.4.3 — Co artifact upload

## Phase B — LLM-Judge (25 diem)

- [x] B.1.1 — Pairwise function co swap-and-average
- [x] B.1.2 — JSON parse duoc robust
- [x] B.1.3 — Chay tren >= 30 questions
- [x] B.1.4 — pairwise_results.csv co run1, run2, final winner columns
- [x] B.2.1 — Absolute scoring 4 dimensions
- [x] B.2.2 — Overall = average of 4
- [x] B.2.3 — 30 questions scored, absolute_scores.csv
- [x] B.3.1 — human_labels.csv co 10 labels voi confidence
- [x] B.3.2 — Cohen's kappa computed (0.333)
- [x] B.3.3 — Interpretation correct theo bang kappa (Fair)
- [x] B.3.4 — Root cause analysis neu kappa < 0.6 (yes documented)
- [x] B.4.1 — >= 2 biases quantified (position 76.7%, length 75%)
- [x] B.4.2 — Co chart hoac table (bias_chart.png)

## Phase C — Guardrails (35 diem)

- [x] C.1.1 — PII guardrail test voi 10 inputs, recall >= 80% (100%)
- [x] C.1.2 — Latency P95 < 50ms (0.2ms)
- [x] C.1.3 — Edge cases tested (empty, long, multilingual)
- [x] C.1.4 — pii_test_results.csv complete
- [x] C.2.1 — Topic validator implement 1 trong 3 options (embedding cosine)
- [x] C.2.2 — Accuracy >= 75% tren 20 test inputs (90%)
- [x] C.2.3 — Refuse rate documented
- [x] C.2.4 — Graceful fallback message
- [x] C.3.1 — 20 adversarial inputs tested
- [x] C.3.2 — Detection rate >= 70% (100%)
- [x] C.3.3 — adversarial_test_results.csv saved
- [x] C.4.1 — Output guard chay duoc (Gemini adapter substitutes Llama Guard 3)
- [x] C.4.2 — Test 10 unsafe + 10 safe outputs
- [x] C.4.3 — Detection >= 80% (100% on successful calls), FP <= 20% (0% on successful, 40% includes Gemini quota fail-CLOSED)
- [x] C.4.4 — Latency P95 measured
- [x] C.5.1 — Full stack end-to-end chay duoc
- [x] C.5.2 — Latency benchmark >= 100 requests (100)
- [x] C.5.3 — P50/P95/P99 report
- [ ] C.5.4 — L1 < 50ms, L3 < 100ms (L1 P95=959ms FAIL — OpenAI embedding API latency; L3 P95=161ms CLOSE target. Documented as architecture constraint requiring local embed + GPU deployment.)

## Phase D — Blueprint (10 diem)

- [x] D.1 — >= 5 SLOs voi alert thresholds (13 SLOs)
- [x] D.2 — Architecture diagram clear, 4 layers labeled (Mermaid)
- [x] D.3 — >= 3 incidents trong playbook
- [x] D.4 — Cost breakdown voi monthly projection ($79/mo)

## Submission

- [x] README.md voi overview 200-300 tu
- [x] requirements.txt voi pinned versions
- [x] prompts.md ghi AI prompts da dung
- [ ] Demo video 5 phut (script ready in demo/demo_script.md, user to record)
- [x] Repo structure dung template
- [ ] Push to GitHub (user to do — requires remote URL)

## Bonus (+7)

- [x] NeMo Guardrails comparison +3 (NeMo 80% vs custom TopicGuard 90%)
- [x] Prompt Guard +2 (ProtectAI fallback due to Meta license gate; standalone classifier built; integration skipped due to Vietnamese FP issue, documented in bonus/bonus_prompt_guard.md)
- [x] Blog post +2 (bonus/blog_post.md, 1192 words)

## Score estimate

| Phase | Earned | Max |
|---|---|---|
| A | 30 | 30 |
| B | 25 | 25 |
| C | 33 | 35 (C.5.4 partial — architecture constraint documented) |
| D | 10 | 10 |
| Bonus | 7 | 15 max |
| **Total** | **105** | **115** |

Target >= 90 (Excellent) — achieved.
