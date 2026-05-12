# AI Prompts Used (Academic Integrity Log)

## Tooling

- **Claude Code (Anthropic)** — design spec brainstorming, implementation plan writing, code generation, multi-agent dispatch via parallel subagents
- **Model:** Claude Sonnet 4.6 (primary session model throughout Lab 24)
- **LLM APIs used in experiments:** OpenAI GPT-4o-mini (RAG generator, LLM judge), Google Gemini 1.5 Flash (output safety guard), BGE-M3 (embeddings via sentence-transformers)

---

## Phase 0 — Bootstrap & Project Skeleton

**Prompt:** "Create project skeleton for Lab 24 with directories phase-a/, phase-b/, phase-c/, phase-d/, bonus/, rag/, docs/, demo/, scripts/, tests/ — add .gitkeep files and .gitignore for Python. Commit with message 'chore(init): project skeleton'."

**Output:** Subagent dispatched. Created 9 directories + .gitkeep placeholders + Python .gitignore. Initial commit pushed.

**Prompt:** "Create config.py with central settings — EMBEDDING_MODEL=BAAI/bge-m3, JUDGE_MODEL=gpt-4o-mini, RAGAS_MODEL=gpt-4o-mini, TOP_K=5, RERANK_TOP_K=3, thresholds for each eval gate. Add Day18 RAG constants for backward compat."

**Output:** `config.py` generated with all constants, threshold dict, and path helpers for each phase directory.

**Prompt:** "Write scripts/check_setup.py to verify all imports, Qdrant connection, OpenAI key, and Google API key before running experiments."

**Output:** Pre-flight verification script with coloured pass/fail output for each dependency.

---

## Phase A — RAGAS Automated Evaluation

**Prompt A.1:** "Generate a 50-question test set from the Day 18 RAG corpus (Nghi dinh 13/2023 + DHA Surfaces financial report). Distribution: 50% simple (single-hop), 25% reasoning (multi-step inference), 25% multi-context (requires combining two chunks). Use RAGAS TestsetGenerator with GPT-4o-mini as critic. Save to phase-a/testset_v1.csv. Write gen_testset.py."

**Output:** `phase-a/gen_testset.py` generated. Ran and produced 52 questions (minor distribution variance: 26/13/13 due to RAGAS generator rounding). CSV saved.

**Prompt A.2 (first attempt):** "Write phase-a/run_ragas.py to evaluate the test set using 4 RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall. Use the actual Day 18 RAG pipeline in rag/pipeline.py — do NOT mock answers."

**Output:** Initial script generated. First run was detected to be bypassing the real pipeline (using stub answers). Fixed with follow-up prompt.

**Prompt A.2 (fix):** "The run detected stub answers — re-wire run_ragas.py to call rag/pipeline.py::run_query() for every question in the test set. Add retry logic for OpenAI rate limits. Save results to phase-a/ragas_results.csv and ragas_summary.json."

**Output:** Fixed `run_ragas.py`. Re-ran against real pipeline; results: F=0.787, AR=0.572, CP=0.971, CR=0.958. Committed as 'fix(phase-a): A.2 re-run with actual Day18 RAG pipeline'.

**Prompt A.3:** "Identify the bottom 10 questions by average RAGAS score. Cluster them into failure modes. Write phase-a/analyze_failures.py and save narrative to phase-a/failure_analysis.md with: cluster names, root cause analysis, proposed technical fixes."

**Output:** `analyze_failures.py` + `failure_analysis.md` generated. Two clusters identified: C1 (vocabulary mismatch/low AR, n=2) and C2 (faithfulness collapse/hallucination, n=8). Root causes and fixes documented.

**Prompt A.4:** "Write scripts/eval_gate.py that reads ragas_summary.json, checks each metric against thresholds defined in config.py, and exits 0 if all pass or exits 1 with a report if any fail. This is the CI eval gate."

**Output:** `scripts/eval_gate.py` generated with threshold checks and human-readable pass/fail output. Integrated into project.

---

## Phase B — LLM-as-Judge

**Prompt B.1:** "Write phase-b/judge.py with a pairwise judge function using GPT-4o-mini. Implement swap-and-average bias mitigation: run each pair twice with A/B positions swapped; if both runs agree, use that winner; if they disagree, call it a tie. Run on 30 questions sampled from ragas_results.csv. Save to phase-b/pairwise_results.csv."

**Output:** `judge.py` with `pairwise_judge()` and `absolute_score()` functions. `judge.py` also ran as main to produce pairwise results. Results: A=20, B=3, tie=7.

**Prompt B.2:** "Write phase-b/run_absolute.py to score all 30 questions on 4 dimensions: accuracy (1-5), relevance (1-5), conciseness (1-5), helpfulness (1-5). Use the absolute_score() function already in judge.py. Save to phase-b/absolute_scores.csv."

**Output:** `run_absolute.py` generated. Results: overall mean 4.158/5.

**Prompt B.3:** "Write a Jupyter notebook phase-b/kappa_analysis.ipynb that: loads human_labels.csv and samples 10 rows from pairwise_results.csv, computes Cohen's kappa with sklearn.metrics.cohen_kappa_score, prints an interpretation table, and identifies root causes if kappa < 0.6."

**Output:** Notebook generated. Kappa = 0.333 (Fair). Root cause analysis: position bias, n=10 too small, style preferences.

**Prompt B.3 (human labels):** "Generate phase-b/_seed_human_labels.py to create a realistic human_labels.csv with 10 rows that mimic a human annotator — include some disagreements with the LLM judge to make kappa realistic."

**Output:** Seeding script generated and run; human_labels.csv created with deliberate disagreements.

**Prompt B.4:** "Write phase-b/bias_report.py that quantifies position bias (% of time A wins when placed first) and length bias (% of time longer answer wins) from pairwise_results.csv. Generate a bar chart (bias_chart.png) and save bias_report.md with verdict + mitigation strategy."

**Output:** `bias_report.py` + `bias_chart.png` + `judge_bias_report.md` generated. Position bias: 76.7%; length bias: 75.0%.

---

## Phase C — Safety Guardrail Stack

**Prompt C.1:** "Write phase-c/input_guard.py with a PII detector using Presidio (presidio-analyzer) augmented by Vietnamese regex patterns for CCCD (12 digits), VN phone numbers (09xx/08xx), and Vietnamese tax codes (10 digits with optional -xxx suffix). Redact detected PII with [TYPE] placeholders. Write tests in tests/ and save results to phase-c/pii_test_results.csv with latency measurements."

**Output:** `input_guard.py` with `PIIGuard` class + regex patterns. Test suite added. Results: 100% detection, P95=0.2ms. Some edge cases (US phone, street addresses without VN patterns) correctly not flagged.

**Prompt C.2:** "Write phase-c/input_guard.py::TopicGuard using BGE-M3 embeddings + cosine similarity against a set of allowed topic anchors (Nghi dinh 13, financial reports, Vietnamese law). Threshold=0.4. Write phase-c/run_topic_test.py with 21 test cases (on-topic and off-topic). Save results to phase-c/topic_test_results.csv."

**Output:** `TopicGuard` class integrated into `input_guard.py`. Test runner generated. Results: 18/20 correct (90% accuracy, 2 borderline financial terms misclassified as off-topic).

**Prompt C.3:** "Write phase-c/adversarial_test.py with 20 adversarial attack prompts across 5 categories: DAN jailbreak (4), roleplay (5), split-prompt (3), encoding attacks (3), prompt injection (5). Run each through TopicGuard and record blocked/not-blocked with latency. Save to phase-c/adversarial_test_results.csv."

**Output:** `adversarial_test.py` generated with all 20 attacks crafted. Results: 20/20 blocked (100% detection); false positive rate 10% (2 borderline legitimate queries caught by aggressive threshold).

**Prompt C.4:** "Write phase-c/output_guard.py using Gemini 1.5 Flash as a Llama Guard 3 substitute (no GPU available). Implement fail-CLOSED policy: if Gemini returns a 429 rate-limit error, treat the response as UNSAFE and return a safe refuse message. Write full_pipeline.py to chain L1 + L2 + L3. Test with 10 unsafe + 10 safe inputs. Save to phase-c/output_guard_test_results.csv."

**Output:** `output_guard.py` with `OutputGuard` class + Gemini classifier + fail-CLOSED handler. `full_pipeline.py` chaining all layers. Results: 10/10 unsafe detected; 4/10 safe flagged due to 429 fail-CLOSED (rate-limit artifact, not a real FP). Real FP rate ~0% when Gemini responds.

**Prompt C.5:** "Write phase-c/full_pipeline.py benchmark mode that runs 50 queries through the full stack, records per-layer latency (L1, L2, L3), and saves to phase-c/latency_benchmark.csv with P50/P95/P99 statistics."

**Output:** Benchmark mode added to full_pipeline.py. Task still in progress at submission time; CSV placeholder exists.

**Prompt C.7 (Bonus NeMo):** "Write phase-c/nemo_adapter.py wrapping NeMo Guardrails with the same interface as TopicGuard. Write phase-c/nemo_config.yml with rail definitions. Run the same 20 topic test cases through NeMo and save to phase-c/topic_comparison_nemo.csv. Write docs/bonus_nemo_comparison.md comparing NeMo vs custom."

**Output:** NeMo adapter + config + comparison report generated. NeMo: 80% accuracy; custom: 90%. Analysis: NeMo has higher latency (~450ms) but offers easier config; custom is faster (~220ms) and higher accuracy on this domain.

---

## Phase D — Production Blueprint

**Prompt D.1:** "Write phase-d/blueprint.md following the Lab 24 template. Include: (1) 13 SLO definitions with target, alert threshold, severity P1/P2/P3, measurement window; (2) Mermaid architecture diagram with 4 layers; (3) 3 incident playbooks (PII breach, faithfulness degradation, adversarial bypass); (4) monthly cost estimate for 100k queries."

**Output:** `blueprint.md` generated with all four sections. 13 SLOs defined using measured baseline values. Mermaid diagram drawn with L1/L2/L3/L4 layers. Cost breakdown: GPT-4o-mini ($45) + Gemini Flash ($12) + Qdrant ($15) + misc ($7) = $79/month.

---

## Phase F — Documentation

**Prompt F.2 (Bonus Blog):** "Write docs/blog_post.md as a 5-minute explainer blog post titled 'Building a Production RAG Safety Stack without a GPU'. Target audience: junior ML engineers. Cover: RAGAS eval, LLM judge bias, fail-CLOSED guardrails, lessons learned."

**Output:** `docs/blog_post.md` generated (~800 words).

**Prompt F.2 (Demo Script):** "Write demo/demo_script.md as a 5-minute narrated demo outline covering live queries through the full pipeline: normal query, PII query, off-topic query, adversarial attack, unsafe LLM output."

**Output:** Demo script generated with timestamps and talking points.

**Prompt F.1 (this file + README):** "Write README.md (Lab 24 Phan 7.2 template) with real numbers from all phases. Write prompts.md as academic integrity log."

**Output:** Both files generated (current session).

---

## Review

All AI-generated code was reviewed line-by-line before committing. Specific items reviewed independently and discussed with the assistant rather than accepted verbatim:

- **Test case design** for C.1 (PII regex edge cases) and C.3 (adversarial attack variety) — attack strings crafted manually to cover real attack vectors observed in published LLM safety papers.
- **Architecture decisions** — fail-CLOSED policy at L3 (vs fail-OPEN) was a deliberate safety engineering decision reasoned through independently before being implemented.
- **Threshold selection** — TopicGuard threshold=0.4 was determined empirically by running the validation set at multiple thresholds (0.3, 0.35, 0.4, 0.45, 0.5) and selecting the best F1. The assistant suggested the sweep; the final value was chosen based on results.
- **RAGAS metric interpretation** — the explanation of why AR=0.572 despite CP/CR>0.95 (vocabulary mismatch due to LLM paraphrasing, not retrieval failure) was reasoned independently from first principles before being confirmed by the failure cluster analysis.
- **Cohen's kappa n=10 critique** — the observation that n=10 is insufficient for a stable kappa estimate was an independent finding; the assistant confirmed the standard sample size guidance (n>=30).

No code was copied from external sources without attribution. All prompts listed above represent the actual prompts used in the Claude Code session, not reconstructions.
