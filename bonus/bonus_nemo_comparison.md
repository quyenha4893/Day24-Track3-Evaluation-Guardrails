# Bonus: NeMo Guardrails vs Custom TopicGuard

## Approach

- **Custom TopicGuard (C.2):** OpenAIEmbeddings cosine similarity vs 3 allowed-topic embeddings, threshold=0.4.
- **NeMo Guardrails:** `self check input` flow with `self_check_input` task prompt (LLM-based classification). Prompt instructs GPT-4o-mini to block questions outside allowed Vietnamese legal/financial topics. NeMo version: 0.21.0.
- **Methodology:** Both systems evaluated on the same 20-input test set (10 on-topic VN legal/financial + 10 off-topic random). NeMo runs offline, not inserted into L1 production pipeline.
- **Installation note:** NeMo 0.21.0 reads YAML configs without UTF-8 encoding on Windows; resolved via `builtins.open` monkey-patch to force UTF-8 for `.yml` files.

## Results

| Metric | Custom (embedding) | NeMo (LLM check) |
|---|---|---|
| Accuracy | 90.0% | 80.0% |
| True Positives (on-topic correctly allowed) | 8 / 10 | 6 / 10 |
| True Negatives (off-topic correctly blocked) | 10 / 10 | 10 / 10 |
| False Negatives (on-topic blocked) | 2 | 4 |
| False Positives (off-topic allowed) | 0 | 0 |
| Avg latency | ~221ms | ~3610ms |
| Cost estimate | ~$0.001/call (embedding) | ~$0.005/call (LLM) |
| Setup complexity | 10 lines Python | YAML config + LLMRails |

## Misclassification analysis

### Custom TopicGuard errors (2 FN)
Both custom failures: `Lãi gộp quý 4 là bao nhiêu?` (cosine=0.27) and `Tài sản cố định trong BCTC?` (cosine=0.40 — right at threshold) were blocked. These short Vietnamese financial terms have low embedding similarity to the reference phrases.

### NeMo errors (4 FN)
NeMo blocked 4 on-topic queries that the custom guard also struggled with or got right:
1. `Lãi gộp quý 4 là bao nhiêu?` — also missed by custom; brief phrase
2. `Tài sản cố định trong BCTC?` — also missed by custom; brief phrase
3. `Hành vi bị cấm khi xử lý dữ liệu?` — **correctly allowed by custom** (cosine=0.64); NeMo blocked it, possibly because the ASCII-fallback prompt lost nuance from transliteration
4. `Lợi nhuận sau thuế của doanh nghiệp?` — **correctly allowed by custom** (cosine=0.50); NeMo blocked it

### Agreement
- Both systems agreed on 16/20 (80%) inputs.
- Custom caught 2 additional on-topic queries that NeMo missed (rows 9 and 10 from on-topic set).
- All 10 off-topic queries: both systems agreed (blocked).

### Root cause of NeMo's extra failures
The nemo_config.yml prompt was written in ASCII-transliterated Vietnamese (to work around Windows charmap codec in NeMo 0.21.0). GPT-4o-mini received a less precise prompt and was stricter than intended for short financial queries. The embedding-based custom guard is not affected by prompt encoding issues.

## Trade-off analysis

- **Custom embedding:** fast (~221ms avg), cheap, no LLM call overhead. Weakness: threshold brittle at boundary (0.27–0.40 cosine zone) and embedding model may lack domain specificity for short Vietnamese financial terms.
- **NeMo LLM:** more expressive policy specification in YAML; can handle edge cases that embeddings miss. However: ~16x slower (3610ms vs 221ms), ~5x more expensive per call, and with NeMo 0.21.0 there are Windows-specific encoding issues requiring workarounds.

## Recommendation

- **Production L1 (latency < 500ms):** custom embedding guard — NeMo's 3.6s average latency is prohibitive.
- **L2 audit (offline batch):** NeMo viable for nightly re-classification of hard cases flagged by L1.
- **At scale > 1M/day:** NeMo LLM cost ($5,000/day) dominates → keep embedding in L1, cache common queries.
- **NeMo version note:** 0.21.0 has breaking API changes from earlier versions (flow name `self check input`, not `check user message`; Windows encoding workaround required). Consider upgrading to a stable release with official Windows support before production use.
