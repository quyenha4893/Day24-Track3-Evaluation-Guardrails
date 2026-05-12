"""B.1 — Pairwise judge with swap-and-average bias mitigation.
Variant A = current Day 18 pipeline (RERANK_TOP_K=3)
Variant B = pipeline with RERANK_TOP_K=5 (test if more context helps)
"""

import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from config import PHASE_A_DIR, PHASE_B_DIR, JUDGE_MODEL, PAIRWISE_SAMPLE_SIZE
from rag.pipeline import build_pipeline, run_query


JUDGE_PROMPT = PromptTemplate.from_template("""You are an impartial evaluator. Compare two answers to the same question.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Rate based on:
- Factual accuracy
- Relevance to question
- Conciseness

Output JSON only:
{{"winner": "A" or "B" or "tie", "reason": "..."}}
""")


def parse_judge_output(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', text)
        winner = m.group(1) if m else "tie"
        return {"winner": winner, "reason": "parse_fallback"}


def pairwise_judge_with_swap(question, ans_a, ans_b, judge_llm) -> dict:
    """Run twice with swapped order. Both-agree → that. Disagree → tie."""
    prompt1 = JUDGE_PROMPT.format(question=question, answer_a=ans_a, answer_b=ans_b)
    out1 = judge_llm.invoke(prompt1).content
    r1 = parse_judge_output(out1)

    prompt2 = JUDGE_PROMPT.format(question=question, answer_a=ans_b, answer_b=ans_a)
    out2 = judge_llm.invoke(prompt2).content
    r2 = parse_judge_output(out2)

    r2_flipped = r2["winner"]
    if r2_flipped == "A":
        r2_flipped = "B"
    elif r2_flipped == "B":
        r2_flipped = "A"

    final = r1["winner"] if r1["winner"] == r2_flipped else "tie"
    return {
        "run1_winner": r1["winner"],
        "run2_winner_flipped": r2_flipped,
        "winner_after_swap": final,
        "reason": r1.get("reason", ""),
    }


def build_variant_b_answer(question, search, reranker):
    """Variant B: bump RERANK_TOP_K to 5 temporarily."""
    import config
    original = config.RERANK_TOP_K
    config.RERANK_TOP_K = 5
    try:
        ans, _ = run_query(question, search, reranker)
    finally:
        config.RERANK_TOP_K = original
    return ans


def main():
    df_full = pd.read_csv(PHASE_A_DIR / "ragas_results.csv")
    # ragas_results.csv uses 'user_input' and 'response' column names
    if "question" not in df_full.columns and "user_input" in df_full.columns:
        df_full = df_full.rename(columns={"user_input": "question", "response": "answer"})
    df = df_full.head(PAIRWISE_SAMPLE_SIZE).copy()
    print(f"Pairwise judge on {len(df)} questions")

    search, reranker = build_pipeline()
    judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0)

    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        ans_a = r.answer
        try:
            ans_b = build_variant_b_answer(r.question, search, reranker)
        except Exception as e:
            print(f"[{i}] variant B failed: {e}")
            ans_b = ans_a + " [variant_b_failed]"
        try:
            verdict = pairwise_judge_with_swap(r.question, ans_a, ans_b, judge)
        except Exception as e:
            print(f"[{i}] judge failed: {e}")
            verdict = {"run1_winner": "tie", "run2_winner_flipped": "tie",
                       "winner_after_swap": "tie", "reason": f"judge_error: {e}"}
        rows.append({
            "question": r.question,
            "answer_a": ans_a,
            "answer_b": ans_b,
            **verdict,
        })
        print(f"[{i}/{len(df)}] {verdict['winner_after_swap']}")

    out = pd.DataFrame(rows)
    out.to_csv(PHASE_B_DIR / "pairwise_results.csv", index=False)
    print(f"Saved {len(out)} rows → {PHASE_B_DIR / 'pairwise_results.csv'}")
    print("\nDistribution:")
    print(out["winner_after_swap"].value_counts())


ABSOLUTE_PROMPT = PromptTemplate.from_template("""Score the answer on 4 dimensions, each 1-5 scale:

1. Factual accuracy (1=many errors, 5=fully accurate)
2. Relevance (1=off-topic, 5=directly answers)
3. Conciseness (1=verbose, 5=appropriately brief)
4. Helpfulness (1=unclear, 5=actionable)

Question: {question}
Answer: {answer}

Output JSON only:
{{"accuracy": int, "relevance": int, "conciseness": int, "helpfulness": int, "overall": float}}
""")


def absolute_score(question, answer, judge_llm) -> dict:
    prompt = ABSOLUTE_PROMPT.format(question=question, answer=answer)
    out = judge_llm.invoke(prompt).content
    parsed = parse_judge_output(out)
    dims = ["accuracy", "relevance", "conciseness", "helpfulness"]
    for d in dims:
        if d not in parsed or not isinstance(parsed[d], (int, float)):
            parsed[d] = 3
    if "overall" not in parsed or not isinstance(parsed["overall"], (int, float)):
        parsed["overall"] = sum(parsed[d] for d in dims) / 4
    return parsed


if __name__ == "__main__":
    main()
