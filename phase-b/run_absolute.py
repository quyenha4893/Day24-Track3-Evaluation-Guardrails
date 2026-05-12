"""B.2 — Absolute scoring 4-dim trên 30 questions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import importlib.util
from langchain_openai import ChatOpenAI

spec = importlib.util.spec_from_file_location("judge_mod", Path(__file__).parent / "judge.py")
judge_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge_mod)
absolute_score = judge_mod.absolute_score

from config import PHASE_A_DIR, PHASE_B_DIR, JUDGE_MODEL, ABSOLUTE_SAMPLE_SIZE


def main():
    df = pd.read_csv(PHASE_A_DIR / "ragas_results.csv")
    # Normalise column names (ragas_results.csv uses user_input/response)
    if "question" not in df.columns and "user_input" in df.columns:
        df = df.rename(columns={"user_input": "question", "response": "answer"})
    df = df.head(ABSOLUTE_SAMPLE_SIZE)
    judge = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        try:
            s = absolute_score(r.question, r.answer, judge)
        except Exception as e:
            print(f"[{i}] error: {e}")
            s = {"accuracy": 3, "relevance": 3, "conciseness": 3, "helpfulness": 3, "overall": 3.0}
        rows.append({"question": r.question, "answer": r.answer, **s})
        print(f"[{i}/{len(df)}] overall={s['overall']:.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(PHASE_B_DIR / "absolute_scores.csv", index=False)
    print(f"Saved {len(out)} -> {PHASE_B_DIR / 'absolute_scores.csv'}")
    print("\nSummary:")
    for d in ["accuracy", "relevance", "conciseness", "helpfulness", "overall"]:
        print(f"  {d}: mean={out[d].mean():.2f}")


if __name__ == "__main__":
    main()
