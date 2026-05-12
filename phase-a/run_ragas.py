"""A.2 — Chay RAGAS 4 metrics tren test set qua Day 18 RAG pipeline THAT."""

import sys, json
from pathlib import Path
# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import PHASE_A_DIR, JUDGE_MODEL
from rag.pipeline import build_pipeline, run_query


def main():
    df = pd.read_csv(PHASE_A_DIR / "testset_v1.csv")
    print(f"Loaded {len(df)} questions")

    print("Building RAG pipeline (this may download models ~2GB on first run)...")
    search, reranker = build_pipeline()

    results_data = []
    failures = 0
    for i, row in df.iterrows():
        try:
            answer, contexts = run_query(row["question"], search, reranker)
            if not isinstance(contexts, list) or len(contexts) == 0:
                contexts = [""]
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            print(f"[{i+1}] FAIL: {e}")
            answer, contexts = "", [""]
            failures += 1
        results_data.append({
            "question": row["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": row["ground_truth"],
        })
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{i+1}/{len(df)}] done")

    print(f"\n{len(results_data) - failures}/{len(results_data)} RAG queries succeeded")

    dataset = Dataset.from_list(results_data)
    print("Running RAGAS evaluate ...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatOpenAI(model=JUDGE_MODEL, temperature=0),
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        run_config=RunConfig(max_workers=4, timeout=180),
    )

    df_scores = scores.to_pandas()
    df_scores.to_csv(PHASE_A_DIR / "ragas_results.csv", index=False)
    print(f"Saved ragas_results.csv ({len(df_scores)} rows)")

    summary = {
        "faithfulness": float(df_scores["faithfulness"].mean(skipna=True)),
        "answer_relevancy": float(df_scores["answer_relevancy"].mean(skipna=True)),
        "context_precision": float(df_scores["context_precision"].mean(skipna=True)),
        "context_recall": float(df_scores["context_recall"].mean(skipna=True)),
        "n_questions": len(df_scores),
        "n_rag_failures": failures,
        "total_cost_usd": None,
        "rag_pipeline": "Day18 HybridSearch (BM25+Dense) + CrossEncoder rerank + GPT-4o-mini",
    }
    with open(PHASE_A_DIR / "ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
