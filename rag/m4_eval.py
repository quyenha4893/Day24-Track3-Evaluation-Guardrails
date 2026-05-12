"""Module 4: RAGAS Evaluation - 4 metrics + failure analysis."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation, with a deterministic lexical fallback."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row.question,
                answer=row.answer,
                contexts=row.contexts,
                ground_truth=row.ground_truth,
                faithfulness=float(row.faithfulness),
                answer_relevancy=float(row.answer_relevancy),
                context_precision=float(row.context_precision),
                context_recall=float(row.context_recall),
            )
            for _, row in df.iterrows()
        ]
    except Exception:
        per_question = []
        for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
            context_text = " ".join(ctxs)
            per_question.append(EvalResult(
                question=question,
                answer=answer,
                contexts=ctxs,
                ground_truth=ground_truth,
                faithfulness=_token_overlap(answer, context_text),
                answer_relevancy=_token_overlap(question, answer),
                context_precision=_token_overlap(answer, context_text),
                context_recall=_token_overlap(ground_truth, context_text),
            ))

    def mean(metric: str) -> float:
        values = [getattr(r, metric) for r in per_question]
        return sum(values) / len(values) if values else 0.0

    return {
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using a diagnostic tree."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ranked = sorted(
        eval_results,
        key=lambda r: sum(getattr(r, m) for m in metric_names) / len(metric_names),
    )[:bottom_n]

    failures = []
    for result in ranked:
        scores = {metric: getattr(result, metric) for metric in metric_names}
        worst_metric = min(scores, key=scores.get)
        score = scores[worst_metric]
        if worst_metric == "faithfulness" and score < 0.85:
            diagnosis = "LLM hallucinating"
            fix = "Tighten prompt, lower temperature"
        elif worst_metric == "context_recall" and score < 0.75:
            diagnosis = "Missing relevant chunks"
            fix = "Improve chunking or add BM25"
        elif worst_metric == "context_precision" and score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            fix = "Add reranking or metadata filter"
        elif worst_metric == "answer_relevancy" and score < 0.80:
            diagnosis = "Answer doesn't match question"
            fix = "Improve prompt template"
        else:
            diagnosis = "Borderline quality issue"
            fix = "Inspect retrieval and generation for this question"

        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": float(score),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
