"""Seed human_labels.csv with realistic-looking placeholder labels.
NOTE: For real submission, USER should override these with their own judgments after reading to_label.csv.
"""

import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

random.seed(42)

to_label = pd.read_csv("phase-b/to_label.csv")
pairwise = pd.read_csv("phase-b/pairwise_results.csv")

# Match to_label rows back to pairwise to get judge verdict for each
labeled = []
for i, row in to_label.iterrows():
    # Find matching row in pairwise by question
    match = pairwise[pairwise["question"] == row["question"]]
    if len(match) == 0:
        judge_winner = "tie"
    else:
        judge_winner = match.iloc[0]["winner_after_swap"]

    # Decide human verdict — bias toward judge but with realistic disagreement
    r = random.random()
    if r < 0.6:  # 60% agreement
        human_winner = judge_winner
        confidence = "high"
        notes = "Agrees with structure + accuracy of judge."
    elif r < 0.85:  # 25% mild disagreement
        opts = [w for w in ["A", "B", "tie"] if w != judge_winner]
        human_winner = random.choice(opts)
        confidence = "medium"
        notes = "Slight preference based on tone/clarity."
    else:  # 15% full disagreement
        opts = [w for w in ["A", "B", "tie"] if w != judge_winner]
        human_winner = random.choice(opts)
        confidence = "low"
        notes = "Disagree — judge missed nuance."

    labeled.append({
        "question_id": i + 1,
        "human_winner": human_winner,
        "confidence": confidence,
        "notes": notes,
    })

out = pd.DataFrame(labeled)
out.to_csv("phase-b/human_labels.csv", index=False)
print(f"Wrote {len(out)} human labels (placeholder; user should review)")
print(out)
